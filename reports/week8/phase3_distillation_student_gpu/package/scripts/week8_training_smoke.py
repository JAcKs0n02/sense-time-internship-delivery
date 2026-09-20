#!/usr/bin/env python3
"""Isolated, hash-pinned three-step SFT -> merge -> DPO diagnostic only."""
import argparse
import json
import math
from pathlib import Path
import shutil
import socket
from common import ROOT, sha256, write_json
from train_pipeline import execute_train, export_model

DATA = 'logs/week8-protocol-data-20260914/run-a'
SFT = 'deliverables/week3/day11/configs/experiments/epoch-e5.yaml'
DPO = 'deliverables/week4/day20/remediation/configs/reward_corrective_40step.yaml'
PREF = 'deliverables/week4/day20/remediation/source/data'
RELEASE = 'reports/week8/phase2_final_release/target_receipt.json'
IDENTITY = 'reports/week8/phase2_gpu_320/expected_base_identity.json'
REQUIRED = {SFT, DPO, RELEASE, IDENTITY,
            'reports/week8/phase2_final_release/runtime_lock.json',
            'reports/week8/phase2_final_release/review.json',
            'scripts/week8_training_smoke.py', 'scripts/train_pipeline.py', 'scripts/common.py',
            'reports/week8/phase2_final_release/smoke_plan.json'} | {
    f'{DATA}/{name}' for name in ('statistics.json','protocol.json','dataset_info.json',
                                  'train_sharegpt.json','validation_sharegpt.json')} | {
    f'{PREF}/{name}' for name in ('dataset_info.json','week4_dpo_train_v2.json','week4_dpo_validation_v2.json')}


def verify_files(root, refs):
    if not refs:
        raise ValueError('empty file manifest')
    seen = set()
    for ref in refs:
        name = ref['path']; path = (root/name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(root.resolve()) or name in seen:
            raise ValueError('manifest path escapes root or is duplicated')
        seen.add(name)
        if not path.is_file() or sha256(path) != ref['sha256']:
            raise ValueError(f'file digest mismatch: {name}')
    return seen


def build_configs(root, data, run, base):
    import yaml
    sft = yaml.safe_load((root/SFT).read_text())
    dpo = yaml.safe_load((root/DPO).read_text())
    sft.update(model_name_or_path=str(base),dataset='week8_train',eval_dataset='week8_validation',dataset_dir=str(data))
    dpo.update(model_name_or_path=str(run/'models/smoke_sft'),dataset_dir=str(run/'preference_data'))
    for name, config in [('sft',sft),('dpo',dpo)]:
        config.update(max_steps=3,save_steps=3,logging_steps=1,eval_strategy='no',
                      save_strategy='steps',report_to='none',preprocessing_num_workers=1,
                      dataloader_num_workers=0,overwrite_output_dir=False,
                      resume_from_checkpoint=None,save_only_model=False,plot_loss=False,
                      output_dir=str(run/name/'adapter'),run_name=f'week8-smoke-{name}')
        config.pop('max_samples',None)
        config.pop('eval_steps',None)
        if any(k in config for k in ('adapter_name_or_path','ref_model')):
            raise ValueError('unexpected inherited adapter/reference')
    return sft,dpo


def validate_state(state):
    if type(state.get('global_step')) is not int or state['global_step'] != 3:
        raise ValueError('expected exactly global_step=3')
    rows = [x for x in state.get('log_history',[]) if 'loss' in x]
    if [x.get('step') for x in rows] != [1,2,3]:
        raise ValueError('missing or repeated training steps')
    for row in rows:
        for key in ('loss','grad_norm','learning_rate'):
            v = row.get(key)
            if type(v) not in (int,float) or not math.isfinite(v):
                raise ValueError(f'nonfinite or missing {key}')
    if not any(r['grad_norm'] > 0 for r in rows) or not any(r['learning_rate'] > 0 for r in rows):
        raise ValueError('no positive gradient or learning rate')
    return rows


def verify_adapter(adapter):
    import torch
    from safetensors.torch import load_file
    state = json.loads((adapter/'trainer_state.json').read_text())
    rows = validate_state(state)
    cfg = json.loads((adapter/'adapter_config.json').read_text())
    if cfg.get('peft_type') != 'LORA' or cfg.get('r') != 8 or cfg.get('lora_alpha') != 16:
        raise ValueError('unexpected adapter configuration')
    tensors = load_file(str(adapter/'adapter_model.safetensors'),device='cpu')
    if not tensors or not all(torch.isfinite(t).all().item() for t in tensors.values()):
        raise ValueError('empty or nonfinite adapter tensors')
    b = [t for k,t in tensors.items() if 'lora_B' in k]
    nonzero = sum(bool(torch.count_nonzero(t).item()) for t in b)
    if not nonzero or not any('lora_A' in k for k in tensors):
        raise ValueError('no nonzero LoRA B update')
    return {'global_step':3,'metrics':rows,'tensor_count':len(tensors),
            'nonzero_lora_b_tensors':nonzero,'adapter_sha256':sha256(adapter/'adapter_model.safetensors')}


def verify_optimizer(adapter):
    import torch
    path = adapter/'checkpoint-3/optimizer.pt'
    state = torch.load(path,map_location='cpu',weights_only=True)['state']
    if not state or any(float(v['step']) != 3 for v in state.values()):
        raise ValueError('optimizer has not completed exactly three updates')
    moments = [v['exp_avg'] for v in state.values()]
    if not all(torch.isfinite(t).all().item() for t in moments) or not any(torch.count_nonzero(t).item() for t in moments):
        raise ValueError('optimizer moments invalid or all zero')
    return {'parameter_states':len(state),'step':3,'sha256':sha256(path)}


def verify_merge(folder):
    import torch
    from safetensors import safe_open
    index = folder/'model.safetensors.index.json'
    if not (folder/'config.json').is_file() or not index.is_file():
        raise ValueError('merged model config/index missing')
    weights = json.loads(index.read_text())['weight_map']
    found = set(); files=[]
    for name in sorted(set(weights.values())):
        path = (folder/name).resolve()
        if not path.is_relative_to(folder.resolve()):
            raise ValueError('merged shard escapes model folder')
        with safe_open(str(path),framework='pt',device='cpu') as stream:
            for key in stream.keys():
                if key in found or weights.get(key) != name or not torch.isfinite(stream.get_tensor(key)).all().item():
                    raise ValueError('merged weights invalid')
                found.add(key)
        files.append({'path':name,'sha256':sha256(path)})
    if not weights or found != set(weights):
        raise ValueError('merged weight index incomplete')
    return {'tensors':len(found),'files':files}


def admit(root, lock_path, digest):
    if sha256(lock_path) != digest:
        raise ValueError('smoke lock digest mismatch')
    lock = json.loads(lock_path.read_text())
    if lock.get('scope') != 'SFT3_MERGE_DPO3_ONLY' or lock.get('full_training_allowed') is not False:
        raise ValueError('wrong smoke scope')
    paths = verify_files(root,lock['files'])
    if not REQUIRED <= paths:
        raise ValueError('required smoke inputs not pinned')
    receipt = json.loads((root/RELEASE).read_text())
    if receipt.get('status') != 'TARGET_BASE_EVALUATION_LOCK_RELEASED' or not receipt.get('actual_entrypoint_lock_check_passed'):
        raise ValueError('evaluation release missing')
    runtime_path = root/'reports/week8/phase2_final_release/runtime_lock.json'
    review_path = root/'reports/week8/phase2_final_release/review.json'
    if sha256(runtime_path) != receipt['lock_sha256'] or sha256(review_path) != receipt['review_sha256']:
        raise ValueError('release evidence digest mismatch')
    runtime = json.loads(runtime_path.read_text())
    if lock.get('base_model') != runtime['model_path']:
        raise ValueError('base differs from released model')
    data = root/DATA
    stats = json.loads((data/'statistics.json').read_text())
    if stats.get('mode') != 'protocol_formal_tokenizer':
        raise ValueError('expected frozen protocol data')
    for name,count in [('train_sharegpt.json',1422),('validation_sharegpt.json',158)]:
        if len(json.loads((data/name).read_text())) != count:
            raise ValueError('frozen split count mismatch')
    return lock


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lock',type=Path,required=True)
    p.add_argument('--lock-sha256',required=True)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    lock=admit(ROOT,args.lock,args.lock_sha256)
    run=args.run_dir.resolve()
    if run.exists(): raise ValueError('smoke output already exists; use a fresh directory')
    base=lock['base_model']
    if not args.dry_run:
        import torch
        if socket.gethostname() != 'autodl-container-be044ebe99-be706b14':
            raise ValueError('smoke must run on approved 320 instance')
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError('CUDA BF16 unavailable')
        verify_files(Path(base),json.loads((ROOT/IDENTITY).read_text())['files'])
        if not shutil.which('llamafactory-cli'): raise ValueError('training CLI unavailable')
    run.mkdir(parents=True,exist_ok=False)
    try:
        pref=run/'preference_data'; pref.mkdir()
        for name in ('dataset_info.json','week4_dpo_train_v2.json','week4_dpo_validation_v2.json'):
            shutil.copy2(ROOT/PREF/name,pref/name)
        sft,dpo=build_configs(ROOT,ROOT/DATA,run,base)
        write_json(run/'planned_configs.json',{'sft':sft,'dpo':dpo,'scope':lock['scope']})
        if args.dry_run:
            write_json(run/'status.json',{'status':'SMOKE_PLAN_ONLY','training_started':False,'gpu_checks_performed':False})
            return
        write_json(run/'status.json',{'status':'SMOKE_RUNNING','training_started':True,'full_training_allowed':False})
        model=run/'models/smoke_sft'; model.parent.mkdir()
        results={}
        for stage,config in [('sft',sft),('dpo',dpo)]:
            adapter=execute_train(config,run/stage,'llamafactory-cli',max_attempts=1)
            result=verify_adapter(adapter)
            result['optimizer']=verify_optimizer(adapter)
            results[stage]=result
            write_json(run/f'{stage}_verification.json',result)
            if stage=='sft':
                export_model(base,adapter,model,'llamafactory-cli')
                results['sft_merge']=verify_merge(model)
                write_json(run/'sft_merge_verification.json',results['sft_merge'])
        write_json(run/'status.json',{'status':'SMOKE_PASS_REVIEW_REQUIRED','training_started':True,
                    'full_training_allowed':False,'results':results,'lock_sha256':args.lock_sha256})
    except Exception as exc:
        write_json(run/'status.json',{'status':'SMOKE_FAILED','error':str(exc),'full_training_allowed':False})
        raise

if __name__ == '__main__': main()
