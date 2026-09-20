#!/usr/bin/env python3
"""Hash-bound formal SFT -> merge -> cold load -> DPO -> merge -> cold load."""
import argparse
from contextlib import contextmanager
import re
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

import yaml
from common import ROOT, sha256, write_json
from week8_training_smoke import admit as admit_smoke, verify_files, verify_merge, REQUIRED as SMOKE_INPUTS

SMOKE_LOCK='reports/week8/phase2_training_smoke/smoke_lock.json'
SMOKE_SHA='7c9bc62c7870b5a652eb098a6451319728271902ab280d46eef078769d2be882'
REVIEW='reports/week8/phase2_training_smoke/target_review.json'
CONFIG='configs/week8_full_training_candidate'
TARGET='/root/autodl-tmp/week8-full-training-runtime-20260916'
INSTANCE='be044ebe99-be706b14'
REQUIRED=SMOKE_INPUTS | {SMOKE_LOCK, REVIEW, 'scripts/week8_full_training.py', 'scripts/week8_full_cold_load.py', 'scripts/week8_full_verify.py'} | {f'{CONFIG}/{name}.yaml' for name in ('sft','dpo','export_sft','export_dpo')}


def read(path):
    return json.loads(path.read_text())


def validate_configs(configs):
    for stage in ('sft','dpo'):
        c=configs[stage]
        fixed={'stage':stage,'do_train':True,'cutoff_len':2048,'quantization_bit':4,
               'quantization_method':'bnb','quantization_type':'nf4','double_quantization':True,
               'lora_rank':8,'lora_alpha':16,'lora_dropout':.05,'lora_target':'all','seed':42,
               'bf16':True,'fp16':False,'per_device_train_batch_size':1,
               'per_device_eval_batch_size':1,'overwrite_output_dir':False,
               'resume_from_checkpoint':None,'save_only_model':False,'load_best_model_at_end':False,
               'report_to':'none','logging_steps':1,'save_strategy':'steps'}
        fixed.update({'gradient_accumulation_steps':4,'learning_rate':1e-4,'num_train_epochs':5.,
                      'max_steps':-1,'lr_scheduler_type':'cosine','warmup_ratio':.1,
                      'save_steps':500,'save_total_limit':2,'eval_strategy':'epoch'} if stage=='sft' else
                     {'gradient_accumulation_steps':8,'learning_rate':2e-6,'max_steps':40,
                      'lr_scheduler_type':'constant_with_warmup','warmup_steps':29,
                      'save_steps':40,'save_total_limit':1,'eval_strategy':'steps','eval_steps':20,
                      'pref_beta':.1,'pref_loss':'sigmoid'})
        for k,v in fixed.items():
            if k not in c or c[k]!=v:raise ValueError(f'formal configuration mismatch: {stage}.{k}')
        if any(k in c for k in ('max_samples','adapter_name_or_path','ref_model')):
            raise ValueError('inherited truncation/adapter/reference is forbidden')
    return configs


def load_configs(root):
    return {name:yaml.safe_load((root/CONFIG/f'{name}.yaml').read_text())
            for name in ('sft','dpo','export_sft','export_dpo')}


def build_release(root):
    lock=admit_smoke(root,root/SMOKE_LOCK,SMOKE_SHA)
    paths={x['path'] for x in lock['files']}|REQUIRED
    validate_configs(load_configs(root))
    return {'schema':1,'scope':'FORMAL_SFT5_DPO40_TWO_MERGES_COLD_LOADS',
            'execution_enabled':False,'instance_id':INSTANCE,'target_root':TARGET,
            'run_dir':TARGET+'/logs/full-run-01','expected_steps':{'sft':1775,'dpo':40},
            'resource_policy':{'quality_first':True,'reduce_training_for_price':False,
                               'shutdown_when_idle_or_failed':True,'wall_time_guard_seconds':10800,
                               'guard_is_completion_target':False},
            'files':[{'path':p,'sha256':sha256(root/p)} for p in sorted(paths)]}


def admit_release(root,path,digest,dry_run):
    if sha256(path)!=digest:raise ValueError('release digest mismatch')
    r=read(path)
    if r.get('schema')!=1 or r.get('scope')!='FORMAL_SFT5_DPO40_TWO_MERGES_COLD_LOADS':
        raise ValueError('invalid formal release scope')
    paths=verify_files(root,r['files'])
    if not REQUIRED<=paths:raise ValueError('required formal inputs not pinned')
    lock=admit_smoke(root,root/SMOKE_LOCK,SMOKE_SHA)
    if not {x['path'] for x in lock['files']}<=paths:raise ValueError('smoke dependencies missing')
    if read(root/REVIEW).get('status')!='GPU_SMOKE_PASS_LOCAL_REVIEWED':raise ValueError('smoke not reviewed')
    if r.get('instance_id')!=INSTANCE or r.get('target_root')!=TARGET or r.get('run_dir')!=TARGET+'/logs/full-run-01':
        raise ValueError('wrong execution target')
    if r.get('expected_steps')!={'sft':1775,'dpo':40}:raise ValueError('wrong formal step targets')
    cfg=validate_configs(load_configs(root));run=Path(r['run_dir'])
    for stage in ('sft','dpo'):
        base=lock['base_model'] if stage=='sft' else str(run/'models/final_sft')
        c=cfg[stage];e=cfg['export_'+stage]
        if c['model_name_or_path']!=base or c['output_dir']!=str(run/stage/'attempt-01/adapter'):
            raise ValueError('training lineage/output mismatch')
        data=TARGET+'/logs/week8-protocol-data-20260914/run-a' if stage=='sft' else TARGET+'/deliverables/week4/day20/remediation/source/data'
        if c['dataset_dir']!=data:raise ValueError('unexpected dataset directory')
        if c['dataset']!=('week8_train' if stage=='sft' else 'week4_dpo_train_v2') or c['eval_dataset']!=('week8_validation' if stage=='sft' else 'week4_dpo_validation_v2'):
            raise ValueError('dataset registration mismatch')
        expected_export={'model_name_or_path':base,'adapter_name_or_path':c['output_dir'],
                         'template':'qwen','finetuning_type':'lora','export_dir':str(run/f'models/final_{stage}'),
                         'export_size':5,'export_device':'cpu','export_legacy_format':False}
        if e!=expected_export:raise ValueError('merge lineage/config mismatch')
    policy=r.get('resource_policy',{})
    guard=policy.get('wall_time_guard_seconds')
    if policy.get('quality_first') is not True or policy.get('reduce_training_for_price') is not False:
        raise ValueError('quality policy missing')
    if type(guard) is not int or guard<=0:raise ValueError('finite wall-time protection required')
    if not dry_run:
        if r.get('execution_enabled') is not True:raise ValueError('formal execution not released')
        expiry=datetime.fromisoformat(r['session']['shutdown_deadline_utc'])
        now=datetime.now(timezone.utc)
        if expiry.tzinfo is None or not 0<(expiry-now).total_seconds()<=guard:
            raise ValueError('shutdown protection expired or exceeds session guard')
        if r['session'].get('shutdown_confirmed') is not True:raise ValueError('shutdown protection not confirmed')
        if root.resolve()!=Path(TARGET):raise ValueError('wrong deployed root')
    return r,cfg


def validate_formal_state(state,expected_steps):
    if type(state.get('global_step')) is not int or state['global_step']!=expected_steps or state.get('max_steps')!=expected_steps:
        raise ValueError('incomplete or unexpected formal steps')
    rows=[r for r in state.get('log_history',[]) if 'loss' in r]
    if [r.get('step') for r in rows]!=list(range(1,expected_steps+1)):
        raise ValueError('missing/duplicated formal steps')
    for row in rows:
        if any(not math.isfinite(v) for v in row.values() if type(v) in (int,float)):
            raise ValueError('nonfinite supplementary training metric')
        for key in ('loss','grad_norm','learning_rate'):
            v=row.get(key)
            if type(v) not in (int,float) or not math.isfinite(v):raise ValueError('nonfinite/missing training metric')
        if row['grad_norm']<0 or row['learning_rate']<0:raise ValueError('negative gradient norm/rate')
    if not any(r['grad_norm']>0 for r in rows) or not any(r['learning_rate']>0 for r in rows):
        raise ValueError('no effective training updates')
    return rows


def verify_formal_adapter(adapter,expected_steps,base,stage):
    import torch
    from safetensors.torch import load_file
    state=read(adapter/'trainer_state.json');rows=validate_formal_state(state,expected_steps)
    evaluations=[x for x in state['log_history'] if 'eval_loss' in x]
    if len(evaluations)!=(5 if stage=='sft' else 2):raise ValueError('validation records missing')
    steps=[r['step'] for r in evaluations]
    if steps!=sorted(set(steps)) or any(not 0<s<=expected_steps for s in steps):raise ValueError('invalid validation steps')
    if stage=='dpo' and steps!=[20,40]:raise ValueError('unexpected DPO validation cadence')
    if any(type(r['eval_loss']) not in (int,float) or not math.isfinite(r['eval_loss']) for r in evaluations):raise ValueError('nonfinite validation loss')
    cfg=read(adapter/'adapter_config.json')
    if any(cfg.get(k)!=v for k,v in {'peft_type':'LORA','r':8,'lora_alpha':16,'base_model_name_or_path':str(base)}.items()):
        raise ValueError('adapter config/base mismatch')
    tensors=load_file(str(adapter/'adapter_model.safetensors'),device='cpu')
    if not tensors or not all(torch.isfinite(t).all().item() for t in tensors.values()):raise ValueError('nonfinite adapter')
    bs=[t for k,t in tensors.items() if 'lora_B' in k]
    nz=sum(bool(torch.count_nonzero(t).item()) for t in bs)
    if not nz or not any('lora_A' in k for k in tensors):raise ValueError('no actual LoRA update')
    opt=adapter/f'checkpoint-{expected_steps}/optimizer.pt'
    states=torch.load(opt,map_location='cpu',weights_only=True)['state']
    if len(states)!=len(tensors) or not states:raise ValueError('optimizer parameter state missing')
    for v in states.values():
        if float(v['step'])!=expected_steps:raise ValueError('wrong optimizer step')
        if not all(torch.isfinite(v[k]).all().item() for k in ('exp_avg','exp_avg_sq')):raise ValueError('nonfinite optimizer moment')
    if not any(torch.count_nonzero(v['exp_avg']).item() for v in states.values()):raise ValueError('zero optimizer moments')
    return {'global_step':expected_steps,'first_loss':rows[0]['loss'],'last_loss':rows[-1]['loss'],
            'validation':evaluations,'tensor_count':len(tensors),'nonzero_lora_b_tensors':nz,
            'optimizer_step':expected_steps,'adapter_sha256':sha256(adapter/'adapter_model.safetensors'),
            'optimizer_sha256':sha256(opt)}


def stop_group(proc):
    try:os.killpg(proc.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:proc.wait(timeout=2)
    except subprocess.TimeoutExpired:pass
    try:os.killpg(proc.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    proc.wait()


@contextmanager
def session_timeout(deadline):
    remaining=deadline-time.monotonic()
    if remaining<=0:raise TimeoutError('session deadline expired')
    old_handler=signal.getsignal(signal.SIGALRM)
    old_timer=signal.getitimer(signal.ITIMER_REAL)
    started=time.monotonic()
    def timeout(signum,frame):raise TimeoutError('protected session deadline reached')
    signal.signal(signal.SIGALRM,timeout)
    signal.setitimer(signal.ITIMER_REAL,remaining)
    try:yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,old_handler)
        if old_timer[0]>0:signal.setitimer(signal.ITIMER_REAL,max(.001,old_timer[0]-(time.monotonic()-started)),old_timer[1])


def run_command(command,log,cwd,deadline,training_steps=None,training_samples=None):
    if deadline<=time.monotonic():raise TimeoutError('session deadline reached before stage')
    receipt={'command':command,'started_utc':datetime.now(timezone.utc).isoformat(),
             'returncode':None,'timed_out':False,'header_verified':training_steps is None}
    proc=None
    with log.open('x') as stream:
        try:
            proc=subprocess.Popen(command,cwd=cwd,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            while True:
                remaining=deadline-time.monotonic()
                if remaining<=0:raise TimeoutError('stage exceeded protected session')
                try:code=proc.wait(timeout=min(.2,remaining))
                except subprocess.TimeoutExpired:code=None
                if training_steps is not None and not receipt['header_verified']:
                    text=log.read_text(errors='replace')
                    matches=re.findall(r'Total optimization steps\s*=\s*([0-9,]+)',text)
                    if matches and any(int(x.replace(',',''))!=training_steps for x in matches):
                        raise RuntimeError('Trainer step target differs from released plan')
                    # Evaluation uses the same field name; inspect only the startup header.
                    header=text.split('***** Running training *****',1)[-1].split('Total optimization steps',1)[0]
                    counts=re.findall(r'Num examples\s*=\s*([0-9,]+)',header)
                    if training_samples is not None and counts and any(int(x.replace(',',''))!=training_samples for x in counts):
                        raise RuntimeError('Trainer sample count differs from released plan')
                    receipt['header_verified']=bool(matches) and (training_samples is None or bool(counts))
                if code is not None:
                    receipt['returncode']=code
                    if code:raise RuntimeError(f'command failed with exit {code}: {log}')
                    if not receipt['header_verified']:raise RuntimeError('Trainer header absent; refuse acceptance')
                    break
        except BaseException as exc:
            receipt['timed_out']=isinstance(exc,TimeoutError)
            receipt['error']=str(exc)
            if proc is not None:stop_group(proc)
            raise
        finally:
            if proc is not None:receipt['returncode']=proc.returncode
            receipt['finished_utc']=datetime.now(timezone.utc).isoformat()
            stream.flush()
            receipt['log_sha256']=sha256(log)
            write_json(log.with_name(log.name+'.process.json'),receipt)
    return receipt


def verify_cold_receipt(path,model):
    r=read(path)
    if r.get('status')!='COLD_LOAD_PASS' or r.get('model_path')!=str(model.resolve()) or r.get('finite_logits') is not True:
        raise ValueError('cold-load receipt mismatch')
    if type(r.get('generated_tokens')) is not int or not 1<=r['generated_tokens']<=2 or r.get('dtype')!='torch.bfloat16' or r.get('device')!='cuda:0':
        raise ValueError('cold-load inference did not meet requirements')
    return r


def target_preflight(root,release):
    import torch
    if socket.gethostname()!='autodl-container-'+INSTANCE:raise ValueError('wrong target host')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise ValueError('CUDA BF16 unavailable')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=10)
    if any(int(p.strip())!=os.getpid() for p in apps.splitlines() if p.strip()):raise ValueError('GPU in use by another process')
    if shutil.disk_usage(root).free<45*1024**3:raise ValueError('less than 45GiB free disk')
    if not shutil.which('llamafactory-cli'):raise ValueError('training CLI unavailable')
    import importlib.metadata
    versions={p:importlib.metadata.version(p) for p in ('torch','transformers','llamafactory','peft','trl','accelerate','datasets','bitsandbytes')}
    pinned={'torch':'2.5.1+cu121','transformers':'4.50.0','llamafactory':'0.9.3','peft':'0.15.1','trl':'0.9.6','accelerate':'1.2.1','datasets':'3.2.0','bitsandbytes':'0.43.3'}
    if versions!=pinned:raise ValueError('target dependency drift')
    base=read(root/SMOKE_LOCK)['base_model']
    verify_files(Path(base),read(root/'reports/week8/phase2_gpu_320/expected_base_identity.json')['files'])
    mem=Path('/sys/fs/cgroup/memory.max')
    if not mem.exists() or mem.read_text().strip()=='max' or int(mem.read_text())<60*1024**3:raise ValueError('unexpected memory limit')
    return {'hostname':socket.gethostname(),'versions':versions,'base_files_verified':15,
            'disk_free_bytes':shutil.disk_usage(root).free,'memory_limit_bytes':int(mem.read_text()),'cuda_bf16':True}


def execute_stages(root,run,configs,release,deadline,cli='llamafactory-cli'):
    results={};phase='initial'
    try:
        for stage in ('sft','dpo'):
            phase=stage;folder=run/stage/'attempt-01';folder.mkdir(parents=True,exist_ok=False)
            write_json(run/'status.json',{'status':'FORMAL_RUNNING','phase':phase,'training_started':True})
            command=[cli,'train',str(root/CONFIG/f'{stage}.yaml')]
            run_command(command,folder/'train.log',root,deadline,training_steps=release['expected_steps'][stage],training_samples=1422 if stage=='sft' else 783)
            phase=stage+'_verify'
            write_json(run/'status.json',{'status':'FORMAL_RUNNING','phase':phase,'training_started':True})
            worker=root/'scripts/week8_full_verify.py'
            verification=run/f'{stage}_verification.json'
            run_command([sys.executable,str(worker),'adapter','--path',str(folder/'adapter'),'--output',str(verification),'--steps',str(release['expected_steps'][stage]),'--base',configs[stage]['model_name_or_path'],'--stage',stage],run/f'{phase}.log',root,deadline)
            results[stage]=read(verification)
            phase=stage+'_merge';model=run/f'models/final_{stage}';model.parent.mkdir(exist_ok=True)
            write_json(run/'status.json',{'status':'FORMAL_RUNNING','phase':phase,'training_started':True})
            if model.exists():raise ValueError('refuse existing merged model')
            run_command([cli,'export',str(root/CONFIG/f'export_{stage}.yaml')],run/f'{phase}.log',root,deadline)
            phase=stage+'_merge_verify'
            write_json(run/'status.json',{'status':'FORMAL_RUNNING','phase':phase,'training_started':True})
            merged=run/f'{stage}_merge_verification.json'
            run_command([sys.executable,str(worker),'merge','--path',str(model),'--output',str(merged)],run/f'{phase}.log',root,deadline)
            results[stage+'_merge']=read(merged)
            phase=stage+'_cold_load';receipt=run/f'{phase}.json'
            write_json(run/'status.json',{'status':'FORMAL_RUNNING','phase':phase,'training_started':True})
            run_command([sys.executable,str(root/'scripts/week8_full_cold_load.py'),'--model',str(model),'--output',str(receipt)],run/f'{phase}.log',root,deadline)
            results[phase]=verify_cold_receipt(receipt,model)
        if time.monotonic()>=deadline:raise TimeoutError('deadline reached before final acceptance')
        write_json(run/'status.json',{'status':'FORMAL_TRAINING_PASS_REVIEW_REQUIRED','training_started':True,
                   'quality_evaluated':False,'formal_evaluation_allowed':False,'results':results})
    except BaseException as exc:
        write_json(run/'status.json',{'status':'INCOMPLETE_TIMEOUT' if isinstance(exc,TimeoutError) else 'FORMAL_FAILED',
                   'phase':phase,'error':str(exc),'automatic_retry':False,'formal_evaluation_allowed':False})
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,required=True);p.add_argument('--release-sha256',required=True)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();release,cfg=admit_release(ROOT,a.release,a.release_sha256,a.dry_run)
    run=a.run_dir.resolve()
    if run.exists():raise ValueError('refuse existing output directory')
    if not a.dry_run and run!=Path(release['run_dir']):raise ValueError('run directory differs from release')
    if a.dry_run:
        run.mkdir(parents=True,exist_ok=False)
        write_json(run/'planned_configs.json',cfg)
        write_json(run/'status.json',{'status':'FORMAL_PLAN_ONLY','training_started':False,'gpu_checks_performed':False})
        return
    os.environ.update(OMP_NUM_THREADS='14',MKL_NUM_THREADS='14')
    shutdown=datetime.fromisoformat(release['session']['shutdown_deadline_utc'])
    remaining=(shutdown-datetime.now(timezone.utc)).total_seconds()-180
    if remaining<=0:raise TimeoutError('insufficient time before protected shutdown')
    deadline=time.monotonic()+min(remaining,release['resource_policy']['wall_time_guard_seconds'])
    run.mkdir(parents=True,exist_ok=False)
    try:
        with session_timeout(deadline):preflight=target_preflight(ROOT,release)
        write_json(run/'target_preflight.json',preflight)
    except BaseException as exc:
        write_json(run/'status.json',{'status':'PREFLIGHT_FAILED','error':str(exc),'training_started':False})
        raise
    write_json(run/'release.json',release);write_json(run/'planned_configs.json',cfg)
    def interrupted(signum,frame):raise InterruptedError(f'signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    execute_stages(ROOT,run,cfg,release,deadline)

if __name__=='__main__':main()
