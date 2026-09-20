"""Real LLaMA-Factory CPU preprocessing and independent Qwen supervision audit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from common import ROOT, sha256, write_json


def assert_supervision(expected, actual):
    for key in ['input_ids','labels','attention_mask']:
        if actual[key] != expected[key]:
            raise ValueError(f'{key} differs from independent frozen expectation')
        if not actual[key] or any(type(v) is not int for v in actual[key]):
            raise ValueError(f'invalid {key}')
    if len({len(actual[k]) for k in ['input_ids','labels','attention_mask']}) != 1:
        raise ValueError('unequal array lengths')
    for start,end in expected['assistant_spans']:
        if expected['eos_token_id'] not in actual['labels'][start:end]:
            raise ValueError('assistant EOS not supervised')
    supervised=sum(t!=-100 for t in actual['labels'])
    if not supervised:
        raise ValueError('no assistant supervision')
    return {'supervised_tokens':supervised,'assistant_turns':len(expected['assistant_spans'])}


def expected_tokens(row, tokenizer, cutoff=2048):
    """Independent explicit ChatML renderer: never calls LF template/processor helpers."""
    system=row.get('system') or 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'
    turns=row['conversations'];ids=[];labels=[];spans=[]
    if not turns or len(turns)%2:
        raise ValueError('invalid source turns')
    messages=([{'role':'system','content':row['system']}] if row.get('system') else [])
    for n in range(0,len(turns),2):
        user,assistant=turns[n:n+2]
        if user['from']!='human' or assistant['from']!='gpt':raise ValueError('invalid source roles')
        prompt=(f'<|im_start|>system\n{system}<|im_end|>\n' if n==0 else '')
        prompt+=f'<|im_start|>user\n{user["value"]}<|im_end|>\n<|im_start|>assistant\n'
        response=assistant['value']+'<|im_end|>\n'
        p=tokenizer.encode(prompt,add_special_tokens=False);a=tokenizer.encode(response,add_special_tokens=False)
        start=len(ids)+len(p);ids+=p+a;labels += [-100]*len(p)+a;spans.append([start,len(ids)])
        messages += [{'role':'user','content':user['value']},{'role':'assistant','content':assistant['value']}]
    full=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=False)
    if ids!=full:raise ValueError('independent span tokenization differs from frozen full chat template')
    if len(ids)>cutoff:raise ValueError('untruncated original exceeds cutoff')
    return {'input_ids':ids,'labels':labels,'attention_mask':[1]*len(ids),'assistant_spans':spans,'eos_token_id':tokenizer.eos_token_id}


def stage_tokenizer(protocol, asset):
    """Bind original model config to avoid AutoConfig guessing from directory names."""
    for entry in protocol['tokenizer_files']:
        src=ROOT/entry['path']
        if sha256(src)!=entry['sha256']:raise ValueError('tokenizer asset changed')
        shutil.copy2(src,asset/src.name)
    config=ROOT/'deliverables/week1/day3/source/config/config.json'
    expected='7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c'
    if sha256(config)!=expected or json.loads(config.read_text())['model_type']!='qwen2':
        raise ValueError('original Qwen architecture config mismatch')
    shutil.copy2(config,asset/'config.json')
    return {'config_source':str(config.relative_to(ROOT)),'config_sha256':expected,'model_type':'qwen2','weights_verified':False,'adapter_metadata_copied':False}


def run(args):
    out=args.output_dir.resolve();data=args.data_dir.resolve()
    if out.exists():raise ValueError('use a new audit output directory')
    stats=json.loads((data/'statistics.json').read_text());protocol=json.loads((data/'protocol.json').read_text())
    if sha256(ROOT/'configs/week8_data_protocol.json')!=stats['protocol_sha256']:raise ValueError('current protocol changed')
    for name,h in stats['outputs'].items():
        if sha256(data/name)!=h:raise ValueError('data artifact changed: '+name)
    review=json.loads((ROOT/'reports/week8/phase1_task4/verification.json').read_text())
    if review['status']!='TASK4_SAMPLE_REVIEW_PASS_NOT_DATA_READY' or review['blocking_issues']:
        raise ValueError('content review not complete')
    for name,h in review['snapshot_hashes'].items():
        if sha256(data/name)!=h:raise ValueError('content review does not bind current data')
    out.mkdir(parents=True)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',HF_DATASETS_CACHE=str(out/'cache/datasets'),HF_HOME=str(out/'cache/hf'),TOKENIZERS_PARALLELISM='false',WANDB_DISABLED='true',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2')
    # Copy only the five pinned tokenizer files, never adapter metadata or model weights.
    asset=out/'tokenizer';asset.mkdir()
    config_identity=stage_tokenizer(protocol,asset)
    write_json(out/'config_identity.json',config_identity)
    import importlib.metadata as metadata
    import platform
    import torch,transformers,yaml,llamafactory
    versions={k:metadata.version(k) for k in ['torch','transformers','llamafactory','peft','accelerate','datasets','trl','tokenizers']}
    if versions['llamafactory']!='0.9.3' or versions['transformers']!=protocol['transformers_version']:raise ValueError('unapproved framework version')
    environment={'python':sys.version,'python_executable':sys.executable,'platform':platform.platform(),'versions':versions,'cpu_only':True,'cuda_available':torch.cuda.is_available(),'weight_loading_called':False,'training_called':False,'framework_path':llamafactory.__file__}
    write_json(out/'environment.json',environment)
    frozen=transformers.AutoTokenizer.from_pretrained(asset,local_files_only=True,use_fast=True)
    if hashlib.sha256(frozen.chat_template.encode()).hexdigest()!=protocol['chat_template_sha256']:raise ValueError('frozen chat template changed')
    if transformers.AutoConfig.from_pretrained(asset,local_files_only=True).model_type != 'qwen2':
        raise ValueError('runtime inferred wrong architecture')
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_tokenizer
    from llamafactory.data import get_dataset,get_template_and_fix_tokenizer
    base=yaml.safe_load((ROOT/'deliverables/week3/day11/configs/experiments/epoch-e5.yaml').read_text())
    # Inherit the SFT recipe while making CPU-only, no-training audit overrides explicit.
    config=dict(base)
    config.pop('max_samples',None)
    config.update(model_name_or_path=str(asset),trust_remote_code=False,quantization_bit=None,bf16=False,fp16=False,use_cpu=True,
        dataset='week8_train',eval_dataset='week8_validation',dataset_dir=str(data),
        template='qwen',cutoff_len=2048,train_on_prompt=False,mask_history=False,packing=False,
        val_size=0.,seed=42,data_seed=42,preprocessing_num_workers=1,dataloader_num_workers=0,
        overwrite_cache=True,cache_dir=str(out/'cache/model'),report_to='none',plot_loss=False,
        output_dir=str(out/'unused_trainer'),overwrite_output_dir=False,eval_strategy='epoch')
    write_json(out/'audit_overrides.json',{'base_config':str((ROOT/'deliverables/week3/day11/configs/experiments/epoch-e5.yaml').relative_to(ROOT)),'base_sha256':sha256(ROOT/'deliverables/week3/day11/configs/experiments/epoch-e5.yaml'),'removed':['max_samples'],'overrides':{k:v for k,v in config.items() if k not in base or base[k]!=v},'do_train_note':'do_train=true selects the SFT preprocessing branch; no Trainer or model is constructed'})
    (out/'effective_sft_config.yaml').write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False))
    def load(cfg):
        ma,da,ta,fa,_=get_train_args(cfg)
        if str(ta.device)!='cpu':raise ValueError('audit did not select CPU')
        tm=load_tokenizer(ma);tpl=get_template_and_fix_tokenizer(tm['tokenizer'],da)
        if tpl.efficient_eos or tm['tokenizer'].eos_token_id!=frozen.eos_token_id:raise ValueError('template/EOS contract differs')
        ds=get_dataset(tpl,ma,da,ta,stage='sft',**tm)
        return ds,tm['tokenizer'],tpl
    datasets,tok,template=load(config)
    lineage=json.loads((data/'lineage.json').read_text());summaries={};actuals={};selected=[]
    for split,key in [('train','train_dataset'),('validation','eval_dataset')]:
        source=json.loads((data/f'{split}_sharegpt.json').read_text());ds=datasets[key]
        if isinstance(ds,dict):raise ValueError('unexpected evaluation dataset mapping')
        origins=sorted([r for r in lineage if r['split']==split],key=lambda r:r['row_index'])
        if len(ds)!=len(source) or len(origins)!=len(ds):raise ValueError('loader dropped or added rows')
        summary={'records':len(ds),'supervised_tokens':0,'assistant_turns':0,'multiturn_records':0,'max_tokens':0,'mismatches':0}
        lines=[];actuals[split]=[]
        for n,(row,origin) in enumerate(zip(source,origins)):
            expected=expected_tokens(row,frozen);actual={k:ds[n][k] for k in ['input_ids','labels','attention_mask']}
            details=assert_supervision(expected,actual)
            if len(expected['input_ids'])!=origin['tokens']:raise ValueError('task3 token length changed')
            summary['supervised_tokens']+=details['supervised_tokens'];summary['assistant_turns']+=details['assistant_turns']
            summary['multiturn_records']+=details['assistant_turns']>1;summary['max_tokens']=max(summary['max_tokens'],len(actual['input_ids']))
            result={'sample_id':origin['sample_id'],'row_index':n,'split':split,'assistant_spans':expected['assistant_spans'],'actual':actual,'supervised_tokens':details['supervised_tokens']}
            lines.append(result);actuals[split].append(actual)
            if details['assistant_turns']>1 or len(actual['input_ids'])>=1900 or n==0:selected.append(result)
        (out/f'{split}_token_rows.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in lines))
        summaries[split]=summary
    # Run the real Alpaca converter on the same delivered arrays and compare every token/label.
    alt=out/'alpaca';alt.mkdir();info={}
    for split in ['train','validation']:
        name=f'{split}_alpaca.json';shutil.copy2(data/name,alt/name)
        info[f'week8_{split}']={'file_name':name,'formatting':'alpaca','columns':{'prompt':'instruction','query':'input','response':'output','history':'history','system':'system'}}
    write_json(alt/'dataset_info.json',info)
    alt_config=dict(config,dataset_dir=str(alt));write_json(out/'alpaca_config.json',alt_config)
    al,_,_=load(alt_config)
    for split,key in [('train','train_dataset'),('validation','eval_dataset')]:
        if len(al[key])!=len(actuals[split]):raise ValueError('Alpaca row count mismatch')
        for n,expected in enumerate(actuals[split]):
            if any(al[key][n][k]!=expected[k] for k in expected):raise ValueError(f'Alpaca loader mismatch: {split}:{n}')
    # Synthetic explicit-system and multi-turn probes are separate from production rows.
    probe=out/'synthetic';probe.mkdir()
    sample={'system':'只返回中文答案。','conversations':[{'from':'human','value':'给出数字2。'},{'from':'gpt','value':'2'},{'from':'human','value':'再加3。'},{'from':'gpt','value':'5'}]}
    write_json(probe/'probe.json',[sample]);write_json(probe/'dataset_info.json',{'probe':{'file_name':'probe.json','formatting':'sharegpt','columns':{'messages':'conversations','system':'system'},'tags':{'role_tag':'from','content_tag':'value','user_tag':'human','assistant_tag':'gpt'}}})
    pc=dict(config,dataset='probe',eval_dataset=None,eval_strategy='no',dataset_dir=str(probe));pd,_,_=load(pc)
    exp=expected_tokens(sample,frozen);act={k:pd['train_dataset'][0][k] for k in ['input_ids','labels','attention_mask']};probe_result=assert_supervision(exp,act)
    write_json(out/'synthetic_receipt.json',{'status':'PASS','production_sample':False,**probe_result,'actual':act,'assistant_spans':exp['assistant_spans']})
    # Readable token-span views use the real returned labels, including EOS and whitespace.
    views=[]
    for row in selected:
        views.append({'sample_id':row['sample_id'],'split':row['split'],'row_index':row['row_index'],'full_text':tok.decode(row['actual']['input_ids'],skip_special_tokens=False),'assistant_supervised_texts':[tok.decode(row['actual']['input_ids'][a:b],skip_special_tokens=False) for a,b in row['assistant_spans']]})
    write_json(out/'representative_spans.json',views)
    package_root=Path(llamafactory.__file__).parent
    code_paths=['data/loader.py','data/converter.py','data/template.py','data/processor/supervised.py','data/processor/processor_utils.py','hparams/parser.py','model/loader.py']
    write_json(out/'framework_code_sha256.json',{p:sha256(package_root/p) for p in code_paths})
    for name,h in stats['outputs'].items():
        if sha256(data/name)!=h:raise ValueError('data mutated during audit')
    receipt={'status':'TASK5_REAL_LF_CPU_LOADER_PASS_NOT_DATA_READY','protocol_id':protocol['protocol_id'],'protocol_sha256':stats['protocol_sha256'],'source_data_dir':str(data.relative_to(ROOT)),'data_hashes':{p.name:sha256(p) for p in data.glob('*.json')},'environment':environment,'architecture_config':config_identity,'splits':summaries,'sharegpt_alpaca_identical_records':sum(len(a) for a in actuals.values()),'frozen_template_exact_match_all_rows':True,'label_shift':'unshifted labels; causal loss shift belongs to later model forward, not this audit','eos_token_id':frozen.eos_token_id,'eos_token':frozen.eos_token,'synthetic_system_multiturn_pass':True,'actual_linux_cuda_environment_verified':False,'model_weights_loaded':False,'trainer_constructed':False,'training_started':False,'pending':['Linux_GPU_environment_recheck_before_training','benchmark_and_judge_runtime_lock','final_data_release']}
    write_json(out/'verification.json',receipt)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    try:
        result=run(a);print(json.dumps(result,ensure_ascii=False,indent=2))
    except Exception as exc:
        if a.output_dir.exists():write_json(a.output_dir/'failure.json',{'status':'FAIL','error':str(exc),'training_started':False})
        raise

if __name__=='__main__':main()
