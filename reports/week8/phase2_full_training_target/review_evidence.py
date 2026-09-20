"""Read-only local audit of downloaded formal-training evidence; no GPU/API use."""
import json
import math
from pathlib import Path
import sys
import yaml
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
from common import sha256,write_json
from week8_full_training import admit_release,validate_formal_state
here=Path(__file__).resolve().parent
e=here/'evidence';run=e/'logs/full-run-01'
def read(p):return json.loads(p.read_text())
assert sha256(here/'formal_training_evidence.tar.gz')=='51c3ebf6c4829b024f074146d1646965d5a31312ddc0c93d70bd9cee727cdd6c'
assert sha256(e/'evidence_manifest.json')=='9b8b902833affe2d9c308545a166661d2857766dba0a30f78d5331056d6fbb0d'
manifest=read(e/'evidence_manifest.json');names=set()
for item in manifest['files']:
 p=(e/item['path']).resolve()
 assert p.is_relative_to(e.resolve()) and item['path'] not in names
 names.add(item['path']);assert p.stat().st_size==item['bytes'] and sha256(p)==item['sha256']
assert len(names)==90 and manifest['final_gpu_processes']==[]
release,cfg=admit_release(ROOT,here/'release.json','c43eebb6b58e80d4be42d9535ea9257fbf6101d0f1d3a4ecd61041489c536c28',True)
assert read(e/'release.json')==release==read(run/'release.json')
assert read(run/'planned_configs.json')==cfg
for name,c in cfg.items():assert yaml.safe_load((e/'configs/week8_full_training_candidate'/f'{name}.yaml').read_text())==c
status=read(run/'status.json');assert status['status']=='FORMAL_TRAINING_PASS_REVIEW_REQUIRED'
assert not status['quality_evaluated'] and not status['formal_evaluation_allowed']
pre=read(run/'target_preflight.json');assert {k:v for k,v in pre.items() if k!='disk_free_bytes'}=={k:v for k,v in read(e/'deployment_preflight.json').items() if k!='disk_free_bytes'}
assert pre['base_files_verified']==15 and pre['cuda_bf16'] and pre['disk_free_bytes']>=45*1024**3
assert pre['hostname']=='autodl-container-be044ebe99-be706b14'
receipts=list(run.rglob('*.process.json'));assert len(receipts)==10
for p in receipts:
 x=read(p);log=p.with_name(p.name.removesuffix('.process.json'))
 assert x['returncode']==0 and not x['timed_out'] and x['header_verified']
 assert sha256(log)==x['log_sha256']
summary={}
for stage,n in [('sft',1775),('dpo',40)]:
 a=run/stage/'attempt-01/adapter';state=read(a/'trainer_state.json');rows=validate_formal_state(state,n)
 v=read(run/f'{stage}_verification.json');assert v==status['results'][stage]
 assert v['global_step']==v['optimizer_step']==n
 assert v['tensor_count']==392 and v['nonzero_lora_b_tensors']==196
 assert v['first_loss']==rows[0]['loss'] and v['last_loss']==rows[-1]['loss']
 ev=[x for x in state['log_history'] if 'eval_loss' in x];assert ev==v['validation']
 assert len(ev)==(5 if stage=='sft' else 2)
 for row in ev:assert all(math.isfinite(x) for x in row.values() if type(x) in (int,float))
 assert read(a/f'checkpoint-{n}/trainer_state.json')['global_step']==n
 ac=read(a/'adapter_config.json');assert ac['base_model_name_or_path']==cfg[stage]['model_name_or_path']
 assert ac['r']==8 and ac['lora_alpha']==16
 merge=read(run/f'{stage}_merge_verification.json');assert merge==status['results'][stage+'_merge']
 assert merge['tensors']==339 and len(merge['files'])==4
 identities={x['path']:x['sha256'] for x in merge['identity']}
 for f in merge['files']:assert identities[f['path']]==f['sha256']
 for name in ['config.json','generation_config.json']:assert sha256(run/'models'/f'final_{stage}'/name)==identities[name]
 cold=read(run/f'{stage}_cold_load.json');assert cold==status['results'][stage+'_cold_load']
 assert cold['status']=='COLD_LOAD_PASS' and cold['finite_logits'] and cold['generated_tokens']==2
 assert cold['model_path']==release['run_dir']+f'/models/final_{stage}'
 assert cold['dtype']=='torch.bfloat16' and cold['device']=='cuda:0' and not cold['benchmark_scored']
 summary[stage]={'steps':n,'first_training_loss':v['first_loss'],'last_training_loss':v['last_loss'],'validation':ev,'merged_tensors':339,'merged_shards':4,'cold_load':'PASS','optimizer_step_verified_on_target':n,'adapter_sha256':v['adapter_sha256'],'optimizer_sha256':v['optimizer_sha256']}
assert read(run/'sft_cold_load.log.process.json')['finished_utc']<read(run/'dpo/attempt-01/train.log.process.json')['started_utc']
result={'status':'FORMAL_TRAINING_PASS_LOCAL_REVIEWED','evidence_files_verified':90,'frozen_inputs_rechecked':40,'successful_stage_processes':10,'stages':summary,'final_gpu_processes':[],'quality_evaluated':False,'formal_evaluation_allowed':False,'limits':['Binary adapter/optimizer/merged weights verified on target and remain there; local audit checks raw logs, metadata and receipts.','Rising SFT validation loss signals possible overfitting; no quality-improvement claim.','New model identities must be bound to new evaluation locks before benchmarks.'],'archive_sha256':sha256(here/'formal_training_evidence.tar.gz'),'manifest_sha256':sha256(e/'evidence_manifest.json'),'review_script_sha256':sha256(Path(__file__))}
write_json(here/'target_review.json',result)
print(json.dumps(result,ensure_ascii=False,indent=2))
