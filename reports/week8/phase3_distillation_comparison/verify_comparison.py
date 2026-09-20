"""Recount retrieved comparison evidence, with explicit remote bundle SHA supplied."""
from pathlib import Path
import base64, csv, hashlib, json, math, sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
from compare_distillation import summarize_ceval,validate_speed,validate_speed_pair
R=Path(__file__).resolve().parent
D=R/'retrieved'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(expected):
 assert len(expected)==64 and sha(R/'received_bundle.json')==expected
 files=read(R/'received_bundle.json')['files'];assert len({f['path'] for f in files})==len(files)
 for f in files:
  p=Path(f['path']);assert not p.is_absolute() and '..' not in p.parts
  assert sha(D/p)==f['sha256']==hashlib.sha256(base64.b64decode(f['base64'])).hexdigest()
 assert read(D/'status.json')['status']=='COMPARISON_COMPLETED_PENDING_LOCAL_RECOUNT'
 audit=read(D/'tokenizer_audit.json');assert audit['status']=='TOKENIZER_AND_ACTUAL_OC_PROMPTS_PASS' and audit['questions']==1346 and audit['no_truncation']
 assert sha(D/'prompt_rows.json')==audit['prompt_rows_sha256']
 assert read(D/'comparison/status.json')=={'status':'completed','same_protocol':True}
 results={};speeds={};preds={};origins={};invalid={}
 for label in ['before','after']:
  p=D/'comparison'/label
  result=summarize_ceval(p/'opencompass',ROOT);assert result==read(p/'ceval_verified.json');results[label]=result
  speed=read(p/'speed.json');checked=validate_speed(speed)
  for k,v in checked.items():assert math.isclose(speed[k],v,rel_tol=1e-12)
  speeds[label]=speed;preds[label]={};origins[label]={};invalid[label]=0
  predictions=list((p/'opencompass').glob('*/predictions/*/ceval-*.json'));assert len(predictions)==52
  for rawpath in predictions:
   raw=read(rawpath);resultpath=rawpath.parent.parent.parent/'results'/rawpath.parent.name/rawpath.name;details=read(resultpath)['details']
   assert set(raw)==set(details)-{'type'}
   for index,row in raw.items():
    detail=details[index];assert row['gold']==detail['references'] and row['prediction']==detail['origin_prediction'] and row['origin_prompt']==detail['prompt']
    parsed=next((ch for ch in row['prediction'] if ch.isupper()),'')
    assert parsed==detail['predictions'];invalid[label]+=int(parsed not in ('A','B','C','D'))
    key=(rawpath.stem,index);preds[label][key]=detail['predictions']==detail['references'];origins[label][key]=row['origin_prompt']
  assert len(preds[label])==1346 and sum(preds[label].values())==result['correct']
 validate_speed_pair(speeds['before'],speeds['after'])
 assert origins['before']==origins['after']
 from transformers import AutoTokenizer
 tokenpath=Path(sys.argv[2] if len(sys.argv)>2 else '/tmp/week8-student-tokenizer-7ae557')
 base_manifest=read(R/'package/before_manifest.json')
 for f in base_manifest['files']:
  if (tokenpath/f['path']).exists():assert sha(tokenpath/f['path'])==f['sha256']
 tok=AutoTokenizer.from_pretrained(tokenpath,local_files_only=True)
 frozen={(r['subject'],str(r['row_index'])):r for r in read(ROOT/'reports/week8/phase2_eval_hardening_v2/prompt_lengths.json') if r['dataset']=='ceval'}
 for (subject,index),prompt in origins['before'].items():
  messages=[{'role':{'HUMAN':'user','BOT':'assistant'}[r['role']],'content':r['prompt']} for r in prompt]
  assert [m['role'] for m in messages]==['user','assistant']*5+['user']
  ids=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
  exp=frozen[(subject.removeprefix('ceval-'),index)]
  assert hashlib.sha256(json.dumps(ids).encode()).hexdigest()==exp['token_ids_sha256'] and len(ids)==exp['input_tokens'] and len(ids)+32<=2048
 from mmengine.config import Config
 for label in ['before','after']:
  folder=D/'comparison'/label;cfgpath=folder/'ceval_explicit.py';binding=read(folder/'input_binding.json')
  assert sha(cfgpath)==binding['config_sha256']
  cfg=Config.fromfile(str(cfgpath)).to_dict();assert len(cfg['datasets'])==52
  m=cfg['models'][0];assert m['max_seq_len']==2048 and m['max_out_len']==32 and m['batch_size']==1 and m['tokenizer_only'] is False
  assert m['generation_kwargs']=={'do_sample':False,'num_beams':1} and m['path']==m['tokenizer_path']==read(D/(label+'_view_manifest.json'))['model_dir']
  for dataset in cfg['datasets']:assert dataset['reader_cfg']['test_split']=='val' and dataset['infer_cfg']['retriever']['fix_id_list']==[0,1,2,3,4]

 b=preds['before'];a=preds['after'];paired={'both_correct':sum(b[k] and a[k] for k in b),'before_only_correct':sum(b[k] and not a[k] for k in b),'after_only_correct':sum(not b[k] and a[k] for k in b),'both_wrong':sum(not b[k] and not a[k] for k in b)}
 assert sum(paired.values())==1346
 rows=list(csv.DictReader((D/'comparison/comparison.csv').open()));assert [r['stage'] for r in rows]==['before','after']
 for row in rows:
  label=row['stage'];assert int(row['ceval_correct'])==results[label]['correct'] and int(row['ceval_questions'])==1346
  assert math.isclose(float(row['ceval']),results[label]['accuracy']) and math.isclose(float(row['tokens_per_second']),speeds[label]['tokens_per_second'])
 for label,source in [('before',R/'package/before_manifest.json'),('after',R/'package/after_manifest.json')]:
  manifest=read(D/(label+'_view_manifest.json'));source=read(source)
  weights=lambda v:{f['path']:f['sha256'] for f in v['files'] if f['path'].endswith('.safetensors')}
  assert weights(manifest)==weights(source)
 checks={'status':'PASS','bundle_sha256':expected,'retrieved_files':len(files),'ceval':results,'paired_correctness':paired,'non_abcd_predictions':invalid,'ceval_delta_percentage_points':results['after']['accuracy']-results['before']['accuracy'],'speed':{k:validate_speed(v) for k,v in speeds.items()},'speed_ratio_after_before':speeds['after']['tokens_per_second']/speeds['before']['tokens_per_second'],'all_1346_origin_prompts_identical':True,'scope':'Local raw prediction/result/gold/receipt recount; actual weight loading and timing occurred on AutoDL320.'}
 (R/'checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n');print(json.dumps(checks,ensure_ascii=False,indent=2))
if __name__=='__main__':main(sys.argv[1])
