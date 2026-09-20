"""Normalize pinned official benchmark inputs and bind explicit config sources."""
from pathlib import Path
import ast,csv,hashlib,io,json,zipfile
import pyarrow.parquet as pq
O=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def subject_map(p,name):
 for n in ast.parse(p.read_text()).body:
  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets):return ast.literal_eval(n.value)
 raise ValueError('no subject map')
def main():
 manifest=json.loads((O/'download_manifest.json').read_text())
 for e in manifest['files']:assert sha(O/e['path'])==e['sha256']
 source=O/'package_source/opencompass/configs/datasets'
 configs={'ceval':source/'ceval/ceval_gen_5f30c7.py','cmmlu':source/'cmmlu/cmmlu_gen_c13365.py'}
 subjects={b:subject_map(p,b+'_subject_mapping') for b,p in configs.items()}
 assert {b:len(s) for b,s in subjects.items()}=={'ceval':52,'cmmlu':67}
 rows=[];converted=[]
 for p in sorted((O/'snapshots/ceval/ceval-exam').glob('*/*.parquet')):
  subject=p.parent.name;split=p.name.split('-')[0]; assert subject in subjects['ceval']
  data=pq.read_table(p).to_pylist()
  target=O/'local_data/ceval'/split/(subject+'_'+split+'.csv');target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=['id','question','A','B','C','D','answer','explanation']);w.writeheader();w.writerows(data)
  converted.append({'path':str(target.relative_to(O)),'sha256':sha(target),'source':str(p.relative_to(O)),'records':len(data)})
  for i,x in enumerate(data):
   assert x['question'] and all(x[k] for k in 'ABCD')
   if split!='test':assert x['answer'] in 'ABCD' and len(x['answer'])==1
   rows.append({'benchmark':'ceval','subject':subject,'split':split,'row_index':i,'id':f'ceval:{subject}:{split}:{i}','question':x['question'],'options':{k:x[k] for k in 'ABCD'},'answer':x['answer'],'role':{'val':'scored','dev':'fewshot','test':'unscored_protected'}[split]})
 with zipfile.ZipFile(O/'snapshots/haonan-li/cmmlu/cmmlu_v1_0_1.zip') as z:
  for n in sorted(z.namelist()):
   if not n.endswith('.csv'):continue
   p=Path(n);assert len(p.parts)==2 and p.parts[0] in ['dev','test'] and p.stem in subjects['cmmlu']
   payload=z.read(n);target=O/'local_data/cmmlu'/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload)
   data=list(csv.DictReader(io.StringIO(payload.decode('utf-8-sig'))))
   converted.append({'path':str(target.relative_to(O)),'sha256':sha(target),'source':'snapshots/haonan-li/cmmlu/cmmlu_v1_0_1.zip:'+n,'records':len(data)})
   for i,x in enumerate(data):
    assert x['Question'] and len(x['Answer'])==1 and x['Answer'] in 'ABCD' and all(x[k] for k in 'ABCD')
    rows.append({'benchmark':'cmmlu','subject':p.stem,'split':p.parts[0],'row_index':i,'id':f'cmmlu:{p.stem}:{p.parts[0]}:{i}','question':x['Question'],'options':{k:x[k] for k in 'ABCD'},'answer':x['Answer'],'role':'scored' if p.parts[0]=='test' else 'fewshot'})
 counts={}
 for b,ss in subjects.items():
  counts[b]={}
  for split in (['dev','val','test'] if b=='ceval' else ['dev','test']):
   subset=[x for x in rows if x['benchmark']==b and x['split']==split]
   assert {x['subject'] for x in subset}==set(ss)
   if split=='dev':assert all(sum(x['subject']==s for x in subset)==5 for s in ss)
   counts[b][split]={'subjects':len(ss),'records':len(subset)}
 assert counts['ceval']['val']['records']==1346 or counts['ceval']['val']['records']==1398,counts
 assert counts['cmmlu']['test']['records']==11582 or counts['cmmlu']['test']['records']==11649,counts
 # Exact official counts are recorded; historical remote counts are a comparison, not the source of truth.
 assert len({x['id'] for x in rows})==len(rows)
 write(O/'benchmark_records.json',rows)
 write(O/'local_data_manifest.json',converted)
 lock={'status':'BENCHMARK_INPUTS_PINNED_RUNTIME_NOT_READY','counts':counts,'normalized_records':len(rows),'benchmark_records_sha256':sha(O/'benchmark_records.json'),'official_download_manifest_sha256':sha(O/'download_manifest.json'),'local_data_manifest_sha256':sha(O/'local_data_manifest.json'),'opencompass_version':'0.5.3','opencompass_git_tag_commit':json.loads((O/'opencompass_tag.json').read_text())['object']['sha'],'wheel_sha256':sha(next((O/'packages').glob('*.whl'))),'explicit_configs':{b:{'path':str(p.relative_to(O)),'sha256':sha(p),'fewshot_indices':[0,1,2,3,4],'test_split':'val' if b=='ceval' else 'test','answer_postprocess':'first_capital_postprocess'} for b,p in configs.items()},'generation_candidate':{'do_sample':False,'max_out_len':32,'seed':42,'max_seq_len':2048,'actual_prompt_length_and_truncation_audit_pending':True},'runtime_checks_pending':['bind explicit configs and local datasets in fresh evaluation entrypoint','validate expanded OpenCompass runtime and full 119-subject prompts, lengths and decoding','pin judge deployment/version/endpoint','fresh target GPU and model weight identity checks'],'judge':None,'gpu_target':None,'training_allowed':False,'historical_remote_counts':{'ceval':1398,'cmmlu':11649},'historical_counts_not_used_as_current_dataset_counts':True}
 write(O/'benchmark_input_lock.json',lock);print(json.dumps(counts,indent=2));print('normalized',len(rows))
if __name__=='__main__':main()
