"""Read-only source audit; writes only this directory's inventory and receipt."""
from pathlib import Path
import json, hashlib, collections
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
B = ROOT/'outputs/01a057f0-219e-7f43-9161-67e37e5d00d4'
F, T = B/'keep_only_freeze_20260911', B/'technical_preflight_20260911'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())
def rows(p): return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def write(name,v): (OUT/name).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def item(p,role):
 p=Path(p); v=rows(p) if p.suffix=='.jsonl' else read(p)
 n=len(v) if isinstance(v,list) else len(next((v[k] for k in ('items','questions','prompts','records') if isinstance(v.get(k),list)),[]))
 return {'path':str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),'sha256':sha(p),'records':n,'role':role}
checks=[]
def check(name,ok):
 checks.append({'check':name,'passed':bool(ok)})
 if not ok: raise ValueError(name)
fm,tm=read(F/'manifest.json'),read(T/'local_preflight.json')
for folder,manifest in [(F,fm),(T,tm)]:
 for name,expected in manifest['output_sha256'].items(): check(str((folder/name).relative_to(ROOT)),sha(folder/name)==expected)
deps=[]
for path,expected in fm['input_and_dependency_sha256'].items():
 p=Path(path); deps.append({'path':path,'expected_sha256':expected,'exists':p.exists(),'matches':p.exists() and sha(p)==expected})
check('all original freeze dependencies available and unchanged',all(x['matches'] for x in deps))
selected,train,val=rows(F/'sft_keep_only.sharegpt.jsonl'),rows(T/'train.sharegpt.jsonl'),rows(T/'validation.sharegpt.jsonl')
sl,tl,vl=rows(F/'selected_lineage.jsonl'),rows(T/'train_lineage.jsonl'),rows(T/'validation_lineage.jsonl')
source={x['sample_id']:(x,r) for x,r in zip(sl,selected)}
check('counts 1663 = 1580 + 83',len(selected)==1663 and len(train)==1580 and len(val)==83 and len(sl)==1663 and len(tl)==1580 and len(vl)==83)
ids=lambda rr:{x['sample_id'] for x in rr}
check('unique IDs and disjoint complete partition',len(source)==1663 and len(ids(tl))==1580 and len(ids(vl))==83 and not ids(tl)&ids(vl) and ids(tl)|ids(vl)==ids(sl))
check('all partition content exactly equals frozen original',all(source[x['sample_id']][1]==r and source[x['sample_id']][0]['content_sha256']==x['content_sha256'] for ll,rr in [(tl,train),(vl,val)] for x,r in zip(ll,rr)))
check('all selected retain common KEEP rule',all(all(x[k]=='KEEP' for k in ['old_decision','frozen_old_decision','new_decision']) for x in sl))
excluded=rows(F/'excluded_records.jsonl')
check('all 4908 candidate rows accounted once',len(excluded)==3245 and len({x['candidate_row_number'] for x in sl+excluded})==4908)
for name,local in [('train',T/'train.sharegpt.jsonl'),('validation',T/'validation.sharegpt.jsonl')]:
 check('remote consumed '+name+' parsed content and order identity',rows(T/f'remote_evidence/derived/{name}.jsonl')==rows(local))
inventory=[]
for p,role in [
 (ROOT/'deliverables/week2/day6/source/data/formatted/week2_5k_sharegpt.jsonl','historical_raw_not_week8_input'),
 (ROOT/'deliverables/week2/day7/data/week2_clean_sharegpt.jsonl','historical_clean_not_week8_input'),
 (B/'recovery_closeout_20260908/dispositions_4908.jsonl','historical_review_dispositions'),
 (B/'revision_review_2165_20260910/03_保留修订正文_2165条.jsonl','revision_draft_not_released_not_training'),
 (F/'sft_keep_only.sharegpt.jsonl','approved_original_parent_pool'),
 (F/'selected_lineage.jsonl','parent_lineage'),(F/'excluded_records.jsonl','excluded_candidate_ledger'),
 (T/'train.sharegpt.jsonl','week8_source_before_new_split'),(T/'train_lineage.jsonl','week8_source_lineage'),
 (T/'validation.sharegpt.jsonl','protected_exposed_historical_diagnostic'),(T/'validation_lineage.jsonl','protected_lineage')]: inventory.append(item(p,role))
evals=[]
for e in tm['evaluation_sources']:
 p=Path(e['path']);check('historical eval hash '+p.name,sha(p)==e['sha256']);evals.append(item(p,'protected_exposed_diagnostic'))
for folder in ['generation_comparison_20260911','checkpoint_diagnostic_20260911','precision_path_20260911','conservative_sft_20260911']:
 manifest=read(B/folder/'input_manifest.json')
 for name in ['inference_inputs.json','confirmation_questions.json']:
  if name not in manifest: continue
  p=B/folder/name;check(folder+'/'+name,sha(p)==manifest[name]);evals.append(item(p,'protected_exposed_diagnostic'))
for name in ['evaluation_questions.json','evaluation_rubric.json']:
 evals.append(item(ROOT/'deliverables/week3/day14/source/data'/name,'custom20_fixed_'+('questions' if 'questions' in name else 'rubric')))
tok=T/'remote_evidence/smoke_adapter'
expected={e['path']:e['sha256'] for e in read(T/'remote_evidence/audit/environment.json')['base_files_rehashed']}
tokenizer=[]
for name in ['tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','added_tokens.json','special_tokens_map.json']:
 p=tok/name
 if p.exists():
  tokenizer.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'original_base_sha256':expected.get(name),'matches_original_base':sha(p)==expected[name] if name in expected else None})
# Serialized tokenizer config can differ; pin exact local template, require LF equivalence later.
template=read(tok/'tokenizer_config.json')['chat_template']
check('local vocabulary matches original base',sha(tok/'vocab.json')==expected['vocab.json'])
check('saved tokenizer matches delivered tokenizer',sha(tok/'tokenizer.json')==sha(ROOT/'outputs/Week7_正式提交_20260912/02_AWQ模型单独交付/tokenizer.json'))
write('input_inventory.json',{'schema_version':1,'source_versions':inventory,'protected_evaluation_files':evals,'tokenizer_files':tokenizer,'chat_template_sha256':hashlib.sha256(template.encode()).hexdigest(),'freeze_dependencies':deps})
write('verification.json',{'status':'PASS_INPUT_IDENTITY_ONLY','checks':checks,'counts':{'parent':len(selected),'source':len(train),'protected':len(val),'excluded':len(excluded)},'train_sources':dict(collections.Counter(x['source'] for x in tl)),'train_multiturn':sum(sum(c['from']=='gpt' for c in r['conversations'])>1 for r in train),'new_data_generated':False,'week8_leakage_check_run':False,'week8_loader_run':False,'gpu_started':False})
print(json.dumps({'checks_passed':len(checks),'inventory_records':len(inventory),'protected_files':len(evals),'tokenizer':tokenizer},ensure_ascii=False,indent=2))
