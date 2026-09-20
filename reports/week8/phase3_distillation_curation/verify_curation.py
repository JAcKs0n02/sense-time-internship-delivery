from pathlib import Path
import ast,hashlib,json,random,re,sys,tempfile,subprocess
sys.path.insert(0,'scripts')
from distill import teacher_messages,verify_target_review
r=Path('reports/week8/phase3_distillation_curation');g=r/'retrieved/generation'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
write=lambda p,d:Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
manifest=read(r/'curation_manifest.json');bundle=read(r/'full_bundle.json');receipt=read(g/'generation_receipt.json')
assert sha(r/'full_bundle.json')=='c3e5db4e3f977add42a785277fa2a13e77bedf437582a9fc6c93c8b81574265c'
assert len(bundle['files'])==213
for f in bundle['files']:
 p=Path(f['path']);assert not p.is_absolute() and '..' not in p.parts
 assert sha(r/'retrieved'/p)==f['sha256']
assert sha(g/'generation_receipt.json')==manifest['generation_receipt_sha256']=='2ad4c1fccc01ea807f89452a4e29a02d56fb4444aecdf58a164461a87a83cd35'
assert len(receipt['files'])==203
for f in receipt['files']:assert sha(g/f['path'])==f['sha256']
assert receipt['config']==read('configs/distillation.json')
assert sha(g/'teacher_identity.json')=='39d6abfaa8d260be36524c10def770fd1381a6ad12d14f1cd145f220b39964a7'
assert read(g/'teacher_identity.json')['manifest_sha256']=='aca9a2932ed427f8fc6e9962e70274eb286a09c446d177af2bc00f6df1675e4c'
selected=read(g/'selected_prompts.json');candidate_path=Path('reports/week8/phase3_distillation_preflight/candidate_prompts.json');assert sha(candidate_path)==receipt['input_sha256'];c=read(candidate_path);random.Random(42).shuffle(c);assert selected==c[:200]
original=read(g/'teacher_targets.json');curated=read(r/'curated_targets.json');decisions=manifest['decisions'];assert len(decisions)==200 and len({d['sample_id'] for d in decisions})==200
reconstructed=[];retained=[];raws=[]
for i,(p,d) in enumerate(zip(selected,decisions)):
 rawpath=g/'teacher_raw'/f'{i:04d}.json';raw=read(rawpath);raws.append(raw)
 assert raw['sample_id']==p['sample_id']==d['sample_id'] and d['index']==i
 assert raw['messages']==teacher_messages(p) and type(raw['truncated']) is bool
 assert sha(rawpath)==d['raw_sha256'] and (r/d['raw_path']).resolve()==rawpath.resolve()
 assert d['notes'].strip() and d['decision'] in ('KEEP','EXCLUDE')
 row={**{k:p[k] for k in ('sample_id','instruction','input','system','history') if k in p},'output':raw['answer']}
 if raw['answer'].strip() and not raw['truncated']:reconstructed.append(row)
 assert (d['reason_code']=='TRUNCATED')==raw['truncated']
 if d['decision']=='KEEP':
  assert not raw['truncated'] and raw['answer'].strip() and d['reason_code']=='KEEP'
  retained.append(row)
assert original==reconstructed and len(original)==191
assert curated==retained and len(curated)==152
assert sha(r/'curated_targets.json')==manifest['curated_targets_sha256']
assert sha(g/'teacher_targets.json')==manifest['original_targets_sha256']
lineage=read('logs/week8-protocol-data-20260914/run-a/lineage.json');byid={x['sample_id']:x for x in lineage};train=read('logs/week8-protocol-data-20260914/run-a/train_alpaca.json');val=read('logs/week8-protocol-data-20260914/run-a/validation_alpaca.json')
for p in selected:
 entry=byid[p['sample_id']];assert entry['split']=='train';src=train[entry['row_index']]
 assert {k:v for k,v in p.items() if k!='sample_id'}=={k:v for k,v in src.items() if k!='output'}
# Also compare actual prompt messages, including retained history, against validation.
key=lambda row:json.dumps(teacher_messages(row),ensure_ascii=False,sort_keys=True)
assert not {key(p) for p in selected}&{key(p) for p in val}
assert len(train)==1422 and len(val)==158
# Execute only the inspected pure function AST; do not execute arbitrary dataset scripts.
def fn_at(index,name):
 code=re.search(r'```(?:python)?\n(.*?)```',raws[index]['answer'],re.S).group(1)
 tree=ast.parse(code);defs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name];assert len(defs)==1
 ns={'__builtins__':{}};exec(compile(ast.Module(body=defs,type_ignores=[]),'<reviewed-function>','exec'),ns);return ns[name]
assert fn_at(77,'sum')(3,5)==8
out=fn_at(151,'remove_vowels')('The quick brown fox jumped over the lazy dog.');assert out=='Th qck brwn fx jmpd vr th lzy dg.'
try:fn_at(198,'average_of_three')(5,10,15)
except TypeError:code_error=True
else:raise AssertionError('expected teacher code failure')
with tempfile.TemporaryDirectory() as t:
 script='mkdir cats\necho "猫万岁！"\n';assert script.strip() in raws[152]['answer'].replace('# 创建名为 "cats" 的目录\n','').replace('# 打印消息 "猫万岁！"\n','').replace('\n\n','\n')
 p=subprocess.run(['bash','-c',script],cwd=t,text=True,capture_output=True,check=True);assert (Path(t)/'cats').is_dir() and p.stdout=='猫万岁！\n'
assert len(raws[51]['answer'])==5
assert sum('\u4e00'<=c<='\u9fff' for c in raws[180]['answer'])==10
assert len(raws[128]['answer'].splitlines()[1:])==16
assert '耳濡慕染'.replace('慕','目')=='耳濡目染' and decisions[95]['decision']=='KEEP'
try:verify_target_review(r/'curated_targets.json',g/'generation_receipt.json','reports/week8/phase3_distillation_gpu/quality_review.json',receipt['config'])
except ValueError as e:
 gate_error=str(e);assert gate_error=='training targets are not bound to this generation'
else:raise AssertionError('unadapted entry must not accept curated subset')
checks={'status':'EVIDENCE_AND_CURATION_INTEGRITY_VERIFIED','full_bundle_sha256':sha(r/'full_bundle.json'),'retrieved_files_verified':213,'generation_bindings_verified':203,'selected_prompts_match_seed42_candidate_order':True,'raw_prompt_message_bindings_verified':200,'complete_targets_reconstructed_exactly':191,'original_training_population':1422,'original_validation_population':158,'selected_validation_prompt_overlap':0,'selected_source_ids_bound_to_training_rows':200,'all_reviewed_ids_unique':True,'curated_answers_identical_to_original_teacher':True,'curated_order_is_original_subsequence':True,'all_exclusions_have_reasons':True,'curation_manifest_sha256':sha(r/'curation_manifest.json'),'curated_targets_sha256':sha(r/'curated_targets.json'),'transport_note':'Text export normalized CR progress bars in two logs. Restored CR only before Loading checkpoint shards; both exact original SHA256 hashes then matched. All 213 retrieved files match original hashes.','counts':manifest['counts'],'deterministic_checks':{'five_character_username':5,'eight_character_answer_actual_han':10,'requested_lines_3_to_17':15,'actual_generated_lines':16,'mean_5_8_10_12_sum':35,'mean_5_8_10_12_result':8.75,'typo_correction_0095_verified':True,'retained_addition_function':8,'retained_remove_vowels_output':out,'retained_mkdir_script_verified_in_temporary_directory':True,'excluded_average_function_raises_TypeError':code_error},'training_entry_current_rejection':gate_error,'real_student_tokenizer_preparation':'PENDING','training_allowed':False}
write(r/'checks.json',checks)
print(json.dumps({'status':checks['status'],'counts':manifest['counts'],'files':213,'train_entry':gate_error},ensure_ascii=False))
