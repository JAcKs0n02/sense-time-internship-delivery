"""Run from the repository root with Transformers 4.50.0 and the pinned tokenizer."""
from pathlib import Path
import hashlib
import json
import sys
from importlib.metadata import version
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
from distill import teacher_messages,verify_target_review,student_training_schedule
from common import sha256,write_json
from transformers import AutoTokenizer
r=Path(__file__).resolve().parent
root=r.parents[2]
c=root/'reports/week8/phase3_distillation_curation'
read=lambda p:json.loads(Path(p).read_text())
receipt=read(r/'tokenizer_receipt.json')
assert version('transformers')=='4.50.0'
for f in receipt['files']:assert sha256(f['path'])==f['sha256']
assert sha256(root/'reports/week8/phase3_distillation_target/retrieved/student_manifest.json')==receipt['student_manifest_sha256']
tok=AutoTokenizer.from_pretrained(receipt['tokenizer_dir'],local_files_only=True)
source=read(c/'curated_targets.json');stats=read(r/'data/statistics.json');audit=read(r/'data/audit.json')
assert stats['source_sha256']==sha256(c/'curated_targets.json')
assert (stats['raw_count'],stats['clean_count'],stats['train_count'],stats['validation_count'])==(152,152,137,15)
assert stats['rejected']==stats['exact_duplicates']==stats['fuzzy_duplicates']==stats['prompt_overlap']==0
for name,digest in stats['outputs'].items():assert sha256(r/'data'/name)==digest
all_ids=set();prompt_keys={};lineage=[];lengths=[]
for split in ('train','validation'):
 rows=read(r/'data'/f'{split}_alpaca.json');prompt_keys[split]=set()
 for row in rows:
  index=int(row['sample_id'].split('-')[1]);old=source[index]
  assert index not in all_ids;all_ids.add(index)
  assert row['output']==old['output']
  messages=teacher_messages(row);prompt_keys[split].add(json.dumps(messages,ensure_ascii=False,sort_keys=True))
  length=len(tok.apply_chat_template(messages+[{'role':'assistant','content':row['output']}],tokenize=True,add_generation_prompt=False));assert length<=1024;lengths.append(length)
  changed=teacher_messages(old)!=messages
  assert changed==(index==95)
  if changed:
   assert '电影艺术理论研究的现状' in row['instruction'] and row['output']=='Art'
   assert row['instruction'].startswith('请判断下面文本的主题是什么')
  lineage.append({'prepared_sample_id':row['sample_id'],'source_index':index,'original_sample_id':old['sample_id'],'split':split,'tokens':length,'prompt_truncated':changed,'assistant_output_unchanged':True})
assert all_ids==set(range(152)) and not prompt_keys['train']&prompt_keys['validation']
changes=[row for row in audit['records'] if row.get('truncated')];assert len(changes)==1
assert changes[0]['id']=='source-000095' and changes[0]['truncated_roles']==['user']
assert changes[0]['original_length']==2048 and changes[0]['final_length']==1024
assert not any(x.get('html') or x.get('controls') or x.get('whitespace') for x in audit['records'])
schedule=student_training_schedule(137);assert schedule==read(r/'student_training_schedule.json')
assert schedule['steps_per_epoch']==18 and schedule['max_steps']==36 and schedule['sample_presentations']==274
result=verify_target_review(c/'curated_targets.json',c/'retrieved/generation/generation_receipt.json',r/'quality_review.json',read(root/'configs/distillation.json'),curation_path=c/'curation_manifest.json')
assert result['accepted']==191 and result['curated_count']==152
write_json(r/'prepared_lineage.json',lineage)
write_json(r/'checks.json',{'status':'CURATED_INPUT_AND_REAL_TOKENIZER_PREPARATION_VERIFIED','teacher_complete_count':191,'curated_count':152,'train_count':137,'validation_count':15,'prompt_overlap':0,'assistant_outputs_unchanged':152,'truncated_user_prompts':1,'maximum_chat_template_tokens':max(lengths),'optimizer_steps':36,'sample_presentations':274,'quality_review_sha256':sha256(r/'quality_review.json'),'curation_manifest_sha256':sha256(c/'curation_manifest.json'),'student_tokenizer_receipt_sha256':sha256(r/'tokenizer_receipt.json'),'versions':{k:version(k) for k in ['transformers','tokenizers','jinja2']},'gpu_started':False,'student_training_started':False})
print('PASS: 152 targets; 137 train / 15 validation; 0 overlap; 36 optimizer steps; 152 unchanged answers.')
