import os
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
from pathlib import Path
import json,time,subprocess,sys,datetime
ROOT=Path('/Users/yifanren/Documents/商汤科技实习');OUT=ROOT/'reports/week8/model_backup_20260919';DEST=ROOT/'models/backups/week8-20260919'
if len(sys.argv)>1:
 import torch
 from transformers import AutoTokenizer,AutoModelForCausalLM
 role=sys.argv[1];p=DEST/role;start=time.time();torch.set_num_threads(4)
 tokenizer=AutoTokenizer.from_pretrained(p,local_files_only=True,trust_remote_code=False)
 model=AutoModelForCausalLM.from_pretrained(p,local_files_only=True,trust_remote_code=False,torch_dtype=torch.bfloat16,device_map='cpu',attn_implementation='eager');model.eval()
 prompt=tokenizer.apply_chat_template([{'role':'user','content':'请回答：1+1等于几？'}],tokenize=False,add_generation_prompt=True)
 inputs=tokenizer(prompt,return_tensors='pt')
 with torch.inference_mode():
  logits=model(**inputs).logits;assert torch.isfinite(logits).all()
  generated=model.generate(**inputs,max_new_tokens=8,do_sample=False,pad_token_id=tokenizer.eos_token_id)
 ids=generated[0,inputs['input_ids'].shape[1]:].tolist();assert ids and all(0<=i<model.config.vocab_size for i in ids)
 answer=tokenizer.decode(ids,skip_special_tokens=True);assert answer.strip()
 r={'status':'PASS_CPU_COLD_LOAD_AND_GENERATION','role':role,'device':'cpu','dtype':'bfloat16','torch':torch.__version__,'parameter_count':sum(p.numel() for p in model.parameters()),'all_logits_finite':True,'new_token_ids':ids,'generated_text':answer,'elapsed_seconds':time.time()-start,'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Local backup fresh-process loading and 8-token smoke only; not benchmark or quality promotion'}
 (OUT/(role+'_cpu_load.json')).write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print(role+' CPU_LOAD_PASS',flush=True)
else:
 plan=json.loads((ROOT/'reports/week8/release_audit_20260919/expected_model_backups.json').read_text())
 deadline=time.time()+7000
 for m in sorted(plan['models'],key=lambda model:model['expected_bytes']):
  while not all((DEST/m['role']/f['path']).is_file() for f in m['expected_files']):
   if time.time()>deadline:raise TimeoutError('Backup wait deadline')
   time.sleep(10)
  log=OUT/(m['role']+'_cpu_load.log')
  with log.open('w') as stream:
   subprocess.run([sys.executable,__file__,m['role']],stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=900)
  print(m['role']+' CPU_LOAD_PASS',flush=True)
 print('ALL_CPU_BACKUP_LOADS_PASS',flush=True)
