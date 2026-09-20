"""SFT/DPO inference only. No judge imports/calls; immutable source locks retained."""
import argparse,hashlib,json,os,socket,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def check(ok,msg):
 if not ok:raise ValueError(msg)
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 with Path(p).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
def validate(plan,root=ROOT,check_models=True):
 check(plan['mode']=='generation_only' and plan['judge_calls']==0,'unexpected mode')
 check(set(plan['models'])=={'final_sft','final_dpo'},'only trained models allowed')
 for name,h in plan['dependencies'].items():
  check(not Path(name).is_absolute() and '..' not in Path(name).parts,'unsafe path');check(sha(root/name)==h,'changed dependency: '+name)
 for name,row in plan['models'].items():
  lock=read(root/row['lock_path']);check(sha(root/row['lock_path'])==row['lock_sha256'],'lock changed')
  check(lock['verified'] is True and Path(lock['model_path']).name==name,'unreleased or mismatched model')
  for item in lock['dependencies']+[lock[k] for k in ['config','records','model_identity','training_review','release_review','target_audit']]:
   path=Path(item['path']);check('..' not in path.parts,'unsafe lock dependency')
   if path.is_absolute():
    check(path.is_relative_to(Path(lock['model_path'])),'unexpected absolute dependency')
    if not check_models:continue
   check(sha(root/path)==item['sha256'],'source artifact changed: '+str(path))
  identity=read(root/lock['model_identity']['path']);check(identity['model_path']==lock['model_path'],'identity path mismatch')
  if check_models:
   model=Path(lock['model_path']);files=list(model.rglob('*'));check(not any(f.is_symlink() for f in files),'model symlink')
   check({str(f.relative_to(model)) for f in files if f.is_file()}=={r['path'] for r in identity['files']},'model file set changed')
   for f in identity['files']:
    path=model/f['path'];check(path.stat().st_size==f['bytes'] and sha(path)==f['sha256'],'model bytes changed')
 return True

def execute(args):
 check(socket.gethostname()=='autodl-container-be044ebe99-be706b14','wrong instance')
 check(sha(args.plan)==args.plan_sha256,'plan hash mismatch');plan=read(args.plan);validate(plan)
 if args.audit_only:
  print('TARGET_GENERATION_AUDIT_PASS',flush=True);return
 check(plan['release_state']=='TARGET_AUDITED_RELEASED','candidate cannot launch GPU work');check(args.model_name in plan['models'],'wrong model');out=args.output.resolve();check(not out.exists(),'output exists; no automatic replay')
 # One permanent claim per model+protocol, independent of output location.
 identity=hashlib.sha256((args.plan_sha256+args.model_name).encode()).hexdigest();claims=ROOT/'logs/generation-only-claims';claims.mkdir(parents=True,exist_ok=True)
 save(claims/(identity+'.json'),{'output':str(out),'plan_sha256':args.plan_sha256})
 out.mkdir(parents=True);save(out/'launch.json',{'model_name':args.model_name,'plan_sha256':args.plan_sha256,'gpu_host':socket.gethostname(),'judge_calls':0})
 lock=read(ROOT/plan['models'][args.model_name]['lock_path']);save(out/'runtime_lock.json',lock)
 try:
  import torch
  import step3_eval as ev
  check(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),'CUDA BF16 unavailable')
  cmd=ev.opencompass_command(ROOT/lock['config']['path'],out/'opencompass');save(out/'opencompass_command.json',cmd)
  with (out/'opencompass.log').open('x') as f:subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=plan['benchmark_timeout_seconds'])
  rows=ev.validate_benchmarks(out/'opencompass',lock,lock['model_path']);save(out/'benchmarks.json',rows)
  from transformers import AutoTokenizer,AutoModelForCausalLM
  tokenizer=AutoTokenizer.from_pretrained(lock['model_path'],local_files_only=True)
  model=AutoModelForCausalLM.from_pretrained(lock['model_path'],local_files_only=True,torch_dtype=torch.bfloat16,device_map='auto').eval()
  spec=read(ROOT/'deliverables/week3/day14/source/data/evaluation_questions.json');check(len(spec['questions'])==len({q['id'] for q in spec['questions']})==20,'invalid questions');torch.manual_seed(42)
  with (out/'answers.jsonl').open('x') as f:
   for q in spec['questions']:
    ids=tokenizer.apply_chat_template([{'role':'system','content':spec['system_message']}]+q['messages'],tokenize=True,add_generation_prompt=True,return_tensors='pt').to(model.device)
    with torch.inference_mode():generated=model.generate(ids,max_new_tokens=512,do_sample=False)
    answer=tokenizer.decode(generated[0,ids.shape[1]:],skip_special_tokens=True);check(bool(answer.strip()),'empty answer')
    f.write(json.dumps({'id':q['id'],'answer':answer},ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
  save(out/'status.json',{'status':'GENERATED_PENDING_REVIEW','model_name':args.model_name,'questions':20,'answers_sha256':sha(out/'answers.jsonl'),'judge_calls':0})
 except BaseException as e:
  save(out/'status.json',{'status':'FAILED_NO_RETRY','error_type':type(e).__name__,'judge_calls':0});raise

def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--audit-only',action='store_true');p.add_argument('--model-name',choices=['final_sft','final_dpo']);p.add_argument('--output',type=Path)
 args=p.parse_args();check(args.audit_only or args.model_name and args.output,'missing run arguments');execute(args)
if __name__=='__main__':main()
