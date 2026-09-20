"""Deploy reviewed resume artifacts and audit target dependencies without GPU/API work."""
import datetime, hashlib, json, pathlib, shutil, socket, sys, tarfile
from types import SimpleNamespace
ROOT=pathlib.Path('/root/autodl-tmp/week8-judge-runtime-20260916')
HERE=pathlib.Path(__file__).resolve().parents[3]
DEST=ROOT/'reports/week8/phase2_eval_resume'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert socket.gethostname()=='autodl-container-be044ebe99-be706b14'
manifest=json.loads((HERE/'bundle_manifest.json').read_text())
for item in manifest:assert sha(HERE/item['path'])==item['sha256'],item['path']
assert not DEST.exists()
assert sha(ROOT/'scripts/step3_eval.py')=='6cfb6f4569aac8b46476f2f9c2f4f928319d4e77150235c8c22781232b88eec3'
shutil.copytree(HERE/'reports/week8/phase2_eval_resume',DEST)
shutil.copy2(ROOT/'scripts/step3_eval.py',DEST/'step3_eval_before.py')
for name in ('step3_eval.py','week8_resume_eval.py'):shutil.copy2(HERE/'scripts'/name,ROOT/'scripts'/name)
source=ROOT/'reports/week8/phase2_formal_inference/recovered_failure'
assert not source.exists()
shutil.copytree(HERE/'reports/week8/phase2_formal_inference/recovered_failure',source)
sys.path.insert(0,str(ROOT/'scripts'))
import step3_eval as ev
import week8_resume_eval as resume
preflight=ROOT/'configs/week8_evaluation_preflight.json';original=preflight.read_bytes()
(DEST/'preflight_before_release.json').write_bytes(original)
results={};released={}
try:
 for name in ('original_base','final_sft','final_dpo'):
  lock=json.loads((DEST/(name+'_lock_candidate.json')).read_text());assert lock['verified'] is False
  lock['verified']=True;lock['pending']=[]
  local_ref={'path':'reports/week8/phase2_eval_resume/local_review.json','sha256':sha(DEST/'local_review.json')}
  lock['dependencies'].append(local_ref);lock['resume_code_review']=local_ref
  current=json.loads(original);current['runtime_lock']=lock;ev.write_json(preflight,current)
  ev.require_evaluation_lock(SimpleNamespace(model=lock['model_path'],judge_model='deepseek-flash',judge_url='https://api.deepseek.com'))
  result={'dependencies_checked':len(lock['dependencies']),'model_path':lock['model_path'],'entrypoint_gate':'PASS'}
  if name=='original_base':result['rows'],result['provenance']=resume.validate_source(source,lock)
  released[name]=lock;results[name]=result
  print('GATE_PASS',name,flush=True)
finally:preflight.write_bytes(original)
assert preflight.read_bytes()==original
for name,lock in released.items():
 p=DEST/(name+'_runtime_lock.json');ev.write_json(p,lock);results[name]['runtime_lock_sha256']=sha(p)
receipt={'status':'RESUME_LOCKS_TARGET_AUDIT_PASS','models':results,'host':socket.gethostname(),'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'preflight_restored':True,'gpu_inference':False,'judge_calls':0,'script_sha256':sha(pathlib.Path(__file__))}
ev.write_json(DEST/'target_release.json',receipt)
with tarfile.open(HERE/'resume-release-receipt.tar.gz','w:gz') as t:
 t.add(DEST/'target_release.json',arcname='target_release.json')
 for name in released:t.add(DEST/(name+'_runtime_lock.json'),arcname=name+'_runtime_lock.json')
print('RECEIPT_SHA',sha(HERE/'resume-release-receipt.tar.gz'),flush=True)
