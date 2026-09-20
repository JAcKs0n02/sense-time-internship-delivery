"""Offline verification of the retrieved, SHA-pinned student training receipts."""
from pathlib import Path
import base64, hashlib, json, math, sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
from distill import validate_student_state
R=Path(__file__).resolve().parent
D=R/'retrieved'
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
expected='52d2561bda709c3779dcd1a2d8a2dc6ea2d52c14fac1919d4f5286353affc8db'
assert sha(R/'received_bundle.json')==expected
files=read(R/'received_bundle.json')['files']
assert len(files)==52 and len({f['path'] for f in files})==52
for f in files:
 p=Path(f['path']); assert not p.is_absolute() and '..' not in p.parts
 assert sha(D/p)==f['sha256']==hashlib.sha256(base64.b64decode(f['base64'])).hexdigest()
prep=ROOT/'reports/week8/phase3_distillation_student_prep'
schedule=read(D/'student/student_training_schedule.json')
assert schedule==read(prep/'student_training_schedule.json')
statepath=D/'student/training/attempt-01/adapter/trainer_state.json'
result=validate_student_state(read(statepath),schedule)
assert result|{'trainer_state_sha256':sha(statepath)}==read(D/'student/student_state_verification.json')
for directory in ['preflight_data','student/data']:
 stats=read(D/directory/'statistics.json')
 assert (stats['train_count'],stats['validation_count'],stats['prompt_overlap'])==(137,15,0)
 for name,h in read(prep/'data/statistics.json')['outputs'].items():
  assert sha(D/directory/name)==sha(prep/'data'/name)==h
binding=read(D/'student/input_binding.json')
assert binding['quality_review_sha256']==sha(prep/'quality_review.json')
assert binding['teacher_targets_sha256']==sha(ROOT/'reports/week8/phase3_distillation_curation/curated_targets.json')
assert binding['curation_manifest_sha256']==sha(ROOT/'reports/week8/phase3_distillation_curation/curation_manifest.json')
assert (binding['accepted_teacher_answers'],binding['curated_teacher_answers'])==(191,152)
art=read(D/'student/student_artifact_verification.json')
assert art['tensor_receipt_sha256']==sha(D/'student/student_tensor_verification.json')
assert art['cold_load_receipt_sha256']==sha(D/'student/student_cold_load.json')
tensor=read(D/'student/student_tensor_verification.json')
assert tensor['status']=='TENSOR_MERGE_VERIFIED'
assert (tensor['updated_modules'],tensor['merged_tensors'],tensor['adapter_tensor_count'])==(168,290,336)
errors=tensor['merge_max_abs_error_by_module']; assert len(errors)==168 and all(math.isfinite(v) and v>=0 for v in errors.values())
for f in tensor['files']:
 p=D/'student/final_distilled'/f['path']
 if p.exists(): assert sha(p)==f['sha256'] and p.stat().st_size==f['bytes']
cold=read(D/'student/student_cold_load.json')
assert cold['status']=='COLD_LOAD_PASS' and cold['finite_logits'] is True
assert cold['dtype']=='torch.bfloat16' and cold['device']=='cuda:0' and cold['generated_tokens']==2
assert cold['benchmark_scored'] is False
assert read(D/'status.json')['status']=='STUDENT_TRAINING_VERIFIED_COMPARISON_PENDING'
assert read(D/'student/status.json')['training_verified'] is True
assert len(read(D/'session_exit.json')['completed'])==2
checks={'status':'PASS','bundle_sha256':expected,'retrieved_files':52,'training_state':result,'train_count':137,'validation_count':15,'merged_model_sha256':next(f['sha256'] for f in tensor['files'] if f['path']=='model.safetensors'),'merged_tensor_count':290,'updated_modules':168,'cold_load':'PASS','comparison':'PENDING','scope':'Local receipt integrity and training/data invariants verified; tensor computation and CUDA cold load performed on AutoDL320. Large weights remain remote.'}
(R/'checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(checks,ensure_ascii=False,indent=2))
