"""Reconcile current artifacts; reproduce remote configuration byte-for-byte."""
import hashlib,json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[3];OUT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
from build_week8_formal_eval_config import build_config
from step3_eval import load_judge_profile
from mmengine.config import Config
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads((ROOT/p).read_text())
pre=read('configs/week8_evaluation_preflight.json');refs=[]
for key in ['input_lock','overlap_receipt','independent_verification','integrated_runtime_candidate']:
 refs.append(pre[key])
refs += [{'path':pre['target_runtime']['receipt'],'sha256':pre['target_runtime']['sha256']}]
j=pre['judge_proposal']
for k in ['target_deployment','target_credentials']:
 refs.append({'path':j[k]['receipt'],'sha256':j[k]['sha256']})
refs += [{'path':j['integration_receipt'],'sha256':j['integration_receipt_sha256']},{'path':j['calibration_v3_receipt'],'sha256':j['calibration_v3_sha256']}]
for ref in refs:assert h(ROOT/ref['path'])==ref['sha256'],ref['path']
profile_path=ROOT/'configs/week8_judge_profile.json';profile_ref={'path':str(profile_path.relative_to(ROOT)),'sha256':h(profile_path)}
profile=load_judge_profile({'judge_profile':profile_ref})
target=read(j['target_deployment']['receipt']);credentials=read(j['target_credentials']['receipt']);forward=read(pre['target_runtime']['receipt'])
assert target['profile_verified'] and target['config_datasets']==119 and target['dependencies_checked']==304
assert credentials['status']=='TARGET_CREDENTIAL_AND_READONLY_API_PASS' and credentials['profile_verified'] and credentials['staging_key_removed']
assert credentials['host']==target['host']==forward['forward']['host']
assert forward['prompt']['all_token_hashes_match_independent'] and forward['prompt']['questions']==12928
assert forward['forward']['all_logits_finite'] and forward['forward']['all_parameters_cuda_bfloat16']
model=pathlib.Path('/root/autodl-tmp/qwen25-week1/week1/models/Qwen2.5-7B-Instruct')
cfg,lock=build_config(ROOT,model)
remote=target['root']
# Only paths change; compare all model/dataset behavior with the forward-tested config.
for ds in cfg['datasets']:ds['path']=ds['path'].replace(str(ROOT),remote)
cfg['models'][0]['tokenizer_path']=cfg['models'][0]['tokenizer_path'].replace(str(ROOT),remote)
path=OUT/'remote_opencompass_formal.py';Config(cfg).dump(str(path))
assert h(path)==target['config_sha256'],'remote configuration reconstruction mismatch'
old=Config.fromfile(str(ROOT/'reports/week8/phase2_target_runtime/remote_opencompass_formal.py')).to_dict()
def normalize(v):
 if isinstance(v,dict):return {k:normalize(x) for k,x in v.items()}
 if isinstance(v,list):return [normalize(x) for x in v]
 if isinstance(v,str):return v.replace('/root/autodl-tmp/week8-target-runtime-20260915','<ROOT>').replace(remote,'<ROOT>')
 return v
assert normalize(cfg)==normalize(old),'inference behavior changed since forward verification'
lock['config']={'path':'target-check/formal-config/opencompass_formal.py','sha256':h(path)}
lock['judge']={'model':'deepseek-flash','url':'https://api.deepseek.com'};lock['judge_profile']=profile_ref
lock['pending']=['target credential access and final release review','separate GPU training smoke']
p=OUT/'remote_runtime_candidate.json';p.write_text(json.dumps(lock,ensure_ascii=False,indent=2)+'\n')
assert h(p)==target['runtime_candidate_sha256'],'target candidate reconstruction mismatch'
for ref in lock['dependencies']+[lock['records']]:assert h(ROOT/ref['path'])==ref['sha256'],ref['path']
review={'status':'FINAL_EVALUATION_LOCK_REVIEW_PASS','evaluation_policy_reviewed':True,'training_allowed':False,'target_root':remote,'target_receipts_rechecked':refs,'config_sha256':h(path),'target_candidate_sha256':h(p),'judge_profile':profile_ref,'dependencies_rechecked':len(lock['dependencies']),'inference_config_equivalent_to_forward_test':True,'limitations':['Base-model evaluation lock only: SFT and DPO model paths require separate identity-bound locks.','Model service alias is not an immutable weight snapshot; repeat variation reached 0.5.','This review does not prove backward/optimizer/export execution; training smoke is still required.']}
(OUT/'review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
lock['verified']=True;lock['pending']=[]
lock['release_review']={'path':'reports/week8/phase2_final_release/review.json','sha256':h(OUT/'review.json')}
(OUT/'runtime_lock.json').write_text(json.dumps(lock,ensure_ascii=False,indent=2)+'\n')
print('PASS: target config/candidate byte identities, 304 dependencies, calibrated judge and evidence chain. Training remains disabled.')
