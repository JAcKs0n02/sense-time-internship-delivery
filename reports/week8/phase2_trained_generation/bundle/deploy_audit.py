"""Install only new generation artifacts and audit 320; no inference or API."""
import datetime,hashlib,json,pathlib,shutil,socket,sys,importlib.metadata
HERE=pathlib.Path(__file__).resolve().parent
ROOT=pathlib.Path('/root/autodl-tmp/week8-judge-runtime-20260916')
assert socket.gethostname()=='autodl-container-be044ebe99-be706b14'
manifest=json.loads((HERE/'bundle_manifest.json').read_text())
for name,h in manifest.items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==h,name
for name in manifest:
 src=HERE/name;dst=ROOT/name
 if dst.exists():assert src.read_bytes()==dst.read_bytes(),name
 else:dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
sys.path.insert(0,str(ROOT/'scripts'))
from week8_generation_only import validate,sha,read,save
planpath=ROOT/'reports/week8/phase2_trained_generation/plan.json';plan=read(planpath)
print('DEPLOYED_VERIFYING_WEIGHTS',flush=True)
validate(plan,ROOT,check_models=True)
receipt={'status':'TARGET_GENERATION_AUDIT_PASS','host':socket.gethostname(),'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'plan_sha256':sha(planpath),'models':list(plan['models']),'model_weights_verified':True,'judge_calls':0,'gpu_inference':False,'versions':{p:importlib.metadata.version(p) for p in ['torch','transformers','opencompass']},'disk_free_bytes':shutil.disk_usage('/root/autodl-tmp').free}
save(ROOT/'reports/week8/phase2_trained_generation/target_audit.json',receipt)
save(HERE/'target_audit.json',receipt)
print(json.dumps(receipt),flush=True)
