"""Install the separately transferred key and verify read-only provider access."""
import datetime,json,os,pathlib,socket,sys,urllib.request,hashlib,shutil
assert socket.gethostname()=='autodl-container-be044ebe99-be706b14'
ROOT=pathlib.Path('/root/autodl-tmp/week8-judge-runtime-20260916')
SRC=pathlib.Path('/root/autodl-tmp/week8-target-runtime-20260915/deepseek.key')
DST=pathlib.Path('/root/.config/internship-week8/deepseek.key')
assert SRC.is_file() and not SRC.is_symlink()
assert not DST.exists() and not DST.is_symlink(),'existing credential must not be overwritten'
os.chmod(SRC,0o600)
DST.parent.mkdir(parents=True,exist_ok=True);os.chmod(DST.parent,0o700)
# Move, not copy: no staging credential remains in the workspace.
shutil.move(str(SRC),str(DST))
os.chmod(DST,0o600)
sys.path.insert(0,str(ROOT/'scripts'))
from setup_week8_judge_key import load_key
from step3_eval import judge_balance,load_judge_profile
key=load_key();assert key
profile=ROOT/'configs/week8_judge_profile.json'
load_judge_profile({'judge_profile':{'path':str(profile),'sha256':hashlib.sha256(profile.read_bytes()).hexdigest()}})
request=urllib.request.Request('https://api.deepseek.com/models',headers={'Authorization':'Bearer '+key})
try:
    with urllib.request.urlopen(request,timeout=30) as response:models=json.load(response)
    ids=[x['id'] for x in models['data']];assert 'deepseek-flash' in ids
    balance=judge_balance();assert balance>=1
except Exception as exc:
    print('CREDENTIAL_API_CHECK_FAILED',type(exc).__name__)
    sys.exit(1)
receipt={'status':'TARGET_CREDENTIAL_AND_READONLY_API_PASS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':socket.gethostname(),'root':str(ROOT),'key_file_mode':oct(DST.stat().st_mode&0o777),'parent_mode':oct(DST.parent.stat().st_mode&0o777),'staging_key_removed':not SRC.exists(),'profile_verified':True,'model_ids':ids,'balance_cny':str(balance),'read_only_api_calls':2,'paid_scoring_calls':0,'training_started':False,'runtime_released':False}
p=ROOT/'target-check/credential_receipt.json'
with p.open('x') as f:f.write(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
# Public receipt for UI recovery, never credential contents or hashes.
(ROOT.parent/'week8-target-runtime-20260915/credential_receipt.txt').write_bytes(p.read_bytes())
print(json.dumps(receipt,ensure_ascii=False,indent=2));print('RECEIPT_SHA256',hashlib.sha256(p.read_bytes()).hexdigest())
