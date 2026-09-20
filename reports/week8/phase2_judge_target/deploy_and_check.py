import hashlib,json,os,pathlib,shutil,socket,subprocess,sys,tarfile,datetime
BASE=pathlib.Path('/root/autodl-tmp/week8-target-runtime-20260915')
ROOT=pathlib.Path('/root/autodl-tmp/week8-judge-runtime-20260916')
assert socket.gethostname()=='autodl-container-be044ebe99-be706b14'
ROOT.mkdir(exist_ok=False)
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((BASE/'transfer_manifest.json').read_text())
# Old source delta changed builder and added package marker; overlay supplies current versions.
archive=BASE/'week8-judge-deploy-20260916.tar.gz'
with tarfile.open(archive) as tar:
    members=tar.getmembers()
    assert all(m.isfile() and not pathlib.PurePosixPath(m.name).is_absolute() and '..' not in pathlib.PurePosixPath(m.name).parts for m in members)
    overlay=json.load(tar.extractfile('deployment_files.json'))
    for item in manifest:
        if item['path'] in overlay:continue
        src=BASE/item['path'];assert h(src)==item['sha256'],item['path']
        dst=ROOT/item['path'];dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    for member in members:
        dst=ROOT/member.name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(tar.extractfile(member).read())
for name,digest in overlay.items():assert h(ROOT/name)==digest,name
os.chdir(ROOT);sys.path.insert(0,str(ROOT/'scripts'))
import step3_eval as ev
profile={'path':'configs/week8_judge_profile.json','sha256':h(ROOT/'configs/week8_judge_profile.json')}
ev.load_judge_profile({'judge_profile':profile})
OUT=ROOT/'target-check';OUT.mkdir()
results={}
def run(name,args):
    with (OUT/(name+'.log')).open('w') as f:
        proc=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT,timeout=240)
    results[name]={'exit_code':proc.returncode,'log_sha256':h(OUT/(name+'.log'))}
    assert proc.returncode==0,name
run('judge_tests',[sys.executable,'-m','unittest','discover','-s','tests','-p','*judge*.py'])
run('config_build',[sys.executable,'scripts/build_week8_formal_eval_config.py','--model','/root/autodl-tmp/qwen25-week1/week1/models/Qwen2.5-7B-Instruct','--output-dir',str(OUT/'formal-config')])
f=OUT/'formal-config/runtime_lock_candidate.json';lock=json.loads(f.read_text());assert lock['verified'] is False
lock['judge']={'model':'deepseek-flash','url':'https://api.deepseek.com'};lock['judge_profile']=profile
lock['pending']=['target credential access and final release review','separate GPU training smoke']
f.write_text(json.dumps(lock,ensure_ascii=False,indent=2)+'\n')
for ref in [lock['config'],lock['records'],lock['judge_profile']]+lock['dependencies']:
    assert h(ROOT/ref['path'])==ref['sha256'],ref['path']
# Parse config only; no model loading or benchmark jobs.
from mmengine.config import Config
cfg=Config.fromfile(str(ROOT/lock['config']['path']));assert len(cfg.datasets)==119
key_present=False;key_usable=False
from setup_week8_judge_key import load_key,KEY_PATH
key_present=KEY_PATH.exists()
try:
    load_key();key_usable=True
except (OSError,ValueError):pass
receipt={'status':'TARGET_CODE_CONFIG_PASS_RELEASE_PENDING','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':socket.gethostname(),'root':str(ROOT),'python':sys.version.split()[0],'overlay_files':len(overlay),'source_files_checked':len(manifest)-sum(x['path'] in overlay for x in manifest),'profile_verified':True,'tests':results,'config_datasets':len(cfg.datasets),'dependencies_checked':len(lock['dependencies']),'runtime_candidate_sha256':h(f),'config_sha256':h(ROOT/lock['config']['path']),'key_file_present':key_present,'key_readable_private':key_usable,'api_calls':0,'model_loaded':False,'training_allowed':False,'deployment_files':overlay}
(OUT/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False,indent=2))
print('RECEIPT_SHA256',h(OUT/'receipt.json'))
