"""Bounded real deployment acceptance: existing step4 entry, API/UI, owned cleanup."""
from pathlib import Path
import base64, hashlib, json, os, platform, signal, socket, subprocess, sys, time, urllib.request, gzip
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'evidence'
MODEL=Path('/root/autodl-tmp/qwen25-week7/models/week4-dpo-awq-4bit-g128-20260911')
ENV=Path('/root/autodl-tmp/qwen25-week7/envs/serving-vllm064-20260911')
def write(name,obj): (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
 return h.hexdigest()
def gpu(): return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).strip()
def free(port):
 with socket.socket() as s:
  return s.connect_ex(('127.0.0.1',port))!=0
def request(url,payload=None):
 data=None if payload is None else json.dumps(payload).encode()
 with urllib.request.urlopen(urllib.request.Request(url,data=data,headers={'Content-Type':'application/json'}),timeout=120) as r:
  return json.load(r)
def model_check():
 expected=json.loads((ROOT/'model_manifest.json').read_text())
 actual={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in MODEL.iterdir() if p.is_file()}
 assert actual==expected, 'model manifest mismatch'
 return actual
def wait_clean():
 until=time.monotonic()+45
 while time.monotonic()<until:
  if free(8000) and free(7860) and not gpu(): return
  time.sleep(1)
 raise RuntimeError('deployment did not release ports/GPU')
def interrupted(sig,frame): raise TimeoutError('session deadline/signal '+str(sig))
def main():
 OUT.mkdir(exist_ok=False)
 result={'status':'RUNNING','started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'quality_promotion':False}
 active=None
 signal.signal(signal.SIGALRM,interrupted); signal.signal(signal.SIGTERM,interrupted); signal.alarm(1500)
 try:
  assert platform.node()=='autodl-container-be044ebe99-be706b14'
  assert Path(sys.prefix)==ENV
  assert not gpu() and free(8000) and free(7860)
  manifest=json.loads((ROOT/'upload_manifest.json').read_text())
  for name,digest in manifest.items(): assert sha(ROOT/name)==digest,name
  write('model_before.json',model_check())
  import importlib.metadata as m
  write('versions.json',{x:m.version(x) for x in ['vllm','torch','transformers','gradio','gradio_client']})
  (OUT/'pip_freeze_before.txt').write_bytes(subprocess.check_output([sys.executable,'-m','pip','freeze','--all']))
  env=dict(os.environ,PIPELINE_PYTHON=sys.executable,WEEK7_SERVING_PYTHON=sys.executable,WEEK7_UI_PYTHON=sys.executable,WEEK7_TEXT_MODEL_PATH=str(MODEL),PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',GRADIO_ANALYTICS_ENABLED='False',WEEK7_UI_AUDIT_DIR=str(OUT/'ui_events'))
  env.pop('PYTHONPATH',None)
  for phase in ['normal_stop','child_failure']:
   run=OUT/phase
   command=['bash',str(ROOT/'scripts/step4_deploy.sh'),'--run-dir',str(run)]
   started=time.monotonic()
   p=subprocess.run(command,env=env,capture_output=True,text=True,timeout=480)
   (OUT/(phase+'_launcher.log')).write_text(p.stdout+'\n'+p.stderr)
   assert p.returncode==0,p.stderr
   ready=json.loads(p.stdout); active=ready['supervisor_pid']
   assert ready['status']=='ready' and len(ready['child_pids'])==2
   write(phase+'_ready.json',dict(ready,startup_seconds=time.monotonic()-started))
   if phase=='normal_stop':
    models=request('http://127.0.0.1:8000/v1/models'); write('models.json',models)
    assert [x['id'] for x in models['data']]==['week4-dpo-quantized']
    payload={'model':'week4-dpo-quantized','messages':[{'role':'user','content':'请用一句话说明什么是知识蒸馏。'}],'temperature':0,'max_tokens':128}
    write('api_request.json',payload)
    answer=request('http://127.0.0.1:8000/v1/chat/completions',payload); write('api_response.json',answer)
    assert answer['model']==payload['model'] and answer['choices'][0]['message']['content'].strip() and answer['choices'][0]['finish_reason']=='stop'
    config=request('http://127.0.0.1:7860/config'); write('gradio_config.json',config)
    assert any(x.get('api_name')=='chat' for x in config['dependencies'])
    from gradio_client import Client
    client=Client('http://127.0.0.1:7860/',verbose=False)
    api=client.view_api(return_format='dict'); write('gradio_api_schema.json',api)
    params=api['named_endpoints']['/chat']['parameters']
    assert len(params)==4,params
    result_ui=client.predict('请只回答：部署验证成功。',0,1,64,api_name='/chat')
    write('gradio_response.json',{'request':'请只回答：部署验证成功。','response':result_ui})
    assert '部署验证成功' in str(result_ui),repr(result_ui)
    os.kill(active,signal.SIGTERM)
   else:
    os.kill(ready['child_pids'][1],signal.SIGTERM)
   deadline=time.monotonic()+45
   while time.monotonic()<deadline:
    state=json.loads((run/'status.json').read_text())
    if state['status']!='ready': break
    time.sleep(.5)
   assert state['status']==('stopped' if phase=='normal_stop' else 'failed'),state
   if phase=='child_failure': assert state['error']=='one deployment child exited'
   wait_clean(); active=None
   write(phase+'_cleanup.json',{'status':state,'ports_closed':{str(x):free(x) for x in [8000,7860]},'gpu_idle':not gpu()})
   print(phase.upper()+'_PASS',flush=True)
  write('model_after.json',model_check())
  after=subprocess.check_output([sys.executable,'-m','pip','freeze','--all'])
  assert after==(OUT/'pip_freeze_before.txt').read_bytes()
  result.update(status='SUPERVISED_DEPLOYMENT_VERIFIED',environment_unchanged=True,model_unchanged=True,gpu_idle=not gpu())
 except BaseException as exc:
  result.update(status='FAILED',error=repr(exc))
  raise
 finally:
  if active:
   try: os.kill(active,signal.SIGTERM)
   except ProcessLookupError: pass
   try: wait_clean()
   except Exception as exc: result['cleanup_error']=repr(exc)
  result['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());write('result.json',result)
  files=[{'path':str(p.relative_to(OUT)),'sha256':sha(p),'base64':base64.b64encode(p.read_bytes()).decode()} for p in sorted(OUT.rglob('*')) if p.is_file()]
  blob=json.dumps({'files':files},ensure_ascii=False).encode()
  (ROOT/'deployment_evidence.txt').write_text(base64.b64encode(gzip.compress(blob)).decode())
  print(json.dumps({'result':result,'bundle_sha256':hashlib.sha256(blob).hexdigest(),'file_count':len(files)}),flush=True)
if __name__=='__main__':main()
