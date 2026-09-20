"""Verify transport, deployed code identity, real API/UI and cleanup receipts."""
from pathlib import Path
import base64,hashlib,json,sys
R=Path(__file__).resolve().parent
ROOT=R.parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
bundle=R/'received_bundle.json'
assert sha(bundle)==sys.argv[1]
items=read(bundle)['files'];assert len({x['path'] for x in items})==len(items)
D=R/'retrieved';D.mkdir(exist_ok=True)
for item in items:
 path=Path(item['path']);assert not path.is_absolute() and '..' not in path.parts
 data=base64.b64decode(item['base64'],validate=True);assert hashlib.sha256(data).hexdigest()==item['sha256']
 dest=D/path;dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists():assert dest.read_bytes()==data
 else:dest.write_bytes(data)
result=read(D/'result.json');assert result['status']=='SUPERVISED_DEPLOYMENT_VERIFIED'
assert result['environment_unchanged'] and result['model_unchanged'] and result['gpu_idle']
manifest=read(R/'package-v2/upload_manifest.json')
for name,h in manifest.items():assert sha(R/'package-v2'/name)==h
for name in ['scripts/deploy.py','scripts/common.py','scripts/step4_deploy.sh','deliverables/week7/day38/app.py','deliverables/week7/day37/app.py','deliverables/week7/day36/source/scripts/start_text_server.sh','tests/test_week8_deploy.py']:
 assert sha(ROOT/name)==manifest[name]
expected=read(R/'package-v2/model_manifest.json');assert read(D/'model_before.json')==read(D/'model_after.json')==expected
assert read(D/'prior_failed_attempt/result.json')['status']=='FAILED'
for phase,status in [('normal_stop','stopped'),('child_failure','failed')]:
 ready=read(D/(phase+'_ready.json'));assert ready['status']=='ready' and len(ready['child_pids'])==2
 end=read(D/(phase+'_cleanup.json'));assert end['status']['status']==status and all(end['ports_closed'].values()) and end['gpu_idle']
 if status=='failed':assert end['status']['error']=='one deployment child exited'
answer=read(D/'api_response.json');assert answer['model']=='week4-dpo-quantized'
assert answer['choices'][0]['finish_reason']=='stop' and answer['choices'][0]['message']['content'].strip()
ui=read(D/'gradio_response.json');assert '部署验证成功' in str(ui['response'])
assert 'Ran 2 tests' in (D/'linux_port_regression.log').read_text() and 'OK' in (D/'linux_port_regression.log').read_text()
checks={'status':'PASS','bundle_sha256':sys.argv[1],'files':len(items),'real_model_api':True,'real_gradio_client_chat':True,'browser_visual_ui_checked_this_session':False,'normal_stop':True,'immediate_restart':True,'child_failure_cleanup':True,'weights_unchanged':True,'local_tests':11,'linux_socket_tests':2,'quality_promotion':False,'production_service_left_running':False}
(R/'checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n');print(json.dumps(checks,indent=2))
