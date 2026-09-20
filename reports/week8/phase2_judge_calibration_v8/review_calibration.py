"""Offline recheck of frozen strict-tool requests and every raw result."""
import json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
from week8_strict_tool_judge import validate_response
manifest=json.loads((OUT/'manifest.json').read_text())
for name,digest in manifest.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
cases=json.loads((OUT/'cases.json').read_text());plan=json.loads((OUT/'plan.json').read_text());status=json.loads((OUT/'status.json').read_text())
usage=dict.fromkeys(['prompt_tokens','completion_tokens','total_tokens'],0);scores={};failures=[]
markers=list(OUT.glob('*-started.json'));raws=list(OUT.glob('*-response.raw'))
assert len(markers)==status['attempts']<=14
for case in cases:
 name=case['id'];file=OUT/(name+'-response.raw')
 body=json.loads((OUT/(name+'-request.json')).read_text())
 assert body['model']=='deepseek-flash' and body['thinking']=={'type':'enabled'} and body['reasoning_effort']=='low' and body['max_tokens']==8192
 assert body['tool_choice']=='auto' and len(body['tools'])==1 and body['tools'][0]['function']['strict'] is True
 inp=json.loads(body['messages'][1]['content']);assert inp['question']==case['question'] and inp['candidate_answer']==case['answer']
 if not file.exists():continue
 raw=json.loads(file.read_text());u=raw['usage'];assert all(type(u[k]) is int and u[k]>=0 for k in usage);assert u['total_tokens']==u['prompt_tokens']+u['completion_tokens']
 for k in usage:usage[k]+=u[k]
 try:s=validate_response(raw)
 except ValueError as e:failures.append({'case':name,'error':str(e)});continue
 checks={}
 for key,t in case['checks'].items():
  dim,op=key.rsplit('_',1);checks[key]=s[dim]>=t if op=='min' else s[dim]<=t
 saved=json.loads((OUT/(name+'-checks.json')).read_text());assert saved=={'scores':s,'checks':checks,'passed':all(checks.values())}
 if not all(checks.values()):failures.append({'case':name,'checks':checks})
 scores[name]=s
assert usage==status['usage']
pairs=[]
for p in plan['pair_checks']:
 if p['left'] not in scores or p['right'] not in scores:continue
 delta={k:abs(scores[p['left']][k]-scores[p['right']][k]) for k in scores[p['left']]}
 ok=all(v<=(p['max_weighted_delta'] if k=='weighted_score' else p['max_dimension_delta'])+1e-9 for k,v in delta.items())
 pairs.append({'pair':p,'delta':delta,'passed':ok})
passed=len(scores)==14 and not failures and len(pairs)==2 and all(p['passed'] for p in pairs) and len(raws)==len(markers)==14
if passed:assert json.loads((OUT/'pair_checks.json').read_text())==pairs
result={'status':'SMALL_BATCH_PASS_NOT_FORMAL_RELEASE' if passed else 'CALIBRATION_FAILED_NOT_RELEASED','manifest_verified':True,'api_calls':len(markers),'retries':0,'valid_scores':len(scores),'failures':failures,'pair_checks':pairs,'usage':usage,'usage_complete':len(raws)==len(markers),'formal_release':False,'gpu_started':False,'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!='verification.json'}}
with (OUT/'verification.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps({k:v for k,v in result.items() if k!='files'},ensure_ascii=False,indent=2))
