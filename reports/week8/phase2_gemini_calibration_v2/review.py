import json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
from week8_gemini_judge import validate_response,usage,MODEL
for name,h in json.loads((OUT/'manifest.json').read_text()).items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
cases=json.loads((OUT/'cases.json').read_text());plan=json.loads((OUT/'plan.json').read_text());status=json.loads((OUT/'status.json').read_text());totals={};scores={};failures=[]
for c in cases:
 name=c['id'];b=json.loads((OUT/(name+'-request.json')).read_text());i=json.loads(b['contents'][0]['parts'][0]['text']);assert i['question']==c['question'] and i['candidate_answer']==c['answer']
 f=OUT/(name+'-response.raw')
 if not f.exists():continue
 r=json.loads(f.read_text());u=usage(r)
 for k,v in u.items():totals[k]=totals.get(k,0)+v
 try:s=validate_response(r)
 except ValueError as e:failures.append({'case':name,'error':str(e)});continue
 checks={}
 for k,v in c['checks'].items():
  d,op=k.rsplit('_',1);checks[k]=s[d]>=v if op=='min' else s[d]<=v
 assert json.loads((OUT/(name+'-checks.json')).read_text())=={'scores':s,'checks':checks,'passed':all(checks.values())}
 scores[name]=s
 if not all(checks.values()):failures.append({'case':name,'checks':checks})
pairs=[]
for p in plan['pair_checks']:
 if p['left'] not in scores or p['right'] not in scores:continue
 delta={k:abs(scores[p['left']][k]-scores[p['right']][k]) for k in scores[p['left']]};ok=all(v<=(p['max_weighted_delta'] if k=='weighted_score' else p['max_dimension_delta'])+1e-9 for k,v in delta.items());pairs.append({'pair':p,'delta':delta,'passed':ok})
calls=len(list(OUT.glob('*-started.json')));assert calls==status['attempts'];assert totals==status['usage']
passed=len(scores)==17 and not failures and len(pairs)==2 and all(p['passed'] for p in pairs)
if passed:assert json.loads((OUT/'pair_checks.json').read_text())==pairs
result={'status':'SMALL_BATCH_PASS_NOT_FORMAL_RELEASE' if passed else 'CALIBRATION_FAILED_NOT_RELEASED','api_calls':calls,'valid_responses':len(scores),'failures':failures,'pair_checks':pairs,'usage':totals,'model':MODEL,'formal_release':False,'gpu_started':False,'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!='verification.json'}}
with (OUT/'verification.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps({k:v for k,v in result.items() if k!='files'},ensure_ascii=False,indent=2))
