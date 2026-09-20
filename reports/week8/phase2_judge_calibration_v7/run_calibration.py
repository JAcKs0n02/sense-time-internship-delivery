"""Frozen Responses-schema calibration, one attempt per case, no replay."""
import sys,json,os,hashlib,urllib.request,urllib.error
from pathlib import Path
from decimal import Decimal
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
from setup_week8_judge_key import load_key
from step3_eval import judge_balance
from week8_responses_judge import validate_response

def save(name,value):
 with (OUT/name).open('x') as f:
  json.dump(value,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
def main():
 manifest=json.loads((OUT/'manifest.json').read_text())
 assert hashlib.sha256((OUT/'manifest.json').read_bytes()).hexdigest()==sys.argv[1]
 for name,digest in manifest.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
 plan=json.loads((OUT/'plan.json').read_text());cases=json.loads((OUT/'cases.json').read_text())
 save('started.json',{'manifest_sha256':sys.argv[1]})
 scores={};error=None;calls=0;usage={'input_tokens':0,'output_tokens':0,'total_tokens':0};usage_complete=True
 try:
  start=judge_balance();save('starting_balance.json',{'cny':str(start)})
  for case in cases:
   current=judge_balance()
   if current<Decimal('1') or start-current>=Decimal('0.50') or calls>=14:raise ValueError('budget limit')
   name=case['id'];body=json.loads((OUT/(name+'-request.json')).read_text())
   save(name+'-started.json',{'balance_cny':str(current)});calls+=1
   req=urllib.request.Request(plan['endpoint'],data=json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+load_key()})
   try:
    with urllib.request.urlopen(req,timeout=180) as r: data=r.read(2*1024*1024+1)
   except urllib.error.HTTPError as e:
    save(name+'-http-error.json',{'status':e.code});raise ValueError('HTTP failure') from None
   with (OUT/(name+'-response.raw')).open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
   raw=json.loads(data);u=raw.get('usage',{})
   if all(type(u.get(k)) is int and u[k]>=0 for k in usage) and u['total_tokens']==u['input_tokens']+u['output_tokens']:
    for k in usage:usage[k]+=u[k]
   else:usage_complete=False
   score=validate_response(raw);checks={}
   for k,v in case['checks'].items():
    d,op=k.rsplit('_',1);checks[k]=score[d]>=v if op=='min' else score[d]<=v
   save(name+'-checks.json',{'scores':score,'checks':checks,'passed':all(checks.values())})
   print(name,'PASS' if all(checks.values()) else 'FAIL',flush=True)
   if not all(checks.values()):raise ValueError('predeclared score check failed: '+name)
   scores[name]=score
  pairs=[]
  for pair in plan['pair_checks']:
   delta={k:abs(scores[pair['left']][k]-scores[pair['right']][k]) for k in scores[pair['left']]}
   passed=all(v<=(pair['max_weighted_delta'] if k=='weighted_score' else pair['max_dimension_delta'])+1e-9 for k,v in delta.items())
   pairs.append({'pair':pair,'delta':delta,'passed':passed})
  save('pair_checks.json',pairs)
  if not all(p['passed'] for p in pairs):raise ValueError('pair stability failed')
 except Exception as e:
  error=str(e) if isinstance(e,ValueError) else type(e).__name__
  if len(list(OUT.glob('*-response.raw')))!=calls:usage_complete=False
 finally:
  save('status.json',{'status':'FAILED_NOT_RELEASED' if error else 'SMALL_BATCH_PASS_PENDING_REVIEW','error':error,'accepted_cases':list(scores),'attempts':calls,'usage':usage,'usage_complete':usage_complete,'formal_release':False})
  try:save('latest_balance.json',{'cny':str(judge_balance())})
  except Exception:pass
 print('FINISHED',error or 'PASS',flush=True)
 return bool(error)
if __name__=='__main__':sys.exit(main())
