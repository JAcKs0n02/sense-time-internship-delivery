import sys,json,hashlib,os,urllib.request,urllib.error
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
from setup_week8_gemini_key import KEY_PATH
from setup_week8_judge_key import load_key
from week8_gemini_judge import validate_response,usage,MODEL

def save(name,obj):
 with (OUT/name).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
def main():
 manifest=json.loads((OUT/'manifest.json').read_text());assert hashlib.sha256((OUT/'manifest.json').read_bytes()).hexdigest()==sys.argv[1]
 for path,h in manifest.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h
 cases=json.loads((OUT/'cases.json').read_text());plan=json.loads((OUT/'plan.json').read_text());assert len(cases)<=plan['max_calls']
 key=load_key(KEY_PATH);save('started.json',{'manifest':sys.argv[1]});scores={};calls=0;error=None;totals={};complete=True
 try:
  for c in cases:
   name=c['id'];data=(OUT/(name+'-request.json')).read_bytes();assert len(data)<65536
   save(name+'-started.json',{'attempt':1});calls+=1
   req=urllib.request.Request('https://generativelanguage.googleapis.com/v1beta/models/'+MODEL+':generateContent',data=data,headers={'Content-Type':'application/json','x-goog-api-key':key})
   try:
    with urllib.request.urlopen(req,timeout=240) as r:rawbytes=r.read(2*1024*1024+1)
   except urllib.error.HTTPError as e:
    # Redact the credential even if a provider echoes it in an error.
    body=e.read(16384).decode(errors='replace').replace(key,'[REDACTED]')
    save(name+'-http-error.json',{'http_status':e.code,'body':body});raise ValueError('HTTP '+str(e.code)) from None
   with (OUT/(name+'-response.raw')).open('xb') as f:f.write(rawbytes);f.flush();os.fsync(f.fileno())
   if len(rawbytes)>2*1024*1024:raise ValueError('oversized response')
   raw=json.loads(rawbytes);u=usage(raw)
   for k,v in u.items():totals[k]=totals.get(k,0)+v
   s=validate_response(raw);checks={}
   for k,t in c['checks'].items():
    d,op=k.rsplit('_',1);checks[k]=s[d]>=t if op=='min' else s[d]<=t
   save(name+'-checks.json',{'scores':s,'checks':checks,'passed':all(checks.values())})
   print(name,'PASS' if all(checks.values()) else 'FAIL',flush=True)
   if not all(checks.values()):raise ValueError('predeclared score check failed: '+name)
   scores[name]=s
  pairs=[]
  for p in plan['pair_checks']:
   delta={k:abs(scores[p['left']][k]-scores[p['right']][k]) for k in scores[p['left']]}
   ok=all(v<=(p['max_weighted_delta'] if k=='weighted_score' else p['max_dimension_delta'])+1e-9 for k,v in delta.items());pairs.append({'pair':p,'delta':delta,'passed':ok})
  save('pair_checks.json',pairs)
  if not all(p['passed'] for p in pairs):raise ValueError('pair mismatch')
 except Exception as e:error=str(e) if isinstance(e,ValueError) else type(e).__name__
 finally:
  complete=len(list(OUT.glob('*-response.raw')))==calls and bool(totals)
  save('status.json',{'status':'FAILED_NOT_RELEASED' if error else 'PASS_PENDING_REVIEW','error':error,'attempts':calls,'accepted_cases':list(scores),'usage':totals,'usage_complete':complete,'formal_release':False})
 print('FINISHED',error or 'PASS',flush=True)
 return bool(error)
if __name__=='__main__':sys.exit(main())
