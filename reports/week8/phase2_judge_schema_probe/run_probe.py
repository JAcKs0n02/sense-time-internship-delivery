"""One isolated Responses-schema compatibility probe; never retries or releases scores."""
import json, sys, os, hashlib, urllib.request, urllib.error
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
from setup_week8_judge_key import load_key
from step3_eval import judge_balance

def save(name,value):
    with (OUT/name).open('x') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())

def prepare():
    source=ROOT/'reports/week8/phase2_judge_calibration_v6'
    case=next(c for c in json.loads((source/'cases.json').read_text()) if c['id']=='V3-01')
    dims=['accuracy','completeness','logic','safety','format']
    item={'type':'object','properties':{'score':{'type':'number','minimum':0,'maximum':5},'reason':{'type':'string'}},'required':['score','reason'],'additionalProperties':False}
    schema={'type':'object','properties':{d:item for d in dims},'required':dims,'additionalProperties':False}
    body={'model':'deepseek-flash','instructions':(source/'system_prompt.txt').read_text(),'input':json.dumps({'question':case['question'],'candidate_answer':case['answer'],'rubric':json.loads((ROOT/'deliverables/week3/day14/source/data/evaluation_rubric.json').read_text())},ensure_ascii=False),'reasoning':{'effort':'low'},'max_output_tokens':8192,'stream':False,'text':{'format':{'type':'json_schema','name':'judge_score','schema':schema}}}
    save('request.json',body)
    save('plan.json',{'endpoint':'https://api.deepseek.com/responses','case':case,'max_calls':1,'automatic_retry':False,'formal_release':False,'reserve_cny':1,'request_sha256':hashlib.sha256((OUT/'request.json').read_bytes()).hexdigest(),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})

def run():
    plan=json.loads((OUT/'plan.json').read_text())
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==plan['runner_sha256']
    data=(OUT/'request.json').read_bytes()
    assert hashlib.sha256(data).hexdigest()==plan['request_sha256']
    balance=judge_balance(); assert balance>=1
    save('started.json',{'balance_cny':str(balance),'max_calls':1})
    req=urllib.request.Request(plan['endpoint'],data=data,headers={'Content-Type':'application/json','Authorization':'Bearer '+load_key()})
    try:
        with urllib.request.urlopen(req,timeout=180) as response: raw=response.read(2*1024*1024+1)
        with (OUT/'response.raw').open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        save('status.json',{'status':'RECEIVED_PENDING_INDEPENDENT_REVIEW','formal_release':False})
    except urllib.error.HTTPError as e:
        # Error body can be inspected locally; never contains request headers.
        save('http_error.json',{'status':e.code,'body':e.read(16384).decode(errors='replace')})
        save('status.json',{'status':'HTTP_REJECTED_NO_RETRY','http_status':e.code,'formal_release':False})
    except Exception as e:
        save('status.json',{'status':'UNKNOWN_OR_FAILED_NO_RETRY','error_type':type(e).__name__,'formal_release':False})
    print((OUT/'status.json').read_text())

if __name__=='__main__':
    {'prepare':prepare,'run':run}[sys.argv[1]]()
