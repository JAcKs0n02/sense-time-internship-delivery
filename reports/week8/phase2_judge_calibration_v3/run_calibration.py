import datetime
import hashlib
import json
import sys
import urllib.request
from decimal import Decimal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
import week8_deepseek_judge as judge
from setup_week8_judge_key import load_key

def save(name,obj):
    with (OUT/name).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2)

def balance():
    req=urllib.request.Request(judge.BASE_URL+'/user/balance',headers={'Authorization':'Bearer '+load_key()})
    with urllib.request.urlopen(req,timeout=30) as r:d=json.load(r)
    assert d['is_available']
    rows=[x for x in d['balance_infos'] if x['currency']=='CNY'];assert len(rows)==1
    return Decimal(rows[0]['total_balance'])

judge.SYSTEM_PROMPT+='\n'+(OUT/'scoring_addendum.txt').read_text()
cases=json.loads((OUT/'cases.json').read_text());assert len(cases)==6
rubric_file=ROOT/'deliverables/week3/day14/source/data/evaluation_rubric.json'
rubric=json.loads(rubric_file.read_text())
files=[OUT/'cases.json',OUT/'plan.json',OUT/'scoring_addendum.txt',Path(__file__),rubric_file,ROOT/'scripts/week8_deepseek_judge.py',ROOT/'scripts/step3_eval.py']
save('manifest.json',{'files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'prompt_sha256':hashlib.sha256(judge.SYSTEM_PROMPT.encode()).hexdigest(),'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
start=balance();save('starting_balance.json',{'cny':str(start)})
count=0

def guard():
    global count
    b=balance();assert b>=1 and start-b<Decimal('0.5') and count<12
    count+=1
    save(f'balance-before-{count:02d}.json',{'cny':str(b),'observed_decrease_cny':str(start-b)})

for c in cases:
    result=judge.call_bounded(c['question'],c['answer'],rubric,judge.MODEL,OUT/c['id'],before_attempt=guard)
    checks={}
    if result['scores'] is not None:
        for key,t in c['checks'].items():
            dim,op=key.rsplit('_',1);checks[key]=result['scores'][dim]>=t if op=='min' else result['scores'][dim]<=t
    save(c['id']+'-checks.json',{'checks':checks,'passed':result['scores'] is not None and all(checks.values())})
    print(c['id'],result,'checks',checks,flush=True)
    if result['status']!='accepted':break
end=balance();save('latest_balance.json',{'cny':str(end),'observed_decrease_cny':str(start-end)})
