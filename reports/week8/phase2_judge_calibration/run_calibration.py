"""Bounded one-off calibration driver; does not enable formal evaluation."""
import sys,json,hashlib,datetime,urllib.request
from pathlib import Path
from decimal import Decimal
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
from week8_deepseek_judge import call_once,MODEL
from setup_week8_judge_key import load_key
OUT=Path(__file__).resolve().parent
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def balance():
 req=urllib.request.Request('https://api.deepseek.com/user/balance',headers={'Authorization':'Bearer '+load_key()})
 with urllib.request.urlopen(req,timeout=30) as r:value=json.load(r)
 assert value['is_available']
 cny=[v for v in value['balance_infos'] if v['currency']=='CNY'];assert len(cny)==1
 return Decimal(cny[0]['total_balance'])
cases=json.loads((OUT/'cases.json').read_text());limit=int(sys.argv[1]);assert 1<=limit<=len(cases)==6
rubric_path=ROOT/'deliverables/week3/day14/source/data/evaluation_rubric.json';rubric=json.loads(rubric_path.read_text())
manifest=dict(cases_sha256=h(OUT/'cases.json'),rubric_sha256=h(rubric_path),adapter_sha256=h(ROOT/'scripts/week8_deepseek_judge.py'),score_validator_sha256=h(ROOT/'scripts/step3_eval.py'),max_calls=6,max_output_tokens_per_call=4096)
if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
else:save(OUT/'manifest.json',manifest)
if not (OUT/'starting_balance.json').exists():save(OUT/'starting_balance.json',{'cny':str(balance())})
start=Decimal(json.loads((OUT/'starting_balance.json').read_text())['cny'])
for c in cases[:limit]:
 path=OUT/(c['id']+'.json')
 if path.exists():
  assert json.loads(path.read_text()).get('checks_passed') is True,'prior failure requires review'
  continue
 assert not (OUT/(c['id']+'.started')).exists(),'uncertain previous request; no automatic retry'
 current=balance();assert current>=1 and start-current<Decimal('0.5'),'calibration spending guard'
 (OUT/(c['id']+'.started')).write_text(datetime.datetime.now(datetime.timezone.utc).isoformat())
 scores,audit=call_once(c['question'],c['answer'],rubric,MODEL)
 predicates={}
 if scores:
  for key,threshold in c['checks'].items():
   dim,op=key.rsplit('_',1);predicates[key]=scores[dim]>=threshold if op=='min' else scores[dim]<=threshold
 success=scores is not None and all(predicates.values())
 save(path,dict(case_id=c['id'],time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scores=scores,checks=predicates,checks_passed=success,**audit))
 after=balance();save(OUT/'latest_balance.json',{'cny':str(after),'observed_decrease_cny':str(start-after)})
 print(c['id'],'PASS' if success else 'FAIL', 'model',audit['response'].get('model'),'scores',scores,'usage',audit['response'].get('usage'),flush=True)
 assert success,'calibration failed; stop for review'
