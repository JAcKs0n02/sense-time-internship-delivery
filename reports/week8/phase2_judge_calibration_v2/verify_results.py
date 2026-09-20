"""Offline audit of the frozen calibration; no API calls or secret access."""
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
from week8_deepseek_judge import validate_response
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cases=json.loads((OUT/'cases.json').read_text())
plan=json.loads((OUT/'plan.json').read_text())
for manifest_name,driver in [('manifest.json','run_calibration.py'),('continuation_manifest.json','run_remaining.py'),('final_batch_manifest.json','run_final_batch.py')]:
    manifest=json.loads((OUT/manifest_name).read_text())
    for key,path in [('cases_sha256',OUT/'cases.json'),('rubric_sha256',ROOT/'deliverables/week3/day14/source/data/evaluation_rubric.json'),('adapter_sha256',OUT/'adapter_snapshot.py'),('score_validator_sha256',ROOT/'scripts/step3_eval.py'),('plan_sha256',OUT/'plan.json'),('driver_sha256',OUT/driver)]:
        assert manifest[key]==h(path),(manifest_name,key)
rows=[];usage={k:0 for k in ('prompt_tokens','completion_tokens','total_tokens')};responses={}
for c in cases:
    f=OUT/(c['id']+'.json');d=json.loads(f.read_text());responses[c['id']]=d
    raw=d['response'];error=None
    try:scores=validate_response(raw,'deepseek-flash')
    except ValueError as exc:scores=None;error=str(exc)
    assert scores==d['scores']
    if error:assert error==d['validation_error']
    checks={}
    if scores:
        for key,t in c['checks'].items():
            dim,op=key.rsplit('_',1);checks[key]=scores[dim]>=t if op=='min' else scores[dim]<=t
    assert checks==d['checks']
    passed=scores is not None and all(checks.values());assert passed==d['checks_passed']
    shape_exact=False
    if scores:
        parsed=json.loads(raw['choices'][0]['message']['content'])
        shape_exact=set(parsed)=={'accuracy','completeness','logic','safety','format'} and all(set(v)=={'score','reason'} for v in parsed.values())
    assert d['request']['max_tokens']==4096
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert hashlib.sha256(d['request']['messages'][0]['content'].encode()).hexdigest()==manifest['prompt_sha256']
    payload=json.loads(d['request']['messages'][1]['content']);assert payload['question']==c['question'] and payload['candidate_answer']==c['answer']
    for k in usage:usage[k]+=raw['usage'][k]
    rows.append(dict(case_id=c['id'],scores=scores,checks_passed=passed,error=error,schema_exact=shape_exact,sha256=h(f),completion_tokens=raw['usage']['completion_tokens']))
repeats=[]
for c in cases:
    if 'repeat_of' not in c:continue
    a=responses[c['id']]['scores'];b=responses[c['repeat_of']]['scores']
    delta={k:abs(a[k]-b[k]) for k in a} if a and b else None
    passed=delta is not None and delta['weighted_score']<=plan['repeat_max_weighted_delta'] and all(v<=plan['repeat_max_dimension_delta'] for k,v in delta.items() if k!='weighted_score')
    repeats.append(dict(case_id=c['id'],repeat_of=c['repeat_of'],deltas=delta,passed=passed))
result=dict(status='PASS' if all(r['checks_passed'] for r in rows) and all(r['passed'] for r in repeats) else 'NOT_APPROVED',formal_scoring_approved=False,calls=len(rows),valid_responses=sum(r['scores'] is not None for r in rows),case_checks_passed=sum(r['checks_passed'] for r in rows),usage=usage,cases=rows,repeats=repeats,latest_balance=json.loads((OUT/'latest_balance.json').read_text()))
(OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
