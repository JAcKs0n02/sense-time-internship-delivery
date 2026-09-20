"""Read-only independent reconstruction of all Gemini formal scores."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'reports/week8/phase2_gemini_score_original_base'
sys.path.insert(0,str(ROOT/'scripts'))
from week8_gemini_score_only import prepare,canonical
from week8_gemini_judge import validate_response,usage
plan=json.loads((OUT/'plan.json').read_text())
assert prepare(ROOT/plan['source'],plan['source_manifest_sha256'],ROOT/plan['calibration'])==plan
status=json.loads((OUT/'status.json').read_text());scores=[];totals={};issues=[]
for row in plan['requests']:
 d=OUT/'judge'/row['question_id']
 if not d.exists():continue
 assert json.loads((d/'started.json').read_text())=={'request_sha256':hashlib.sha256(canonical(row['body'])).hexdigest()}
 if not (d/'response.json').exists():issues.append({'question':row['question_id'],'error':'unknown response'});continue
 r=json.loads((d/'response.json').read_text());assert r['request']==row['body']
 for k,v in usage(r['response']).items():totals[k]=totals.get(k,0)+v
 try:s=validate_response(r['response'])
 except ValueError as e:issues.append({'question':row['question_id'],'error':str(e)});continue
 assert json.loads((d/'result.json').read_text())==s;scores.append({'question_id':row['question_id'],**s})
calls=len(list(OUT.glob('judge/*/started.json')))
u=json.loads((OUT/'usage.json').read_text());assert u['attempts']==calls
assert u['usage']=={'prompt_tokens':totals.get('promptTokenCount',0),'completion_tokens':totals.get('thoughtsTokenCount',0)+totals.get('candidatesTokenCount',0),'total_tokens':totals.get('totalTokenCount',0)}
passed=len(scores)==20 and calls==20 and not issues
result={'status':'VERIFIED_COMPLETE' if passed else 'INCOMPLETE_NOT_RELEASED','api_calls':calls,'valid_scores':len(scores),'issues':issues,'usage':totals,'generation_reused':True,'historical_scores_reused':False,'gpu_started':False,'evidence_kind':'ai_judge_not_human'}
if passed:
 summary=json.loads((OUT/'summary.json').read_text());assert summary['scores']==scores
 mean=round(sum(s['weighted_score'] for s in scores)/20,6);assert summary['weighted_mean']==mean
 result.update(weighted_mean=mean,dimension_means={k:round(sum(s[k] for s in scores)/20,6) for k in ['accuracy','completeness','logic','safety','format']})
result['files']={str(f.relative_to(OUT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in OUT.rglob('*') if f.is_file() and f.name!='verification.json'}
with (OUT/'verification.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps({k:v for k,v in result.items() if k!='files'},ensure_ascii=False,indent=2))
