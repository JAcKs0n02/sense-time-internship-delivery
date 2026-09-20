import sys,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
root=Path('/Users/yifanren/Documents/商汤科技实习');sys.path.insert(0,str(root/'scripts'))
from week8_trained_score import prepare
from week8_gemini_score_only import canonical,save
from week8_gemini_judge import validate_response,usage
from week8_score_recovery_audit import digest,read
model=sys.argv[1];out=root/('reports/week8/phase2_gemini_score_'+model)
plan=read(out/'plan.json');assert plan==prepare(model)
assert read(out/'status.json')['status']=='COMPLETED_PENDING_INDEPENDENT_REVIEW'
assert len(list(out.glob('judge/*/started.json')))==20
assert len(list(out.glob('judge/*/response.json')))==20
scores=[];totals={};dims={}
for row in plan['requests']:
 d=out/'judge'/row['question_id'];raw=read(d/'response.json')
 assert read(d/'started.json')=={'request_sha256':hashlib.sha256(canonical(row['body'])).hexdigest()}
 assert raw['request']==row['body']
 score=validate_response(raw['response']);assert score==read(d/'result.json')
 for k,v in usage(raw['response']).items():totals[k]=totals.get(k,0)+v
 for k,v in score.items():
  if k!='weighted_score':dims[k]=dims.get(k,0)+v
 scores.append({'question_id':row['question_id'],**score})
summary=read(out/'summary.json');mean=round(sum(s['weighted_score'] for s in scores)/20,6)
assert summary['scores']==scores and summary['weighted_mean']==mean
assert summary['plan_hash']==hashlib.sha256(canonical(plan)).hexdigest()
u=read(out/'usage.json');assert u=={'attempts':20,'usage_complete':True,'usage':{'prompt_tokens':totals['promptTokenCount'],'completion_tokens':totals['candidatesTokenCount']+totals['thoughtsTokenCount'],'total_tokens':totals['totalTokenCount']}}
receipt={'status':'VERIFIED_COMPLETE','verified_utc':datetime.now(timezone.utc).isoformat(),'api_calls':20,'valid_scores':20,'issues':[],'usage':totals,'generation_reused':True,'historical_scores_reused':False,'gpu_started':False,'evidence_kind':'ai_judge_not_human','weighted_mean':mean,'dimension_means':{k:v/20 for k,v in dims.items()},'files':{str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file() and p.name!='verification.json'}}
save(out/'verification.json',receipt)
print(model,'VERIFIED_COMPLETE',mean,totals)
