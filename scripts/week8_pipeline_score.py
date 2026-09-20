"""Route the pipeline to frozen Gemini score recovery without repeating generation."""
import argparse
import hashlib
from pathlib import Path
import week8_gemini_score_only as scoring
from week8_gemini_judge import validate_response, usage
from week8_score_recovery_audit import read, digest, require
from common import write_json

ROOT=Path(__file__).resolve().parents[1]
RELEASES={
    'original_base':('reports/week8/phase2_gemini_score_release/plan.json','3e9f731f159775f3fd3a663b4251e654f86508b827484d7b4deef369ab7a5458'),
    'final_sft':('reports/week8/phase2_gemini_trained_release/final_sft.json','2dfdb50fc4f7faec1e34e214df9d2b46f762aaa8e511063e6638d8714f7c898e'),
    'final_dpo':('reports/week8/phase2_gemini_trained_release/final_dpo.json','40b9b551219253c688560d19b9408f4b019db0466f267faf69cea09ce0b5cf01'),
}


def score_directory(model):
    require(model in RELEASES,'unknown score model')
    return ROOT/('reports/week8/phase2_gemini_score_'+model)


def load_release(model):
    require(model in RELEASES,'unknown score model')
    relative,expected=RELEASES[model];path=ROOT/relative
    require(not path.is_symlink() and digest(path)==expected,'release hash mismatch')
    plan=read(path)
    if model=='original_base':
        current=scoring.prepare(ROOT/plan['source'],plan['source_manifest_sha256'],ROOT/plan['calibration'])
    else:
        from week8_trained_score import prepare
        current=prepare(model)
    require(plan==current,'released source/code/calibration changed')
    return plan


def verify_completed(plan,output):
    """Recompute every saved score and usage total without changing the source."""
    output=Path(output)
    require(output.is_dir() and not output.is_symlink(),'missing or symlink score output')
    require(not any(p.is_symlink() for p in output.rglob('*')),'symlink score evidence')
    require(read(output/'plan.json')==plan,'saved plan mismatch')
    require(read(output/'status.json')=={'status':'COMPLETED_PENDING_INDEPENDENT_REVIEW','completed_questions':20},'scores incomplete or failed')
    ids=[r['question_id'] for r in plan['requests']]
    require(len(ids)==20 and len(set(ids))==20,'invalid request coverage')
    require({p.name for p in (output/'judge').iterdir()}==set(ids),'judge question set mismatch')
    scores=[];totals={'prompt_tokens':0,'completion_tokens':0,'total_tokens':0}
    for row in plan['requests']:
        directory=output/'judge'/row['question_id']
        require({p.name for p in directory.iterdir()}=={'started.json','response.json','result.json'},'unknown or missing paid attempt evidence')
        require(read(directory/'started.json')=={'request_sha256':hashlib.sha256(scoring.canonical(row['body'])).hexdigest()},'paid marker mismatch')
        raw=read(directory/'response.json');require(raw['request']==row['body'],'request differs from bound answer/protocol')
        score=validate_response(raw['response']);require(score==read(directory/'result.json'),'saved score differs from raw response')
        u=usage(raw['response'])
        totals['prompt_tokens']+=u['promptTokenCount']
        totals['completion_tokens']+=u['candidatesTokenCount']+u['thoughtsTokenCount']
        totals['total_tokens']+=u['totalTokenCount']
        scores.append({'question_id':row['question_id'],**score})
    require(totals['total_tokens']==totals['prompt_tokens']+totals['completion_tokens'],'inconsistent token usage')
    require(read(output/'usage.json')=={'attempts':20,'usage':totals,'usage_complete':True},'saved usage mismatch')
    mean=round(sum(x['weighted_score'] for x in scores)/20,6)
    expected={'questions':20,'scores':scores,'weighted_mean':mean,'evidence_kind':'ai_judge_not_human','generation_reused':True,'historical_scores_reused':False,'plan_hash':hashlib.sha256(scoring.canonical(plan)).hexdigest()}
    require(read(output/'summary.json')==expected,'summary mismatch')
    return {'questions':20,'weighted_mean':mean,'historical_api_attempts':20,'usage':totals,'evidence_kind':'verified_existing_ai_judge_scores','plan_hash':expected['plan_hash'],'source':str(output.relative_to(ROOT)) if output.is_relative_to(ROOT) else str(output)}


def integrate(model,report_dir,*,execute=False):
    require(model in (*RELEASES,'all'),'unknown score model')
    require(not (execute and model=='all'),'execute requires a single model')
    models=list(RELEASES) if model=='all' else [model]
    plans={name:load_release(name) for name in models}
    report_dir=Path(report_dir).resolve()
    preserved=[ROOT/'reports/week8/phase2_trained_generation',ROOT/'reports/week8/score_only_claims']
    for plan in plans.values():
        preserved.append(ROOT/plan['calibration'])
        if 'source' in plan:
            preserved.append(ROOT/plan['source'])
    require(not any(report_dir.is_relative_to(p.resolve()) for p in preserved),'report must be outside preserved source/calibration/claims')
    for name in RELEASES:
        require(not report_dir.is_relative_to(score_directory(name).resolve()),'report must be outside preserved scores')
    report_dir.mkdir(parents=True,exist_ok=False)
    rows=[];new_calls=0
    try:
        for name,plan in plans.items():
            output=score_directory(name)
            completed=(output/'status.json').is_file() and read(output/'status.json').get('status')=='COMPLETED_PENDING_INDEPENDENT_REVIEW'
            if execute and not completed:
                before=len(list(output.glob('judge/*/started.json')))
                scoring.run(plan,output,ROOT/'reports/week8/score_only_claims')
                new_calls+=len(list(output.glob('judge/*/started.json')))-before
            rows.append({'model':name,**verify_completed(plan,output)})
        result={'status':'VERIFIED_EXISTING_SCORES' if new_calls==0 else 'SCORES_COMPLETED_AND_VERIFIED','mode':'score_only','new_api_calls':new_calls,'gpu_started':False,'generation_repeated':False,'models':rows}
        write_json(report_dir/'score_recovery.json',result)
        return result
    except Exception as exc:
        write_json(report_dir/'status.json',{'status':'FAILED','mode':'score_only','error_type':type(exc).__name__,'completed_models':[r['model'] for r in rows],'paid_attempts_may_exist':execute})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=[*RELEASES,'all'],required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    result=integrate(args.model,args.run_dir,execute=args.execute)
    print(result['status']+'; new_api_calls='+str(result['new_api_calls']))

if __name__=='__main__':main()
