"""Bind retrieved SFT/DPO generations to the unchanged calibrated Gemini runner."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from week8_gemini_score_only import ROOT, canonical, prepare as prepare_base, run, save
from week8_score_recovery_audit import digest, read, require

SOURCE=ROOT/'reports/week8/phase2_trained_generation'
PLAN_SHA='1294064be32e9bf92263346f38b4d7a2e3b2338cfbb3cdea6f5d0cb284a11a8c'
SESSION_SHA='7fdea2b2307422cb538e53fbc4e8869f02109dc7f3e638326d3138ded58e9b8d'
BASE_SHA='3e9f731f159775f3fd3a663b4251e654f86508b827484d7b4deef369ab7a5458'


def prepare(model, source=SOURCE):
    require(model in ('final_sft','final_dpo'),'unknown model')
    source=Path(source); retrieved=source/'retrieved'
    manifest=read(retrieved/'retrieval_manifest.json')
    require(manifest['status']=='REMOTE_RESULT_VERIFICATION_PASS','retrieval not verified')
    require(len(manifest['files'])==750 and len({r['path'] for r in manifest['files']})==750,'manifest coverage')
    for row in manifest['files']:
        path=Path(row['path'])
        require(not path.is_absolute() and '..' not in path.parts,'unsafe artifact path')
        file=retrieved/path
        require(not file.is_symlink() and file.stat().st_size==row['bytes'] and digest(file)==row['sha256'],'retrieved artifact changed')
    release=source/'released_plan.json'
    require(digest(release)==PLAN_SHA and digest(source/'session_release.json')==SESSION_SHA,'release changed')
    plan=read(release)
    for path,h in plan['dependencies'].items():
        require(digest(ROOT/path)==h,'generation dependency changed')
    require(read(source/'retrieval_verification.json')['status']=='ALL_750_ARTIFACTS_SHA256_VERIFIED','retrieval review missing')
    review=read(source/'generation_verification.json')
    require(review['status']=='LOCAL_RESULTS_VERIFIED_JUDGE_PENDING','generation review missing')
    root=retrieved/'logs/trained-generation-20260917'
    require(read(root/'launch.json')['release_sha256']==SESSION_SHA,'session launch mismatch')
    require(read(root/'status.json')=={'status':'GENERATED_PENDING_REVIEW','completed_models':['final_sft','final_dpo'],'judge_calls':0},'session incomplete')
    directory=root/model
    require(read(directory/'launch.json')=={'model_name':model,'plan_sha256':PLAN_SHA,'gpu_host':plan['target_host'],'judge_calls':0},'model launch mismatch')
    lock=plan['models'][model]
    require(digest(ROOT/lock['lock_path'])==lock['lock_sha256'] and read(directory/'runtime_lock.json')==read(ROOT/lock['lock_path']),'runtime lock mismatch')
    require(Path(read(directory/'runtime_lock.json')['model_path']).name==model,'model path mismatch')
    answer_sha=digest(directory/'answers.jsonl')
    require(answer_sha==review['models'][model]['answers_sha256']==manifest['models'][model]['answers_sha256'],'answer receipt mismatch')
    require(read(directory/'status.json')=={'status':'GENERATED_PENDING_REVIEW','model_name':model,'questions':20,'answers_sha256':answer_sha,'judge_calls':0},'generation status mismatch')
    answers=[json.loads(line) for line in (directory/'answers.jsonl').read_text().splitlines()]
    base_path=ROOT/'reports/week8/phase2_gemini_score_release/plan.json'
    require(digest(base_path)==BASE_SHA,'base release changed')
    base=read(base_path)
    require(base==prepare_base(ROOT/base['source'],base['source_manifest_sha256'],ROOT/base['calibration']),'base protocol/dependencies changed')
    require([a['id'] for a in answers]==[r['question_id'] for r in base['requests']],'answer ID/order mismatch')
    require(all(isinstance(a['answer'],str) and a['answer'].strip() for a in answers),'empty answer')
    requests=copy.deepcopy(base['requests'])
    for row,answer in zip(requests,answers):
        part=row['body']['contents'][0]['parts'][0]
        inp=json.loads(part['text']);inp['candidate_answer']=answer['answer']
        part['text']=json.dumps(inp,ensure_ascii=False)
    template={k:requests[0]['body'][k] for k in ('systemInstruction','generationConfig')}
    result={k:copy.deepcopy(base[k]) for k in ('schema','calibration','calibration_receipt_sha256','profile_sha256','dependencies','max_calls','max_estimated_usd','historical_scores_reused','generation_reused','model')}
    result.update(scope='trained_custom20_gemini_score_only',model_name=model,answers_sha256=answer_sha,requests=requests,
        identity=hashlib.sha256(canonical({'answers':answer_sha,'protocol':template,'model':base['model']})).hexdigest(),
        source_manifest_sha256=digest(retrieved/'retrieval_manifest.json'),generation_plan_sha256=PLAN_SHA,session_release_sha256=SESSION_SHA,
        review_hashes={name:digest(source/name) for name in ('retrieval_verification.json','generation_verification.json')})
    result['dependencies']['scripts/week8_trained_score.py']=digest(ROOT/'scripts/week8_trained_score.py')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['final_sft','final_dpo'],required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--plan-sha256')
    args=parser.parse_args()
    plan=prepare(args.model)
    if not args.execute:
        save(args.plan,plan)
        print('PREPARED',args.model,digest(args.plan));return
    require(args.plan_sha256 and digest(args.plan)==args.plan_sha256,'release hash mismatch')
    require(read(args.plan)==plan,'plan changed since review')
    run(plan,ROOT/('reports/week8/phase2_gemini_score_'+args.model),ROOT/'reports/week8/score_only_claims')

if __name__=='__main__':main()
