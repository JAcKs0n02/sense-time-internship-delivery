"""Bounded score-only execution of a separately reviewed immutable request plan."""
import argparse
from decimal import Decimal
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.request

from setup_week8_judge_key import load_key
from setup_week8_gemini_key import KEY_PATH
from week8_gemini_judge import validate_response, usage as gemini_usage, MODEL
from week8_score_recovery_audit import audit, digest, read, require, TOKENS

ROOT = Path(__file__).resolve().parents[1]


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()


def save(path, value, *, replace=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path.with_name(path.name + '.tmp') if replace else path
    with target.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    if replace: os.replace(target, path)


def prepare(source, manifest_sha256, calibration, repo=ROOT):
    source, calibration, repo = Path(source).resolve(), Path(calibration).resolve(), Path(repo).resolve()
    evidence = audit(source, manifest_sha256, repo)
    review = read(calibration/'verification.json')
    require(review['status']=='SMALL_BATCH_PASS_NOT_FORMAL_RELEASE' and review['valid_responses']==17 and not review['failures'], 'calibration failed')
    for name,h in review['files'].items():
        require(Path(name).name==name and digest(calibration/name)==h, 'calibration changed')
    for name,h in read(calibration/'manifest.json').items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and digest(repo/name)==h, 'frozen dependency changed')
    profile=read(calibration/'judge_profile_candidate.json')
    require(profile['verification_sha256']==digest(calibration/'verification.json'), 'receipt mismatch')
    require(profile['model']==MODEL, 'wrong profile model')
    template={'systemInstruction':profile['systemInstruction'],'generationConfig':profile['generationConfig']}
    cases=read(calibration/'cases.json');require(len(cases)==17, 'case count')
    scores={};totals={}
    for c in cases:
        body=read(calibration/(c['id']+'-request.json'))
        require({k:body[k] for k in template}==template, 'calibration protocol mismatch')
        inp=json.loads(body['contents'][0]['parts'][0]['text'])
        require(inp=={'question':c['question'],'candidate_answer':c['answer'],'rubric':profile['rubric']}, 'calibration input mismatch')
        raw=read(calibration/(c['id']+'-response.raw'));score=validate_response(raw)
        for k,t in c['checks'].items():
            d,op=k.rsplit('_',1);require(score[d]>=t if op=='min' else score[d]<=t, 'score check failed')
        scores[c['id']]=score
        for k,v in gemini_usage(raw).items():totals[k]=totals.get(k,0)+v
    require(totals==review['usage'], 'usage mismatch')
    for p in read(calibration/'plan.json')['pair_checks']:
        for k in scores[p['left']]:require(abs(scores[p['left']][k]-scores[p['right']][k])<=(p['max_weighted_delta'] if k=='weighted_score' else p['max_dimension_delta'])+1e-9, 'pair failed')
    requests=[]
    for row in evidence['candidate_requests']:
        inp=json.loads(row['body']['messages'][1]['content']);require(inp['rubric']==profile['rubric'], 'rubric mismatch')
        body={**template,'contents':[{'role':'user','parts':[{'text':json.dumps(inp,ensure_ascii=False)}]}]}
        requests.append({'question_id':row['question_id'],'body':body})
    identity=hashlib.sha256(canonical({'answers':evidence['answers_sha256'],'protocol':template,'model':MODEL})).hexdigest()
    deps=['week8_gemini_score_only.py','week8_score_recovery_audit.py','week8_gemini_judge.py','week8_deepseek_judge.py','step3_eval.py','setup_week8_judge_key.py','setup_week8_gemini_key.py','common.py']
    return {'schema':1,'scope':'original_base_custom20_gemini_score_only','identity':identity,'source':str(source.relative_to(repo)),'source_manifest_sha256':manifest_sha256,'answers_sha256':evidence['answers_sha256'],'calibration':str(calibration.relative_to(repo)),'calibration_receipt_sha256':digest(calibration/'verification.json'),'profile_sha256':digest(calibration/'judge_profile_candidate.json'),'dependencies':{'scripts/'+n:digest(repo/'scripts'/n) for n in deps},'requests':requests,'max_calls':20,'max_estimated_usd':'6.00','historical_scores_reused':False,'generation_reused':True,'model':MODEL}


def call_api(body):
    data=json.dumps(body,ensure_ascii=False).encode()
    require(len(data)<=32768,'request too large')
    req=urllib.request.Request('https://generativelanguage.googleapis.com/v1beta/models/'+MODEL+':generateContent',data=data,headers={'Content-Type':'application/json','x-goog-api-key':load_key(KEY_PATH)})
    with urllib.request.urlopen(req,timeout=240) as r:data=r.read(2*1024**2+1)
    require(len(data)<=2*1024**2,'response too large')
    return json.loads(data)


def record_usage(output):
    markers = list(output.glob('judge/*/started.json'))
    usage = dict.fromkeys(TOKENS, 0); complete = True
    for marker in markers:
        response = marker.with_name('response.json')
        if not response.exists(): complete = False; continue
        raw = read(response)['response']
        try:
            u=gemini_usage(raw); row={'prompt_tokens':u['promptTokenCount'],'completion_tokens':u['candidatesTokenCount']+u['thoughtsTokenCount'],'total_tokens':u['totalTokenCount']}
        except ValueError:
            complete=False;continue
        if (not all(type(row.get(k)) is int and row[k] >= 0 for k in TOKENS)
                or row['total_tokens'] != row['prompt_tokens'] + row['completion_tokens']):
            complete = False; continue
        for key in TOKENS: usage[key] += row[key]
    save(output / 'usage.json', {'attempts': len(markers), 'usage': usage, 'usage_complete': complete}, replace=True)


def run(plan, output, claims, call=call_api):
    output, claims = Path(output).resolve(), Path(claims).resolve()
    rows = plan['requests']; ids = [r['question_id'] for r in rows]
    require(len(rows) == plan['max_calls'] == 20 and len(set(ids)) == 20, 'expected 20 unique requests')
    require(all(re.fullmatch(r'[A-Za-z0-9_-]+', q) for q in ids), 'unsafe question ID')
    require(all(r['body']['generationConfig']['maxOutputTokens']==16384 and r['body']['generationConfig']['thinkingConfig']=={'thinkingLevel':'high'} and len(json.dumps(r['body'],ensure_ascii=False).encode())<=32768 for r in rows), 'unexpected request policy')
    # Conservative planning bound: charge every request byte as an input token,
    # all output allowance at $12/M, input at $2/M. Not a provider billing cap.
    bound=sum((len(json.dumps(r['body'],ensure_ascii=False).encode())*2+16384*12)/1000000 for r in rows)
    require(Decimal(str(bound))<=Decimal(plan['max_estimated_usd']), 'estimated budget exceeded')
    plan_hash = hashlib.sha256(canonical(plan)).hexdigest()
    identity = hashlib.sha256(plan['identity'].encode()).hexdigest()
    claims.mkdir(parents=True, exist_ok=True)
    with (claims / (identity + '.lock')).open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        claim = claims / (identity + '.json')
        expected = {'output': str(output), 'plan_hash': plan_hash}
        if claim.exists(): require(read(claim) == expected, 'existing claim forbids new output or changed plan')
        else:
            require(not output.exists(), 'unclaimed output already exists')
            save(claim, expected)
        output.mkdir(parents=True, exist_ok=True)
        if (output / 'plan.json').exists(): require(read(output / 'plan.json') == plan, 'saved plan mismatch')
        else: save(output / 'plan.json', plan)
        if (output / 'status.json').exists():
            require(read(output / 'status.json')['status'] != 'FAILED_NO_RETRY', 'failed run cannot automatically retry')
        require({p.name for p in (output / 'judge').glob('*')} <= set(ids), 'unknown question output')
        scores = []; deadline = time.monotonic() + 7200
        try:
            require(not any(p.is_symlink() for p in output.rglob('*')), 'symlink in run output')
            markers = list(output.glob('judge/*/started.json'))
            if (output / 'usage.json').exists():
                require(read(output / 'usage.json')['attempts'] <= len(markers), 'paid attempt evidence was removed')
            # Validate every existing response before issuing any new paid call.
            cached = {}
            for row in rows:
                directory = output / 'judge' / row['question_id']
                if not directory.exists(): continue
                require({p.name for p in directory.iterdir()} <= {'started.json', 'response.json', 'result.json', 'balance.json'}, 'unknown question files')
                marker, response = directory / 'started.json', directory / 'response.json'
                require(marker.exists() and response.exists(), 'unknown paid attempt; manual review required')
                require(read(marker) == {'request_sha256': hashlib.sha256(canonical(row['body'])).hexdigest()}, 'request marker mismatch')
                raw = read(response); require(raw['request'] == row['body'], 'cached request mismatch')
                score = validate_response(raw['response'])
                if (directory / 'result.json').exists(): require(read(directory / 'result.json') == score, 'cached score mismatch')
                else: save(directory / 'result.json', score)
                cached[row['question_id']] = score
            for row in rows:
                qid = row['question_id']; directory = output / 'judge' / qid
                if qid in cached:
                    scores.append({'question_id': qid, **cached[qid]}); continue
                require(time.monotonic() < deadline, 'run deadline reached')
                save(output / 'status.json', {'status': 'RUNNING', 'current_question': qid, 'completed_questions': len(scores)}, replace=True)
                save(directory / 'started.json', {'request_sha256': hashlib.sha256(canonical(row['body'])).hexdigest()})
                raw = call(row['body'])
                save(directory / 'response.json', {'request': row['body'], 'response': raw})
                record_usage(output)
                score = validate_response(raw)
                save(directory / 'result.json', score)
                scores.append({'question_id': qid, **score})
                print(f'{qid}: accepted ({len(scores)}/20)', flush=True)
            record_usage(output)
            save(output / 'summary.json', {'questions': 20, 'scores': scores,
                 'weighted_mean': round(sum(r['weighted_score'] for r in scores)/20, 6),
                 'evidence_kind': 'ai_judge_not_human', 'generation_reused': True,
                 'historical_scores_reused': False, 'plan_hash': plan_hash}, replace=True)
            save(output / 'status.json', {'status': 'COMPLETED_PENDING_INDEPENDENT_REVIEW', 'completed_questions': 20}, replace=True)
        except BaseException as exc:
            record_usage(output)
            save(output / 'status.json', {'status': 'FAILED_NO_RETRY', 'error_type': type(exc).__name__,
                 'completed_questions': len(scores)}, replace=True)
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--plan-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    require(digest(args.plan) == args.plan_sha256, 'release plan hash mismatch')
    plan = read(args.plan)
    require(plan == prepare(ROOT / plan['source'], plan['source_manifest_sha256'], ROOT / plan['calibration']), 'release plan no longer matches source/code/calibration')
    require(not args.output.resolve().is_relative_to((ROOT / plan['source']).resolve()), 'output must not modify source')
    run(plan, args.output, ROOT / 'reports/week8/score_only_claims')


if __name__ == '__main__': main()
