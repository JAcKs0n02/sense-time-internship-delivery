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
from step3_eval import judge_balance
from week8_deepseek_judge import validate_response
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
    """Recheck source and calibration evidence before producing a release plan."""
    source, calibration, repo = Path(source).resolve(), Path(calibration).resolve(), Path(repo).resolve()
    evidence = audit(source, manifest_sha256, repo)
    review = read(calibration / 'verification.json')
    require(review['status'] == 'SMALL_BATCH_PASS', 'calibration not passed')
    for relative, expected in review['files'].items():
        path = Path(relative)
        require(not path.is_absolute() and '..' not in path.parts, 'unsafe calibration path')
        require(digest(calibration / path) == expected, 'calibration evidence hash changed')
    profile = read(calibration / 'judge_profile_candidate.json')
    require(profile['calibration_receipt_sha256'] == digest(calibration / 'verification.json'), 'calibration receipt changed')
    template = profile['request_template']
    require(template['max_tokens'] == 8192 and template['model'] == 'deepseek-flash', 'unexpected protocol')
    calibration_plan = read(calibration / 'plan.json')
    require(len(calibration_plan['cases']) == 7, 'incomplete calibration')
    scores = {}; usage = dict.fromkeys(TOKENS, 0)
    for case in calibration_plan['cases']:
        raw = read(calibration / (case['id'] + '-raw.json'))
        require(raw['request'] == case['body'], 'calibration request mismatch')
        score = validate_response(raw['response'], template['model'])
        for key, threshold in case['checks'].items():
            dim, op = key.rsplit('_', 1)
            require(score[dim] >= threshold if op == 'min' else score[dim] <= threshold, 'calibration check failed')
        scores[case['id']] = score
        for key in TOKENS: usage[key] += raw['response']['usage'][key]
    require(usage == review['usage'], 'calibration usage mismatch')
    for key in scores['V3-04']:
        require(abs(scores['V3-04'][key] - scores['V3-05'][key]) <= (0.5 if key == 'weighted_score' else 1), 'repeat check failed')
    for row in evidence['candidate_requests']:
        actual = dict(row['body']); actual['messages'] = actual['messages'][:1]
        require(actual == template, 'candidate does not match calibrated protocol')
    identity = hashlib.sha256(canonical({'answers': evidence['answers_sha256'], 'protocol': template})).hexdigest()
    dependencies = ['scripts/week8_score_only.py', 'scripts/week8_score_recovery_audit.py',
                    'scripts/week8_deepseek_judge.py', 'scripts/step3_eval.py', 'scripts/setup_week8_judge_key.py', 'scripts/common.py']
    return {'schema': 1, 'scope': 'original_base_custom20_score_only', 'identity': identity,
            'source': str(source.relative_to(repo)), 'source_manifest_sha256': manifest_sha256,
            'answers_sha256': evidence['answers_sha256'], 'calibration': str(calibration.relative_to(repo)),
            'calibration_receipt_sha256': digest(calibration / 'verification.json'),
            'profile_sha256': digest(calibration / 'judge_profile_candidate.json'),
            'dependencies': {p: digest(repo / p) for p in dependencies},
            'requests': evidence['candidate_requests'], 'max_calls': 20,
            'reserve_cny': '1.00', 'max_spend_cny': '1.00',
            'historical_scores_reused': False, 'generation_reused': True}


def call_api(body):
    data = json.dumps(body, ensure_ascii=False).encode()
    require(len(data) <= 65536, 'request too large')
    req = urllib.request.Request('https://api.deepseek.com/chat/completions', data=data,
          headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + load_key()})
    with urllib.request.urlopen(req, timeout=180) as response:
        data = response.read(2 * 1024**2 + 1)
    require(len(data) <= 2 * 1024**2, 'response too large')
    return json.loads(data)


def record_usage(output):
    markers = list(output.glob('judge/*/started.json'))
    usage = dict.fromkeys(TOKENS, 0); complete = True
    for marker in markers:
        response = marker.with_name('response.json')
        if not response.exists(): complete = False; continue
        raw = read(response)['response']; row = raw.get('usage', {})
        if (not all(type(row.get(k)) is int and row[k] >= 0 for k in TOKENS)
                or row['total_tokens'] != row['prompt_tokens'] + row['completion_tokens']):
            complete = False; continue
        for key in TOKENS: usage[key] += row[key]
    save(output / 'usage.json', {'attempts': len(markers), 'usage': usage, 'usage_complete': complete}, replace=True)


def run(plan, output, claims, call=call_api, balance=judge_balance):
    output, claims = Path(output).resolve(), Path(claims).resolve()
    rows = plan['requests']; ids = [r['question_id'] for r in rows]
    require(len(rows) == plan['max_calls'] == 20 and len(set(ids)) == 20, 'expected 20 unique requests')
    require(all(re.fullmatch(r'[A-Za-z0-9_-]+', q) for q in ids), 'unsafe question ID')
    require(all(r['body']['model'] == 'deepseek-flash' and r['body']['max_tokens'] == 8192 for r in rows), 'unexpected request policy')
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
            if markers:
                require((output / 'starting_balance.json').exists(), 'paid run missing starting balance')
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
                score = validate_response(raw['response'], row['body']['model'])
                if (directory / 'result.json').exists(): require(read(directory / 'result.json') == score, 'cached score mismatch')
                else: save(directory / 'result.json', score)
                cached[row['question_id']] = score
            start_path = output / 'starting_balance.json'
            start = Decimal(read(start_path)['cny']) if start_path.exists() else None
            for row in rows:
                qid = row['question_id']; directory = output / 'judge' / qid
                if qid in cached:
                    scores.append({'question_id': qid, **cached[qid]}); continue
                require(time.monotonic() < deadline, 'run deadline reached')
                current = balance()
                require(current.is_finite() and current >= Decimal(plan['reserve_cny']), 'balance below reserve')
                if start is None:
                    start = current; save(start_path, {'cny': str(start)})
                require(start - current < Decimal(plan['max_spend_cny']), 'observed spending limit reached')
                save(output / 'status.json', {'status': 'RUNNING', 'current_question': qid, 'completed_questions': len(scores)}, replace=True)
                save(directory / 'balance.json', {'cny': str(current), 'observed_decrease_cny': str(start-current)})
                save(directory / 'started.json', {'request_sha256': hashlib.sha256(canonical(row['body'])).hexdigest()})
                raw = call(row['body'])
                save(directory / 'response.json', {'request': row['body'], 'response': raw})
                record_usage(output)
                score = validate_response(raw, row['body']['model'])
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
