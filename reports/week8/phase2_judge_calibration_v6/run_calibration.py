"""Frozen semantic calibration; reuse strict adapter and its bounded JSON retries."""
import hashlib
import json
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'scripts'))
import week8_deepseek_judge as judge
from step3_eval import judge_balance
from week8_score_only import save


def main():
    manifest = OUT/'manifest.json'
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == sys.argv[1]
    for relative, digest in json.loads(manifest.read_text())['files'].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() == digest, relative
    plan = json.loads((OUT/'plan.json').read_text())
    cases = json.loads((OUT/'cases.json').read_text())
    rubric = json.loads((ROOT/'deliverables/week3/day14/source/data/evaluation_rubric.json').read_text())
    prompt = (OUT/'system_prompt.txt').read_text()
    assert len(cases) == plan['cases'] == 14
    save(OUT/'started.json', {'manifest_sha256': sys.argv[1]})
    # Local process-only override: historical formal adapter file stays unchanged.
    original = judge.make_request_body
    def make_body(*args, **kwargs):
        body = original(*args, **kwargs)
        body['reasoning_effort'] = plan['reasoning_effort']
        body['max_tokens'] = plan['max_tokens']
        return body
    judge.make_request_body = make_body
    scores = {}; calls = 0; error = None
    try:
        start = judge_balance(); save(OUT/'starting_balance.json', {'cny':str(start)})
        def guard():
            nonlocal calls
            current = judge_balance()
            if current < Decimal(plan['reserve_cny']) or start-current >= Decimal(plan['max_observed_spend_cny']) or calls >= plan['max_calls']:
                raise ValueError('budget reached')
            calls += 1
            save(OUT/f'balance-before-{calls:02d}.json', {'cny':str(current), 'observed_decrease_cny':str(start-current)})
        for case in cases:
            result = judge.call_bounded(case['question'], case['answer'], rubric, judge.MODEL,
                OUT/case['id'], before_attempt=guard, system_prompt=prompt)
            if result['status'] != 'accepted' or not result['usage_complete']:
                raise ValueError('response rejected: '+case['id'])
            score = result['scores']; checks = {}
            for key, threshold in case['checks'].items():
                dim, op = key.rsplit('_',1)
                checks[key] = score[dim] >= threshold if op=='min' else score[dim] <= threshold
            save(OUT/(case['id']+'-checks.json'), {'scores':score, 'checks':checks, 'passed':all(checks.values())})
            print(case['id'], 'PASS' if all(checks.values()) else 'FAIL', 'attempts',result['attempts'],flush=True)
            if not all(checks.values()): raise ValueError('predeclared score check failed: '+case['id'])
            scores[case['id']] = score
        pairs=[]
        for pair in plan['pair_checks']:
            delta={k:abs(scores[pair['left']][k]-scores[pair['right']][k]) for k in scores[pair['left']]}
            passed=all(v <= (pair['max_weighted_delta'] if k=='weighted_score' else pair['max_dimension_delta'])+1e-9 for k,v in delta.items())
            pairs.append({'pair':pair,'delta':delta,'passed':passed})
        save(OUT/'pair_checks.json', pairs)
        if not all(p['passed'] for p in pairs): raise ValueError('paired stability check failed')
    except Exception as exc:
        error=type(exc).__name__
        print('CALIBRATION_STOPPED',error,flush=True)
    finally:
        usage=dict.fromkeys(['prompt_tokens','completion_tokens','total_tokens'],0);complete=True
        markers=list(OUT.glob('*/attempt-*.started'))
        for marker in markers:
            f=marker.with_suffix('.json')
            if not f.exists():complete=False;continue
            raw=json.loads(f.read_text())['response'];u=raw.get('usage',{})
            if not all(type(u.get(k)) is int and u[k]>=0 for k in usage) or u['total_tokens']!=u['prompt_tokens']+u['completion_tokens']:
                complete=False;continue
            for k in usage:usage[k]+=u[k]
        save(OUT/'status.json', {'status':'FAILED_NOT_RELEASED' if error else 'SMALL_BATCH_PASS_PENDING_REVIEW',
            'error_type':error,'accepted_cases':list(scores),'attempts':len(markers),'usage':usage,'usage_complete':complete,'formal_release':False})
        try:save(OUT/'latest_balance.json',{'cny':str(judge_balance())})
        except Exception:pass
    return int(error is not None)


if __name__=='__main__':sys.exit(main())
