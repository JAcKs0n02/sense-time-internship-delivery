"""One explicitly bounded seven-request calibration; no retry or GPU usage."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request
from decimal import Decimal

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
from setup_week8_judge_key import load_key
from week8_deepseek_judge import validate_response
from step3_eval import judge_balance


def save(name, obj):
    with (OUT / name).open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def main():
    plan_path = OUT / 'plan.json'
    assert hashlib.sha256(plan_path.read_bytes()).hexdigest() == sys.argv[1]
    plan = json.loads(plan_path.read_text())
    assert len(plan['cases']) == plan['max_calls'] == 7
    assert plan['max_attempts_per_case'] == 1
    save('started.json', {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'plan_sha256': sys.argv[1], 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    accepted = {}; usage = dict.fromkeys(('prompt_tokens', 'completion_tokens', 'total_tokens'), 0)
    completed_calls = 0
    try:
        start = judge_balance()
        save('starting_balance.json', {'cny': str(start)})
        for index, case in enumerate(plan['cases'], 1):
            balance = judge_balance()
            assert balance >= Decimal(plan['reserve_cny']) and start - balance < Decimal(plan['max_observed_spend_cny'])
            save(f'balance-before-{index:02d}.json', {'cny': str(balance), 'observed_decrease_cny': str(start-balance)})
            body = case['body']
            assert body['model'] == 'deepseek-flash' and body['max_tokens'] == 8192
            data = json.dumps(body, ensure_ascii=False).encode()
            assert len(data) <= 65536
            request = urllib.request.Request('https://api.deepseek.com/chat/completions', data=data,
                headers={'Content-Type': 'application/json', 'Authorization': 'Bearer '+load_key()})
            save(case['id']+'.started', {'attempt': 1})
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = response.read(2*1024*1024+1)
            assert len(payload) <= 2*1024*1024
            raw = json.loads(payload)
            save(case['id']+'-raw.json', {'request': body, 'response': raw})
            completed_calls += 1
            score = validate_response(raw, 'deepseek-flash')
            for k in usage: usage[k] += raw['usage'][k]
            checks = {}
            for key, threshold in case['checks'].items():
                dimension, op = key.rsplit('_', 1)
                checks[key] = score[dimension] >= threshold if op == 'min' else score[dimension] <= threshold
            save(case['id']+'-checks.json', {'scores': score, 'checks': checks, 'passed': all(checks.values())})
            print(case['id'], 'PASS' if all(checks.values()) else 'FAIL', 'completion_tokens', raw['usage']['completion_tokens'], flush=True)
            assert all(checks.values()), 'predeclared calibration check failed'
            accepted[case['id']] = score
        a, b = accepted['V3-04'], accepted['V3-05']
        repeat = {k: abs(a[k]-b[k]) for k in a}
        save('repeat_checks.json', repeat)
        assert all(v <= (plan['repeat_max_weighted_delta'] if k == 'weighted_score' else plan['repeat_max_dimension_delta']) for k,v in repeat.items())
        save('status.json', {'status': 'SMALL_BATCH_PASS_PENDING_INDEPENDENT_REVIEW', 'completed_calls': completed_calls,
             'usage': usage, 'formal_release': False})
    except BaseException as exc:
        # Unknown transport attempts remain durable; no automatic replay.
        save('status.json', {'status': 'FAILED_NO_RETRY', 'error_type': type(exc).__name__,
             'completed_calls': completed_calls, 'accepted_cases': list(accepted), 'formal_release': False})
        print('FAILED_NO_RETRY', type(exc).__name__, flush=True)
        raise SystemExit(1)
    finally:
        try:
            save('latest_balance.json', {'cny': str(judge_balance())})
        except Exception:
            pass


if __name__ == '__main__': main()
