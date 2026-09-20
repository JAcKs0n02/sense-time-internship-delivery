"""Offline audit and candidate request preparation; never invokes a model or API.

This does not authorize recovery. A changed judge token cap needs calibration
and a new release; existing scores remain historical evidence, not mixed scores.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from week8_deepseek_judge import make_request_body, validate_response

RUN = Path('logs/formal-resume-20260917-01')
TOKENS = ('prompt_tokens', 'completion_tokens', 'total_tokens')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def audit(source, manifest_sha256, repo):
    source, repo = Path(source).resolve(), Path(repo).resolve()
    manifest_path = source / 'manifest.json'
    require(digest(manifest_path) == manifest_sha256, 'manifest hash mismatch')
    entries = read(manifest_path)['files']
    names = [r['path'] for r in entries]
    require(len(names) == len(set(names)), 'duplicate manifest path')
    actual = set()
    for path in source.rglob('*'):
        require(not path.is_symlink(), 'symlink in evidence')
        if path.is_file() and path != manifest_path:
            actual.add(str(path.relative_to(source)))
    require(actual == set(names), 'unexpected or missing evidence file')
    for row in entries:
        path = Path(row['path'])
        require(not path.is_absolute() and '..' not in path.parts, 'unsafe manifest path')
        p = source / path
        require(p.stat().st_size == row['size'] and digest(p) == row['sha256'], 'evidence size/hash mismatch')
    run = source / RUN
    model = run / 'original_base'
    require(read(run / 'status.json')['status'] == 'FAILED_OR_TIMEOUT', 'source must be stopped failure')
    require(read(run / 'session_exit.json')['preflight_restored'] is True, 'preflight not restored')
    require(not (run / 'final_sft').exists() and not (run / 'final_dpo').exists(), 'unexpected later model')
    lock_path = model / 'resume_runtime_lock.json'
    require(digest(lock_path) == read(run / 'launch.json')['locks']['original_base'], 'source model lock changed')
    lock = read(lock_path)
    profile_path = repo / lock['judge_profile']['path']
    require(digest(profile_path) == lock['judge_profile']['sha256'], 'judge profile hash changed')
    profile = read(profile_path)
    for name in ('evaluation_questions.json', 'evaluation_rubric.json'):
        relative = 'deliverables/week3/day14/source/data/' + name
        require(digest(repo / relative) == profile['artifacts'][relative], 'question/rubric changed')
    spec = read(repo / 'deliverables/week3/day14/source/data/evaluation_questions.json')
    rubric = read(repo / 'deliverables/week3/day14/source/data/evaluation_rubric.json')
    questions = spec['questions']; ids = [q['id'] for q in questions]
    require(len(ids) == 20 and len(set(ids)) == 20, 'invalid question set')
    lines = [json.loads(line) for line in (model / 'answers.jsonl').read_text().splitlines()]
    require(len(lines) == 20 and [r['id'] for r in lines] == ids, 'answer IDs missing, duplicated or reordered')
    require(all(isinstance(r['answer'], str) and r['answer'].strip() for r in lines), 'empty or invalid answer')
    answers = {r['id']: r['answer'] for r in lines}
    require({p.name for p in (model / 'judge').iterdir()} <= set(ids), 'unknown judge directory')
    accepted, truncated, missing, rejected, candidate = [], [], [], [], []
    total = dict.fromkeys(TOKENS, 0); attempts = 0
    for q in questions:
        qid = q['id']; directory = model / 'judge' / qid
        body = make_request_body(q, answers[qid], rubric, profile['request_template']['messages'][0]['content'])
        template = copy.deepcopy(body); template['messages'] = template['messages'][:1]
        require(template == profile['request_template'], 'request adapter differs from released protocol')
        proposed = copy.deepcopy(body); proposed['max_tokens'] = 8192
        candidate.append({'question_id': qid, 'body': proposed})
        if not directory.exists():
            missing.append(qid); continue
        result = read(directory / 'result.json')
        count = result['attempts']
        require(type(count) is int and 1 <= count <= 2, 'unexpected attempts')
        expected = {'result.json'} | {f'attempt-{i}.{suffix}' for i in range(1, count + 1) for suffix in ('json', 'started')}
        require({p.name for p in directory.iterdir()} == expected, 'unknown or incomplete paid attempt')
        subtotal = dict.fromkeys(TOKENS, 0); last_scores = None; finish = None
        for i in range(1, count + 1):
            require(read(directory / f'attempt-{i}.started') == {'attempt': i}, 'invalid attempt marker')
            raw = read(directory / f'attempt-{i}.json')
            require(raw['request'] == body, 'paid request differs from bound answer or protocol')
            response = raw['response']; usage = response.get('usage', {})
            require(all(type(usage.get(k)) is int and usage[k] >= 0 for k in TOKENS), 'invalid usage')
            require(usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens'], 'inconsistent usage')
            require(response.get('model') == lock['judge']['model'], 'judge identity mismatch')
            choices = response.get('choices')
            require(isinstance(choices, list) and len(choices) == 1, 'invalid choices')
            finish = choices[0].get('finish_reason')
            try:
                last_scores = validate_response(response, lock['judge']['model'])
                require(i == count, 'request made after successful score')
            except ValueError as exc:
                # Do not turn a provenance/identity error into a reusable score.
                require(raw.get('validation_error') == str(exc), 'validation error differs from raw response')
                last_scores = None
                if i < count:
                    require(raw.get('retryable') is True and finish == 'stop', 'unauthorized historical retry')
            for key in TOKENS: subtotal[key] += usage[key]
        require(result['usage_complete'] is True and result['usage'] == subtotal, 'result usage mismatch')
        require(result['scores'] == last_scores, 'saved scores differ from raw response')
        require(result['status'] == ('accepted' if last_scores is not None else 'rejected'), 'result status mismatch')
        if last_scores is not None: accepted.append(qid)
        elif finish == 'length': truncated.append(qid)
        else: rejected.append(qid)
        attempts += count
        for key in TOKENS: total[key] += subtotal[key]
    require(read(model / 'judge_usage.json') == {'attempts': attempts, 'usage': total,
            'usage_complete': True, 'completed_questions': len(accepted)}, 'session usage mismatch')
    return {'status': 'OFFLINE_AUDIT_PASS_PENDING_CALIBRATION', 'formal_scoring_ready': False,
            'source_manifest_sha256': manifest_sha256, 'answers_sha256': digest(model / 'answers.jsonl'),
            'verified_files': len(entries), 'accepted_ids': accepted, 'truncated_ids': truncated,
            'rejected_ids': rejected, 'unattempted_ids': missing, 'attempts': attempts, 'usage': total,
            'candidate_protocol': {'max_tokens': 8192, 'calibrated': False,
                'historical_scores': 'preserve separately; do not mix with changed protocol',
                'scope': 'same request policy for all three models; all 20 candidates prepared'},
            'candidate_requests': candidate}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.resolve().is_relative_to(args.source.resolve()), 'output must be outside evidence')
    result = audit(args.source, args.manifest_sha256, Path(__file__).resolve().parents[1])
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(result['status'])


if __name__ == '__main__': main()
