#!/usr/bin/env python3
"""Archived evidence replay or fresh OpenCompass + 20-question AI evaluation."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import urllib.request
from common import ROOT, load_module, sha256, write_json

WEIGHTS = {'accuracy': .30, 'completeness': .25, 'logic': .20, 'safety': .15, 'format': .10}


def validate_judgment(payload):
    result = {}
    for key in WEIGHTS:
        item = payload.get(key, {})
        score = item.get('score')
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 5:
            raise ValueError(f'invalid judge score: {key}')
        if not isinstance(item.get('reason'), str) or not item['reason'].strip():
            raise ValueError(f'missing judge rationale: {key}')
        result[key] = score
    result['weighted_score'] = round(sum(result[k]*w for k, w in WEIGHTS.items()), 6)
    return result


def write_csv(path, rows):
    if not rows:
        raise ValueError('refusing an empty evaluation summary')
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def replay(output):
    score_source = ROOT/'deliverables/week3/day15/source/results/opencompass_normalized_results.json'
    payload = json.loads(score_source.read_text())
    summarizer = load_module('week8_oc_summary', ROOT/'deliverables/week3/day15/source/scripts/summarize_opencompass.py')
    rows = summarizer.build_score_table(payload['results'] if isinstance(payload, dict) else payload)
    for row in rows:
        expected = 52 if row['dataset']=='ceval' else 67
        if int(row['subject_count']) != expected:
            raise ValueError('incomplete archived benchmark subject coverage')
        row['evidence_kind'] = 'archived_opencompass'
    human_source = ROOT/'deliverables/week3/day14/source/results/human_scores.csv'
    with human_source.open() as stream:
        human = list(csv.DictReader(stream))
    if len(human) != 10 or len({r['model_id'] for r in human}) != 10:
        raise ValueError('expected ten distinct historically reviewed models')
    for row in human:
        if int(row['answer_count']) != 20 or int(row['rater_count']) != 2:
            raise ValueError('incomplete archived human review')
        score = validate_judgment({k: {'score': float(row[k]), 'reason': 'archived two-human aggregate'} for k in WEIGHTS})
        if abs(score['weighted_score']-float(row['weighted_total'])) > 1e-5:
            raise ValueError('archived weighted score mismatch')
        rows.append({'model': row['model_id'], 'dataset': 'custom20', 'evidence_kind': 'archived_human_aggregate', **score})
    write_csv(output/'summary.csv', rows)
    write_json(output/'status.json', {'status': 'completed', 'mode': 'archived_evidence_replay', 'fresh_model_inference': False, 'rows': len(rows), 'sources': {str(p.relative_to(ROOT)): sha256(p) for p in [score_source, human_source]}})


def load_judge_profile(lock):
    from week8_deepseek_judge import make_request_body, MODEL, BASE_URL
    ref=lock.get('judge_profile')
    if not isinstance(ref,dict):
        raise ValueError('runtime lock needs a bound judge profile')
    path=ROOT/ref['path']
    if sha256(path)!=ref['sha256']:
        raise ValueError('judge profile artifact changed')
    profile=json.loads(path.read_text())
    if profile.get('base_url')!=BASE_URL or profile.get('max_attempts_per_case')!=2:
        raise ValueError('unsupported judge endpoint or retry policy')
    source=profile['calibrated_source']
    source_path=ROOT/source['path']
    if sha256(source_path)!=source['sha256']:
        raise ValueError('calibrated judge source changed')
    calibrated=json.loads(source_path.read_text())
    receipt=ROOT/calibrated['calibration_receipt']
    if sha256(receipt)!=calibrated['calibration_receipt_sha256'] or json.loads(receipt.read_text())['status']!='SMALL_BATCH_PASS':
        raise ValueError('calibration receipt changed or not passed')
    if profile['request_template']!=calibrated['request_template']:
        raise ValueError('judge request differs from calibrated profile')
    template=profile['request_template']
    if template['model']!=MODEL or len(template['messages'])!=1 or template['messages'][0]['role']!='system':
        raise ValueError('invalid judge template')
    body=make_request_body({},'',{},system_prompt=template['messages'][0]['content'])
    body['messages']=body['messages'][:1]
    if body!=template:
        raise ValueError('runtime judge parameters differ from calibration')
    expected={'scripts/step3_eval.py','scripts/week8_deepseek_judge.py','scripts/setup_week8_judge_key.py',
              'deliverables/week3/day14/source/data/evaluation_questions.json',
              'deliverables/week3/day14/source/data/evaluation_rubric.json'}
    artifacts=profile.get('artifacts',{})
    if set(artifacts)!=expected:
        raise ValueError('judge code/data bindings incomplete')
    for name,digest in artifacts.items():
        if sha256(ROOT/name)!=digest:raise ValueError('judge code/data artifact changed: '+name)
    return profile


def judge_balance():
    from decimal import Decimal
    from setup_week8_judge_key import load_key
    from week8_deepseek_judge import BASE_URL
    request=urllib.request.Request(BASE_URL+'/user/balance',headers={'Authorization':'Bearer '+load_key()})
    with urllib.request.urlopen(request,timeout=30) as response:
        value=json.load(response)
    rows=[r for r in value['balance_infos'] if r['currency']=='CNY']
    if not value['is_available'] or len(rows)!=1:
        raise ValueError('judge account unavailable or unexpected currency')
    balance=Decimal(rows[0]['total_balance'])
    if not balance.is_finite() or balance<0:raise ValueError('invalid judge balance')
    return balance


def score_custom20(spec, answers, rubric, profile, output, starting_balance):
    from decimal import Decimal
    import re
    from week8_deepseek_judge import call_bounded, MODEL
    questions=spec['questions'];ids=[q['id'] for q in questions]
    if len(ids)!=20 or len(set(ids))!=20 or set(answers)!=set(ids):
        raise ValueError('custom set needs exactly 20 matching answers')
    if any(not isinstance(i,str) or re.fullmatch(r'[A-Za-z0-9_-]+',i) is None for i in ids):
        raise ValueError('unsafe question identifier')
    if any(not isinstance(a,str) for a in answers.values()):raise ValueError('invalid answer type')
    count=0;rows=[]
    def guard():
        nonlocal count
        balance=judge_balance()
        if count>=40 or balance<1 or starting_balance-balance>=Decimal('1.00'):
            raise ValueError('judge request or observed spending budget reached')
        count+=1
        write_json(output/'judge_balance'/f'{count:02d}.json',{'cny':str(balance),'observed_decrease_cny':str(starting_balance-balance)})
    def record_usage():
        total={k:0 for k in ('prompt_tokens','completion_tokens','total_tokens')};complete=True
        markers=list((output/'judge').glob('*/attempt-*.started'))
        for marker in markers:
            f=marker.with_suffix('.json')
            if not f.exists():complete=False;continue
            raw=json.loads(f.read_text()).get('response',{});usage=raw.get('usage',{})
            if (not all(type(usage.get(k)) is int and usage[k]>=0 for k in total)
                    or usage['total_tokens']!=usage['prompt_tokens']+usage['completion_tokens']):
                complete=False;continue
            for k in total:total[k]+=usage[k]
        write_json(output/'judge_usage.json',{'attempts':len(markers),'usage':total,'usage_complete':complete,'completed_questions':len(rows)})
    for q in questions:
        try:
            result=call_bounded(q,answers[q['id']],rubric,MODEL,output/'judge'/q['id'],
                before_attempt=guard,system_prompt=profile['request_template']['messages'][0]['content'])
            if result['status']!='accepted' or not result['usage_complete']:
                raise ValueError('judge rejected question: '+q['id'])
            rows.append({'dataset':'custom20','question_id':q['id'],'evidence_kind':'ai_judge_not_human',**result['scores']})
        finally:
            record_usage()
    return rows


def validate_subject_result(value, answers, *, subject_id=None):
    """Validate OC dictionary and native CMMLU list details against frozen golds."""
    details = value.get('details')
    if not isinstance(details, (dict, list)) or not answers:
        raise ValueError('missing benchmark details or golds')
    native_list = isinstance(details, list)
    if native_list:
        if not isinstance(subject_id, str) or not subject_id.startswith('cmmlu-'):
            raise ValueError('native CMMLU details require a bound subject identifier')
        if len(details) != len(answers):
            raise ValueError('missing or extra benchmark rows')
    else:
        if set(details) - {'type'} != {str(i) for i in range(len(answers))}:
            raise ValueError('missing, extra or noncontiguous benchmark rows')
        if 'type' in details and details['type'] != 'GEN':
            raise ValueError('expected generation details')
    correct = 0
    for i, answer in enumerate(answers):
        row = details[i] if native_list else details[str(i)]
        if not isinstance(row, dict):
            raise ValueError('invalid detail row')
        if native_list:
            if row.get('example_abbr') != f'{subject_id}_test_{i}':
                raise ValueError('benchmark subject or row identity differs')
            if any(not isinstance(row.get(k), list) or len(row[k]) != 1
                   for k in ('pred', 'refr', 'is_correct')):
                raise ValueError('expected singleton CMMLU prediction, reference and correctness')
            pred, ref, flag = row['pred'][0], row['refr'][0], row['is_correct'][0]
            if type(flag) is not bool:
                raise ValueError('invalid CMMLU correctness flag')
        elif 'predictions' in row and 'references' in row:
            pred, ref, flag = row['predictions'], row['references'], row.get('correct')
        elif 'pred' in row and 'refr' in row:
            pred, ref, flag = row['pred'], row['refr'], row.get('is_correct')
        else:
            raise ValueError('unknown benchmark detail schema')
        if not isinstance(pred, str) or ref != answer or answer not in ('A', 'B', 'C', 'D'):
            raise ValueError('benchmark reference differs from frozen gold')
        hit = pred == ref
        if flag is not None and (type(flag) is not bool or flag != hit):
            raise ValueError('incorrect correctness flag')
        correct += hit
    accuracy = value.get('accuracy')
    if type(accuracy) not in (int, float) or not math.isfinite(accuracy) or abs(accuracy - correct / len(answers) * 100) > 1e-6:
        raise ValueError('accuracy differs from independently counted answers')
    return len(answers), correct


def require_evaluation_lock(args):
    preflight = json.loads((ROOT / 'configs/week8_evaluation_preflight.json').read_text())
    lock = preflight.get('runtime_lock')
    if not isinstance(lock, dict) or lock.get('verified') is not True:
        raise ValueError('evaluation runtime lock is incomplete; no inference or judge calls allowed')
    if lock.get('model_path') != str(Path(args.model).resolve()):
        raise ValueError('model differs from evaluation runtime lock')
    if not isinstance(lock.get('dependencies'), list) or not lock['dependencies']:
        raise ValueError('evaluation runtime lock dependencies are incomplete')
    artifacts = [(key, lock[key]) for key in ('config', 'records')]
    artifacts += [(item['path'], item) for item in lock['dependencies']]
    for key, artifact in artifacts:
        path = ROOT / artifact['path']
        if sha256(path) != artifact['sha256']:
            raise ValueError(f'evaluation runtime lock artifact changed: {key}')
    if lock.get('judge') != {'model': args.judge_model, 'url': args.judge_url}:
        raise ValueError('judge differs from evaluation runtime lock')
    profile=load_judge_profile(lock)
    if lock['judge']!={'model':profile['request_template']['model'],'url':profile['base_url']}:
        raise ValueError('locked judge differs from calibrated profile')
    return lock


def opencompass_command(config, output):
    return ['opencompass', str(config), '--work-dir', str(output),
            '--max-num-workers', '1', '--dump-eval-details']


def fresh(args):
    if not args.model or not args.judge_model or not args.judge_url:
        raise ValueError('fresh mode needs --model, --judge-model and --judge-url')
    lock = require_evaluation_lock(args)
    profile=load_judge_profile(lock)
    starting_balance=judge_balance()
    if starting_balance<1:raise ValueError('judge balance below reserve')
    write_json(args.output_dir/'judge_starting_balance.json',{'cny':str(starting_balance)})
    oc = args.output_dir/'opencompass'
    cmd = opencompass_command(ROOT / lock['config']['path'], oc)
    write_json(args.output_dir/'opencompass_command.json', cmd)
    with (args.output_dir/'opencompass.log').open('w') as log:
        env = os.environ.copy()
        env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, cwd=ROOT, env=env)
    rows = validate_benchmarks(oc, lock, args.model)
    complete_custom20(args, profile, rows, starting_balance=starting_balance)


def validate_benchmarks(oc, lock, model_path):
    records = json.loads((ROOT / lock['records']['path']).read_text())
    rows = []
    for dataset, count in [('ceval', 52), ('cmmlu', 67)]:
        golds = {}
        split = 'val' if dataset == 'ceval' else 'test'
        for record in records:
            if record['benchmark'] == dataset and record['split'] == split:
                golds.setdefault(record['subject'], []).append(record)
        for subject in golds:
            golds[subject] = [r['answer'] for r in sorted(golds[subject], key=lambda r: r['row_index'])]
        paths = sorted(oc.glob(f'*/results/*/{dataset}-*.json'))
        if len(golds) != count or len(paths) != count or {p.stem for p in paths} != {f'{dataset}-{s}' for s in golds}:
            raise ValueError(f'{dataset}: incomplete or duplicate subject coverage')
        total = correct = 0
        for path in paths:
            n, hits = validate_subject_result(json.loads(path.read_text()), golds[path.stem.removeprefix(dataset + '-')], subject_id=path.stem)
            total += n
            correct += hits
        rows.append({'model': model_path, 'dataset': dataset, 'score': correct / total * 100, 'subjects': count, 'questions': total, 'correct': correct, 'evidence_kind': 'fresh_opencompass'})
    return rows


def complete_custom20(args, profile, rows, *, starting_balance=None, provenance=None):
    if starting_balance is None:
        starting_balance = judge_balance()
        if starting_balance < 1:
            raise ValueError('judge balance below reserve')
        write_json(args.output_dir/'judge_starting_balance.json', {'cny': str(starting_balance)})
    # Fresh custom generation uses the actual unquantized merged model.
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    if not torch.cuda.is_available():
        raise RuntimeError('fresh formal inference requires CUDA')
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, torch_dtype=torch.bfloat16, device_map='auto').eval()
    spec_path = ROOT/'deliverables/week3/day14/source/data/evaluation_questions.json'
    spec = json.loads(spec_path.read_text())
    rubric = json.loads((ROOT/'deliverables/week3/day14/source/data/evaluation_rubric.json').read_text())
    if len(spec['questions']) != 20 or len({q['id'] for q in spec['questions']}) != 20:
        raise ValueError('custom set must contain 20 unique questions')
    torch.manual_seed(42)
    with (args.output_dir/'answers.jsonl').open('w') as stream:
        for q in spec['questions']:
            messages = [{'role': 'system', 'content': spec['system_message']}] + q['messages']
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt').to(model.device)
            with torch.inference_mode():
                out = model.generate(ids, max_new_tokens=512, do_sample=False)
            text = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            stream.write(json.dumps({'id': q['id'], 'answer': text}, ensure_ascii=False)+'\n'); stream.flush()
    del model; torch.cuda.empty_cache()
    answers = {r['id']: r['answer'] for r in (json.loads(line) for line in (args.output_dir/'answers.jsonl').read_text().splitlines())}
    rows.extend({'model':args.model,**row} for row in score_custom20(spec,answers,rubric,profile,args.output_dir,starting_balance))
    write_csv(args.output_dir/'summary.csv', rows)
    write_json(args.output_dir/'status.json', {'status': 'completed', 'mode': 'resumed_custom20' if provenance else 'fresh_model', 'fresh_model_inference': True, 'benchmark_inference_reused': provenance is not None, 'benchmark_provenance': provenance, 'judge_model': args.judge_model, 'custom_questions': 20, 'questions_sha256': sha256(spec_path)})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['replay', 'fresh'], default='fresh')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--model'); p.add_argument('--judge-model'); p.add_argument('--judge-url')
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        replay(args.output_dir) if args.mode == 'replay' else fresh(args)
    except Exception as exc:
        write_json(args.output_dir/'status.json', {'status': 'failed', 'mode': args.mode, 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
