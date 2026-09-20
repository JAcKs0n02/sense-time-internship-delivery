#!/usr/bin/env python3
"""OpenCompass + calibrated Gemini, with separable GPU/API stages.

Default: full CEval/CMMLU and custom20. --limit-per-subject is explicitly a
sample, never a full benchmark result. --dry-run loads no model and calls no API.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, sha256, write_json
import step3_eval as historical
import week8_clean_eval as shared
import week8_gemini_score_only as scoring
from week8_gemini_judge import MODEL
from week8_pipeline_score import verify_completed, load_release


def build_plan(model, output, limit=0):
    from mmengine.config import Config
    from build_week8_formal_eval_config import build_config
    if limit < 0:
        raise ValueError('limit must be nonnegative')
    config, binding = build_config(ROOT, model)
    records = json.loads((ROOT/binding['records']['path']).read_text())
    if limit:
        for dataset in config['datasets']:
            dataset['reader_cfg']['test_range'] = f'[:{limit}]'
        records = [r for r in records if r['row_index'] < limit or
                   r['split'] != ('val' if r['benchmark'] == 'ceval' else 'test')]
    load_release('original_base')  # Verify the existing calibration offline.
    output.mkdir(parents=True, exist_ok=False)
    Config(config).dump(str(output/'opencompass_config.py'))
    write_json(output/'benchmark_records.json', records)
    scored = [r for r in records if r['split'] == ('val' if r['benchmark'] == 'ceval' else 'test')]
    dependencies = {a['path']: a['sha256'] for a in binding['dependencies']}
    dependencies['scripts/pipeline/step3_eval.py'] = sha256(Path(__file__))
    plan = {'model_path': str(model), 'judge_model': MODEL,
            'coverage': 'sample' if limit else 'full', 'limit_per_subject': limit,
            'benchmark_questions': len(scored), 'custom_questions': 20,
            'dependencies': dependencies,
            'config_sha256': sha256(output/'opencompass_config.py'),
            'records_sha256': sha256(output/'benchmark_records.json')}
    write_json(output/'plan.json', plan)
    return plan


def benchmark_rows(output, plan):
    rows = historical.validate_benchmarks(output/'generation/opencompass',
        {'records': {'path': str(output/'benchmark_records.json')}}, plan['model_path'])
    if sum(r['questions'] for r in rows) != plan['benchmark_questions']:
        raise ValueError('benchmark count differs from planned coverage')
    for row in rows:
        row['coverage'] = plan['coverage']
        row['evidence_kind'] = 'fresh_opencompass_sample' if plan['coverage'] == 'sample' else 'fresh_opencompass'
    return rows


def generate_answers(model_path, output):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA BF16 required')
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
        trust_remote_code=False, torch_dtype=torch.bfloat16, device_map='auto').eval()
    spec = json.loads((ROOT/shared.QUESTIONS).read_text())
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    torch.manual_seed(42)
    with (output/'answers.jsonl').open('x') as stream:
        for q in spec['questions']:
            messages = [{'role': 'system', 'content': spec['system_message']}] + q['messages']
            ids = tokenizer.apply_chat_template(messages, tokenize=True,
                add_generation_prompt=True, return_tensors='pt').to(model.device)
            if ids.shape[1]+512 > 2048:
                raise ValueError('custom question exceeds reviewed context')
            with torch.inference_mode():
                tokens = model.generate(ids, max_new_tokens=512, do_sample=False)[0, ids.shape[1]:]
            answer = tokenizer.decode(tokens, skip_special_tokens=True)
            reason = 'eos' if len(tokens) and int(tokens[-1]) in eos else 'length'
            if reason != 'eos' or not answer.strip():
                raise ValueError('custom answer truncated or empty')
            stream.write(json.dumps({'id': q['id'], 'answer': answer,
                'finish_reason': reason, 'generated_tokens': len(tokens)}, ensure_ascii=False)+'\n')
            stream.flush()


def generate(output, plan):
    model = Path(plan['model_path'])
    if not (model/'config.json').is_file() or not list(model.glob('*.safetensors')):
        raise ValueError('model must be a local complete merged model directory')
    generation = output/'generation'
    generation.mkdir()
    with shared.gpu_guard():
        shared.run_child(historical.opencompass_command(output/'opencompass_config.py',
            generation/'opencompass'), generation/'opencompass.log', 14400)
        rows = benchmark_rows(output, plan)
        write_json(generation/'benchmarks.json', rows)
        shared.run_child([sys.executable, str(Path(__file__)), '--phase', 'worker',
            '--output-dir', str(output)], generation/'custom.log', 3600)
    answers = [json.loads(line) for line in (generation/'answers.jsonl').read_text().splitlines()]
    shared.validate_answers(answers)
    files = {str(p.relative_to(generation)): sha256(p) for p in generation.rglob('*') if p.is_file()}
    write_json(generation/'generation_receipt.json', {'files': files,
        'plan_sha256': sha256(output/'plan.json'), 'judge_calls': 0})
    write_json(output/'status.json', {'status': 'generated_pending_scoring',
        'fresh_model_inference': True, 'new_api_calls': 0, 'coverage': plan['coverage']})


def score(output, plan):
    generation = output/'generation'
    receipt = json.loads((generation/'generation_receipt.json').read_text())
    if receipt['plan_sha256'] != sha256(output/'plan.json'):
        raise ValueError('generation plan changed')
    for name, expected in plan['dependencies'].items():
        if sha256(ROOT/name) != expected:
            raise ValueError('evaluation dependency changed: '+name)
    for name, key in [('opencompass_config.py', 'config_sha256'), ('benchmark_records.json', 'records_sha256')]:
        if sha256(output/name) != plan[key]:
            raise ValueError('evaluation configuration changed')
    shared.verify_source_files(generation, receipt)
    rows = benchmark_rows(output, plan)
    if rows != json.loads((generation/'benchmarks.json').read_text()):
        raise ValueError('benchmark summary differs from raw predictions')
    answers = [json.loads(line) for line in (generation/'answers.jsonl').read_text().splitlines()]
    score_plan = shared.bind_score_plan(answers, sha256(generation/'answers.jsonl'),
                                       sha256(generation/'generation_receipt.json'))
    score_plan.update(scope='pipeline_custom20', model_name=plan['model_path'])
    scored = output/'scoring'
    before = len(list(scored.glob('judge/*/started.json')))
    scoring.run(score_plan, scored, ROOT/'reports/week8/score_only_claims')
    checked = verify_completed(score_plan, scored)
    scores = json.loads((scored/'summary.json').read_text())['scores']
    rows.append({'model': plan['model_path'], 'dataset': 'custom20', 'questions': 20,
        'score': checked['weighted_mean'], 'coverage': 'full_custom20',
        'evidence_kind': 'fresh_ai_judge', 'judge_model': MODEL,
        **{k: sum(s[k] for s in scores)/20 for k in historical.WEIGHTS}})
    historical.write_csv(output/'summary.csv', rows)
    write_json(output/'status.json', {'status': 'completed', 'fresh_model_inference': True,
        'coverage': plan['coverage'], 'new_api_calls': checked['historical_api_attempts']-before,
        'custom_questions': 20, 'judge_model': MODEL})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['fresh', 'replay'], default='fresh')
    parser.add_argument('--phase', choices=['all', 'generate', 'score', 'worker'], default='all')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--limit-per-subject', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.mode == 'replay':
        if args.phase != 'all' or args.dry_run or args.limit_per_subject:
            parser.error('replay cannot combine with fresh evaluation options')
        output.mkdir(parents=True, exist_ok=False)
        historical.replay(output)
        return
    if args.phase in ('score', 'worker'):
        if args.dry_run or args.model or args.limit_per_subject:
            parser.error('score/worker uses the saved generation plan; no overrides allowed')
        plan = json.loads((output/'plan.json').read_text())
        if args.phase == 'worker':
            generate_answers(plan['model_path'], output/'generation')
            return
    else:
        if args.model is None:
            parser.error('fresh generation requires --model')
        plan = build_plan(args.model.resolve(), output, args.limit_per_subject)
        write_json(output/'status.json', {'status': 'planned_only', 'fresh_model_inference': False,
                                         'new_api_calls': 0, 'coverage': plan['coverage']})
        if args.dry_run:
            return
    try:
        if args.phase in ('all', 'generate'):
            generate(output, plan)
        if args.phase in ('all', 'score'):
            if args.phase == 'all':
                shared.run_child([sys.executable, str(Path(__file__)), '--phase', 'score',
                    '--output-dir', str(output)], output/'scoring.log', 7200)
            else:
                score(output, plan)
    except BaseException as exc:
        write_json(output/'status.json', {'status': 'failed', 'error_type': type(exc).__name__,
                   'generation_completed': (output/'generation/generation_receipt.json').is_file(),
                   'paid_attempts': len(list((output/'scoring').glob('judge/*/started.json')))})
        raise


if __name__ == '__main__':
    main()
