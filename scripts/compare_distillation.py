#!/usr/bin/env python3
"""Run identical CEval and generation protocols before/after distillation."""
import argparse
import csv
import hashlib
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from common import ROOT, sha256, write_json


def run_bounded(command, log_path, timeout):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('positive finite timeout required')
    with Path(log_path).open('x') as log:
        proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            code = proc.wait(timeout=timeout)
            if code:
                raise subprocess.CalledProcessError(code, command)
        finally:
            # OpenCompass can create workers; stop the entire owned process group.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()


def build_ceval_config(root, model_path):
    from build_week8_formal_eval_config import build_config
    cfg, binding = build_config(root, model_path)
    cfg['datasets'] = [d for d in cfg['datasets'] if d['abbr'].startswith('ceval-')]
    if len(cfg['datasets']) != 52 or len({d['abbr'] for d in cfg['datasets']}) != 52:
        raise ValueError('explicit CEval configuration must contain 52 unique subjects')
    cfg['models'][0]['tokenizer_path'] = str(model_path)
    binding['pending'] = ['student tokenizer prompt-length and CUDA forward check',
                          'model provenance and file manifest verification',
                          'new bounded 320 runtime release']
    binding['protocol'] = 'distillation_ceval_val_5shot'
    return cfg, binding


def verify_model_manifest(model_path, manifest_path):
    """Verify file identity only; this does not prove tensor validity or provenance."""
    model = Path(model_path).resolve()
    manifest = json.loads(Path(manifest_path).read_text())
    bound = Path(manifest.get('model_dir', ''))
    if not bound.is_absolute() or bound.resolve() != model:
        raise ValueError('model path differs from bound manifest')
    entries = manifest.get('files', [])
    expected = {e['path'] for e in entries}
    actual = {str(p.relative_to(model)) for p in model.rglob('*') if p.is_file()}
    required = {'config.json', 'tokenizer.json', 'tokenizer_config.json', 'generation_config.json'}
    if (not required <= expected or len(expected) != len(entries) or actual != expected
            or not any(n.endswith('.safetensors') for n in expected)
            or any('adapter' in n or n.endswith('.bin') for n in expected)):
        raise ValueError('incomplete, extra or adapter-only model files')
    for entry in entries:
        relative = Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('unsafe manifest path')
        path = model / relative
        if (path.resolve() != path or not path.is_file() or path.stat().st_size != entry['bytes']
                or sha256(path) != entry['sha256']):
            raise ValueError(f'model file identity mismatch: {relative}')
    index = model / 'model.safetensors.index.json'
    weights = {n for n in expected if n.endswith('.safetensors')}
    if index.exists():
        mapping = json.loads(index.read_text()).get('weight_map', {})
        if not mapping or set(mapping.values()) != weights:
            raise ValueError('weight index shard coverage differs')
    elif weights != {'model.safetensors'}:
        raise ValueError('sharded model is missing its weight index')
    return manifest


def summarize_ceval(directory, root=ROOT):
    """Recount every prediction against frozen val golds, never reported weights."""
    from step3_eval import validate_subject_result
    base = Path(root) / 'data/evaluation/benchmarks'
    lock = json.loads((base / 'benchmark_input_lock.json').read_text())
    records = base / 'benchmark_records.json'
    if sha256(records) != lock['benchmark_records_sha256']:
        raise ValueError('frozen benchmark records changed')
    golds = {}
    for row in json.loads(records.read_text()):
        if row['benchmark'] == 'ceval' and row['split'] == 'val':
            golds.setdefault('ceval-' + row['subject'], []).append(row)
    if len(golds) != 52 or sum(map(len, golds.values())) != 1346:
        raise ValueError('expected frozen CEval 52 subjects / 1346 questions')
    paths = list(Path(directory).glob('*/results/*/ceval-*.json'))
    if (len(paths) != 52 or {p.stem for p in paths} != set(golds)
            or len({p.parent for p in paths}) != 1):
        raise ValueError('missing, duplicate or mixed-run CEval subjects')
    questions = correct = 0
    for path in sorted(paths):
        rows = sorted(golds[path.stem], key=lambda r: r['row_index'])
        if [r['row_index'] for r in rows] != list(range(len(rows))):
            raise ValueError('frozen gold indices are not contiguous')
        n, hits = validate_subject_result(json.loads(path.read_text()), [r['answer'] for r in rows])
        questions += n
        correct += hits
    return dict(subjects=52, questions=questions, correct=correct,
                accuracy=100 * correct / questions, records_sha256=sha256(records))


def validate_speed(value):
    runs = value.get('runs', [])
    if (value.get('warmups') != 2 or value.get('dtype') != 'bfloat16'
            or value.get('includes_prefill') is not True or len(runs) != 10
            or len({r['id'] for r in runs}) != 10):
        raise ValueError('speed protocol differs: need 2 warmups and 10 unique BF16 runs')
    for row in runs:
        if (type(row.get('tokens')) is not int or row['tokens'] != 128
                or type(row.get('seconds')) not in (int, float)
                or not math.isfinite(row['seconds']) or row['seconds'] <= 0):
            raise ValueError('speed requires exactly 128 new tokens and finite positive timings')
    return dict(tokens_per_second=1280 / sum(r['seconds'] for r in runs),
                median_seconds=statistics.median(r['seconds'] for r in runs))


def validate_speed_pair(before, after):
    for key in ('environment', 'prompt_source_sha256', 'prompt_token_sha256'):
        if not before.get(key) or before[key] != after.get(key):
            raise ValueError(f'before/after speed protocol differs: {key}')
    if len(before['prompt_token_sha256']) != 12:
        raise ValueError('need tokenized prompt identities for warmups and measured runs')


def measure(model_path, output):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError('comparison requires CUDA')
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='auto', local_files_only=True).eval()
    # Teacher-independent, fixed prompts; prefill included in throughput.
    source = ROOT/'deliverables/week3/day14/source/data/evaluation_questions.json'
    spec = json.loads(source.read_text())
    measurements = []
    prompt_hashes = []
    for i, q in enumerate(spec['questions'][:12]):
        ids = tokenizer.apply_chat_template(q['messages'], tokenize=True, add_generation_prompt=True, return_tensors='pt').to(model.device)
        prompt_hashes.append(hashlib.sha256(json.dumps(ids.tolist()).encode()).hexdigest())
        torch.cuda.synchronize(); start = time.perf_counter()
        with torch.inference_mode():
            result = model.generate(ids, do_sample=False, min_new_tokens=128, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize(); elapsed = time.perf_counter()-start
        if i >= 2:
            measurements.append({'id': q['id'], 'tokens': result.shape[1]-ids.shape[1], 'seconds': elapsed})
    value = {'model': model_path, 'warmups': 2, 'runs': measurements, 'dtype': 'bfloat16', 'includes_prefill': True}
    device = torch.cuda.get_device_properties(model.device)
    value['environment'] = dict(gpu_uuid=str(device.uuid), gpu_name=device.name,
                                torch=torch.__version__, transformers=transformers.__version__, cuda=torch.version.cuda)
    value['prompt_source_sha256'] = sha256(source)
    value['prompt_token_sha256'] = prompt_hashes
    value.update(validate_speed(value))
    write_json(output, value)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before'); p.add_argument('--after')
    p.add_argument('--before-manifest', type=Path)
    p.add_argument('--after-manifest', type=Path)
    p.add_argument('--timeout-seconds', type=int, default=7200,
                   help='Total comparison deadline, including both models and speed tests')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--measure-model', help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.measure_model:
        measure(args.measure_model, args.output_dir/'speed.json'); return
    if not args.before or not args.after:
        p.error('--before and --after are required')
    if not args.before_manifest or not args.after_manifest:
        p.error('--before-manifest and --after-manifest are required')
    if args.timeout_seconds <= 0:
        p.error('--timeout-seconds must be positive')
    args.before = str(Path(args.before).resolve())
    args.after = str(Path(args.after).resolve())
    if args.before == args.after:
        p.error('before and after must be distinct model directories')
    args.output_dir = args.output_dir.resolve()
    for model in (args.before, args.after):
        if args.output_dir.is_relative_to(model):
            p.error('output directory must be outside model directories')
    deadline = time.monotonic() + args.timeout_seconds
    identities = {label: verify_model_manifest(model, manifest) for label, model, manifest in
                  [('before', args.before, args.before_manifest), ('after', args.after, args.after_manifest)]}
    tokenizer_files = lambda entries: {e['path']: e['sha256'] for e in entries
                                      if 'token' in e['path'] or e['path'] in ('merges.txt', 'vocab.json', 'chat_template.jinja')}
    if tokenizer_files(identities['before']['files']) != tokenizer_files(identities['after']['files']):
        raise ValueError('before/after tokenizer assets differ')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    speeds = {}
    try:
        for label, model in [('before', args.before), ('after', args.after)]:
            folder = args.output_dir/label; folder.mkdir()
            from mmengine.config import Config
            from step3_eval import opencompass_command
            cfg, binding = build_ceval_config(ROOT, model)
            config_path = folder/'ceval_explicit.py'
            Config(cfg).dump(str(config_path))
            binding['config_sha256'] = sha256(config_path)
            binding['model_manifest'] = identities[label]
            write_json(folder/'input_binding.json', binding)
            cmd = opencompass_command(config_path, folder/'opencompass')
            write_json(folder/'command.json', cmd)
            run_bounded(cmd, folder/'opencompass.log', deadline-time.monotonic())
            result = summarize_ceval(folder/'opencompass')
            write_json(folder/'ceval_verified.json', result)
            run_bounded([sys.executable, __file__, '--measure-model', model, '--output-dir', str(folder)],
                        folder/'speed.log', min(600, deadline-time.monotonic()))
            speed = json.loads((folder/'speed.json').read_text())
            speed.update(validate_speed(speed))
            speeds[label] = speed
            rows.append({'stage': label, 'model': model, 'ceval': result['accuracy'], 'ceval_questions': result['questions'], 'ceval_correct': result['correct'], 'tokens_per_second': speed['tokens_per_second'], 'median_seconds': speed['median_seconds']})
            verify_model_manifest(model, args.before_manifest if label == 'before' else args.after_manifest)
        validate_speed_pair(speeds['before'], speeds['after'])
        with (args.output_dir/'comparison.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        write_json(args.output_dir/'status.json', {'status': 'completed', 'same_protocol': True})
    except Exception as exc:
        write_json(args.output_dir/'status.json', {'status': 'failed', 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
