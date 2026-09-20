#!/usr/bin/env python3
"""Resume only custom20 after independently verifying the bound objective evidence."""
import argparse
import json
from pathlib import Path, PurePosixPath
import step3_eval as ev
from common import ROOT, sha256, write_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_source(source, lock):
    source = Path(source).resolve()
    binding = lock.get('resume_source', {})
    require(binding.get('model_name') == 'original_base', 'only original_base may reuse this evidence')
    run_name = binding.get('run_name')
    require(isinstance(run_name, str) and run_name not in ('', '.', '..') and '/' not in run_name,
            'invalid source run name')
    manifest_path = source/'manifest.json'
    require(not manifest_path.is_symlink() and sha256(manifest_path) == binding.get('manifest_sha256'),
            'source manifest changed')
    manifest = json.loads(manifest_path.read_text())
    require(isinstance(manifest, list) and bool(manifest), 'invalid source manifest')
    expected = set()
    for item in manifest:
        relative = PurePosixPath(item['path'])
        require(not relative.is_absolute() and '..' not in relative.parts and str(relative) == item['path'],
                'unsafe manifest path')
        require(item['path'] not in expected and item['path'] != 'manifest.json', 'duplicate manifest path')
        expected.add(item['path'])
    entries = list(source.rglob('*'))
    require(not any(p.is_symlink() for p in entries), 'symlink in source evidence')
    require({str(p.relative_to(source)) for p in entries if p.is_file()} == expected | {'manifest.json'},
            'source file set changed')
    for item in manifest:
        p = source/item['path']
        require(p.stat().st_size == item['bytes'] and sha256(p) == item['sha256'], 'source artifact changed: '+item['path'])
    previous = binding.get('previous_lock', {})
    old_path = ev.ROOT/previous.get('path', '')
    require(old_path.is_file() and sha256(old_path) == previous.get('sha256'), 'previous lock changed')
    old = json.loads(old_path.read_text())
    require(old.get('verified') is True, 'previous lock was not released')
    for key in ('model_path', 'model_identity', 'config', 'records', 'judge'):
        require(lock.get(key) == old.get(key), 'source/current identity differs: '+key)
    current_deps = {d['path']: d['sha256'] for d in lock['dependencies']}
    require(len(current_deps) == len(lock['dependencies']), 'duplicate current dependencies')
    for dep in old['dependencies']:
        if dep['path'] != 'scripts/step3_eval.py':
            require(current_deps.get(dep['path']) == dep['sha256'], 'source dependency differs: '+dep['path'])
    for name in ('scripts/step3_eval.py', 'scripts/week8_resume_eval.py'):
        require(current_deps.get(name) == sha256(ev.ROOT/name), 'resume code is not bound: '+name)
    run = source/run_name
    launch = json.loads((run/'launch.json').read_text())
    require(launch['locks']['original_base'] == previous['sha256'], 'source launch used another model lock')
    require(launch['host'] == 'autodl-container-be044ebe99-be706b14', 'unexpected source host')
    status = json.loads((run/'status.json').read_text())
    require(status['status'] == 'FAILED_OR_TIMEOUT' and status['completed_models'] == [], 'source session not eligible')
    model_dir = run/'original_base'
    result = json.loads((model_dir/'status.json').read_text())
    require(result.get('status') == 'failed' and result.get('error') == 'missing benchmark details or golds',
            'source failure is not the reviewed schema failure')
    command = json.loads((run/'original_base_command.json').read_text())
    require(command.count('--model') == 1 and command[command.index('--model')+1] == lock['model_path'],
            'source command model differs')
    require(not (model_dir/'answers.jsonl').exists() and not (model_dir/'judge').exists()
            and not (model_dir/'judge_usage.json').exists(), 'source already attempted custom20')
    require(not (run/'final_sft').exists() and not (run/'final_dpo').exists(), 'source has later model outputs')
    rows = ev.validate_benchmarks(model_dir/'opencompass', lock, lock['model_path'])
    for row in rows:
        row['evidence_kind'] = 'reused_verified_opencompass'
        row['source_run'] = run_name
    provenance = {'source_run': run_name, 'source_model': 'original_base',
                  'source_manifest_sha256': binding['manifest_sha256'],
                  'source_lock_sha256': previous['sha256'],
                  'objective_inference_repeated': False}
    return rows, provenance


def resume(args):
    lock = ev.require_evaluation_lock(args)
    rows, provenance = validate_source(args.source, lock)
    profile = ev.load_judge_profile(lock)
    require(args.output_dir.resolve() != args.source.resolve()
            and args.source.resolve() not in args.output_dir.resolve().parents,
            'output must be outside preserved source')
    # The source is immutable; persist a separate exclusive claim before any paid work.
    claims = ROOT/'logs/resume-claims'
    claims.mkdir(parents=True, exist_ok=True)
    claim = claims/(provenance['source_manifest_sha256']+'.started')
    with claim.open('x') as stream:
        json.dump({'output_dir': str(args.output_dir.resolve()), **provenance}, stream)
    write_json(args.output_dir/'resume_runtime_lock.json', lock)
    provenance['resume_runtime_lock_sha256'] = sha256(args.output_dir/'resume_runtime_lock.json')
    write_json(args.output_dir/'benchmark_provenance.json', provenance)
    ev.complete_custom20(args, profile, rows, provenance=provenance)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--judge-model', required=True)
    p.add_argument('--judge-url', required=True)
    args = p.parse_args()
    require(args.source.resolve() not in args.output_dir.resolve().parents
            and args.source.resolve() != args.output_dir.resolve(), 'output must be outside preserved source')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        resume(args)
    except Exception as exc:
        write_json(args.output_dir/'status.json', {'status':'failed', 'mode':'resumed_custom20', 'error':str(exc)})
        raise


if __name__ == '__main__':
    main()
