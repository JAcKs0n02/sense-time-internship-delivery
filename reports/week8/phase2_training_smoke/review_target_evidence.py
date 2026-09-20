"""Recheck downloaded smoke evidence; never launches training or accesses credentials."""
import hashlib
import json
import math
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from week8_training_smoke import admit, build_configs, validate_state


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


here = Path(__file__).resolve().parent
evidence = here / 'target-20260916'
run = evidence / 'run-01'
lock_sha = '7c9bc62c7870b5a652eb098a6451319728271902ab280d46eef078769d2be882'
lock = admit(ROOT, here / 'smoke_lock.json', lock_sha)
assert digest(evidence / 'smoke_evidence.tar.gz') == '6fdeee44fde063e0e62f71215c137fd5971b3954bdecc046ac9d464ffb99a941'
manifest = read(evidence / 'evidence_manifest.json')
assert len(manifest['files']) == 30
names = set()
for item in manifest['files']:
    name = item['path']
    path = (evidence / name).resolve()
    assert not Path(name).is_absolute() and path.is_relative_to(evidence.resolve())
    assert name not in names
    names.add(name)
    assert path.stat().st_size == item['size'] and digest(path) == item['sha256']
assert manifest['gpu_after'] == ''
preflight = read(evidence / 'target_preflight.json')
assert preflight['hostname'] == 'autodl-container-be044ebe99-be706b14'
assert preflight['cuda_bf16'] and preflight['base_files_verified'] == 15
assert preflight['frozen_inputs_verified'] == 31
assert preflight['memory_limit'] == str(60 * 1024**3)
assert preflight['disk_free'] > 50 * 1024**3
assert preflight['versions'] == {
    'torch': '2.5.1+cu121', 'transformers': '4.50.0', 'llamafactory': '0.9.3',
    'peft': '0.15.1', 'trl': '0.9.6', 'accelerate': '1.2.1',
    'datasets': '3.2.0', 'bitsandbytes': '0.43.3',
}
status = read(run / 'status.json')
assert status['status'] == 'SMOKE_PASS_REVIEW_REQUIRED'
assert status['full_training_allowed'] is False and status['lock_sha256'] == lock_sha
remote = Path('/root/autodl-tmp/week8-training-smoke-runtime-20260916')
expected_configs = build_configs(ROOT, remote / 'logs/week8-protocol-data-20260914/run-a', remote / 'run-01', lock['base_model'])
results = {}
for stage, expected in zip(('sft', 'dpo'), expected_configs):
    attempt = run / stage / 'attempt-01'
    expected['output_dir'] = str(remote / 'run-01' / stage / 'attempt-01/adapter')
    actual = yaml.safe_load((attempt / 'train.yaml').read_text())
    assert actual == expected, stage
    assert read(attempt / 'status.json') == {'attempt': 1, 'returncode': 0}
    trainer = read(attempt / 'adapter/trainer_state.json')
    checkpoint = read(attempt / 'adapter/checkpoint-3/trainer_state.json')
    # Final save appends aggregate train metrics after checkpoint-3 was written.
    assert {k: v for k, v in trainer.items() if k != 'log_history'} == {k: v for k, v in checkpoint.items() if k != 'log_history'}
    assert trainer['log_history'][:-1] == checkpoint['log_history']
    assert trainer['log_history'][-1]['step'] == 3 and 'train_loss' in trainer['log_history'][-1]
    rows = validate_state(trainer)
    assert rows == validate_state(checkpoint)
    for row in rows:
        assert all(math.isfinite(v) for v in row.values() if type(v) in (int, float))
    result = read(run / f'{stage}_verification.json')
    assert result == status['results'][stage] and result['metrics'] == rows
    assert result['tensor_count'] == result['optimizer']['parameter_states'] == 392
    assert result['nonzero_lora_b_tensors'] == 196 and result['optimizer']['step'] == 3
    cfg = read(attempt / 'adapter/adapter_config.json')
    assert cfg == read(attempt / 'adapter/checkpoint-3/adapter_config.json')
    assert cfg['r'] == 8 and cfg['lora_alpha'] == 16 and cfg['lora_dropout'] == 0.05
    assert cfg['base_model_name_or_path'] == actual['model_name_or_path']
    log = (attempt / 'train.log').read_text()
    assert 'Traceback (most recent call last)' not in log and 'CUDA out of memory' not in log
    assert 'Total optimization steps = 3' in log
    assert ('Num examples = 1,422' if stage == 'sft' else 'Num examples = 783') in log
    metrics = read(attempt / 'adapter/train_results.json')
    results[stage] = {'global_step': 3, 'loss': [r['loss'] for r in rows],
                      'runtime_seconds': metrics['train_runtime'], 'configuration_matches_frozen_plan': True}
merge = read(run / 'sft_merge_verification.json')
assert merge == status['results']['sft_merge'] and merge['tensors'] == 339 and len(merge['files']) == 4
report = {
    'status': 'GPU_SMOKE_PASS_LOCAL_REVIEWED', 'files_hash_verified': 30,
    'frozen_inputs_rechecked': 31, 'base_files_verified_on_target': 15,
    'stages': results, 'merged_tensors_verified_on_target': 339,
    'merged_shards_verified_on_target': 4, 'final_gpu_processes': [],
    'archive_sha256': digest(evidence / 'smoke_evidence.tar.gz'),
    'manifest_sha256': digest(evidence / 'evidence_manifest.json'),
    'target_status_sha256': digest(run / 'status.json'),
    'review_script_sha256': digest(Path(__file__)),
    'full_training_allowed': False, 'formal_evaluation_run': False, 'judge_api_calls': 0,
    'limits': ['Only three optimizer steps per stage; no quality claim.',
               'Adapter/optimizer/merged tensors verified on target; raw binary weights remain on target.',
               'SDPA sliding-window warning retained; logged model use_sliding_window is false.',
               'Inherited invalid OMP_NUM_THREADS corrected to 14 before training; MKL_NUM_THREADS=14.',
               'DPO retains 29 warmup steps; SFT final logged scheduler learning rate is zero.'],
}
(here / 'target_review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(report, ensure_ascii=False, indent=2))
