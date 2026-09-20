"""Produce and audit formal-training candidates locally. No training or network calls."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import inspect
import json
import math
import sys

import torch
from torch.utils.data import DataLoader
import transformers
from transformers import Trainer
import yaml

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
from week8_training_smoke import admit


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


smoke = ROOT / 'reports/week8/phase2_training_smoke'
lock = admit(ROOT, smoke / 'smoke_lock.json', '7c9bc62c7870b5a652eb098a6451319728271902ab280d46eef078769d2be882')
review = read(smoke / 'target_review.json')
assert review['status'] == 'GPU_SMOKE_PASS_LOCAL_REVIEWED'
index = read(ROOT / 'configs/week8_evaluation_preflight.json')['training_smoke']
assert sha(smoke / 'target_review.json') == index['target_review_sha256']
source = read(ROOT / 'reports/week8/requirements_source.json')
for path, digest in source['sources'].items():
    assert sha(ROOT / path) == digest, 'Requirements changed: reread before generating candidates'

remote = Path('/root/autodl-tmp/week8-full-training-runtime-20260916')
run = remote / 'logs/full-run-01'
data = remote / 'logs/week8-protocol-data-20260914/run-a'
pref = remote / 'deliverables/week4/day20/remediation/source/data'
sft_source = ROOT / 'deliverables/week3/day11/configs/experiments/epoch-e5.yaml'
dpo_source = ROOT / 'deliverables/week4/day20/remediation/configs/reward_corrective_40step.yaml'
sft = yaml.safe_load(sft_source.read_text())
dpo = yaml.safe_load(dpo_source.read_text())
originals = {'sft': dict(sft), 'dpo': dict(dpo)}
sft.update(model_name_or_path=lock['base_model'], dataset='week8_train', eval_dataset='week8_validation',
           dataset_dir=str(data), output_dir=str(run/'sft/attempt-01/adapter'), max_steps=-1,
           eval_strategy='epoch', per_device_eval_batch_size=1, prediction_loss_only=True,
           run_name='week8-full-sft-epoch-e5')
sft.pop('max_samples', None)
dpo.update(model_name_or_path=str(run/'models/final_sft'), dataset_dir=str(pref),
           output_dir=str(run/'dpo/attempt-01/adapter'), run_name='week8-full-dpo-corrective-40')
config_dir = ROOT / 'configs/week8_full_training_candidate'
config_dir.mkdir(exist_ok=True)
overrides = {}
for stage, cfg in [('sft', sft), ('dpo', dpo)]:
    cfg.update(preprocessing_num_workers=1, dataloader_num_workers=0, report_to='none',
               plot_loss=False, overwrite_output_dir=False, resume_from_checkpoint=None,
               save_only_model=False, save_strategy='steps', load_best_model_at_end=False)
    assert cfg['per_device_train_batch_size'] == 1 and cfg['cutoff_len'] == 2048
    assert cfg['quantization_bit'] == 4 and cfg['bf16'] and cfg['lora_rank'] == 8 and cfg['lora_alpha'] == 16
    assert not any(k in cfg for k in ('adapter_name_or_path', 'ref_model'))
    for key in ('learning_rate','gradient_accumulation_steps','lr_scheduler_type','seed','lora_dropout'):
        assert cfg[key] == originals[stage][key]
    overrides[stage] = {k: {'source': originals[stage].get(k), 'candidate': cfg.get(k)}
                        for k in sorted(originals[stage].keys() | cfg.keys())
                        if originals[stage].get(k) != cfg.get(k)}
    (config_dir/f'{stage}.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
assert sft['num_train_epochs'] == 5 and sft['save_steps'] == 500 and sft['save_total_limit'] == 2
assert dpo['max_steps'] == 40 and dpo['warmup_steps'] == 29 and dpo['eval_steps'] == 20
assert dpo['save_steps'] == 40 and dpo['save_total_limit'] == 1
for stage, base, adapter in [('sft', lock['base_model'], run/'sft/attempt-01/adapter'),
                             ('dpo', run/'models/final_sft', run/'dpo/attempt-01/adapter')]:
    cfg = dict(model_name_or_path=str(base), adapter_name_or_path=str(adapter), template='qwen',
               finetuning_type='lora', export_dir=str(run/f'models/final_{stage}'),
               export_size=5, export_device='cpu', export_legacy_format=False)
    (config_dir/f'export_{stage}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))

assert transformers.__version__ == '4.50.0'
trainer = object.__new__(Trainer)
args = SimpleNamespace(max_steps=-1, gradient_accumulation_steps=4, num_train_epochs=5.0)
values = trainer.set_initial_training_values(args, DataLoader(range(1422), batch_size=1), 4)
assert values[-1] == 1775
method = inspect.getsource(Trainer.set_initial_training_values)
(OUT/'trainer_step_calculation_source.py.txt').write_text(method)
base_bytes = sum(x['bytes'] for x in read(ROOT/'reports/week8/phase2_gpu_320/expected_base_identity.json')['files']
                 if x['path'].endswith('.safetensors'))
manifest_paths = set(x['path'] for x in lock['files']) | {
    str(p.relative_to(ROOT)) for p in config_dir.glob('*.yaml')
} | {'reports/week8/phase2_training_smoke/target_review.json',
     'reports/week8/phase2_training_smoke/target-20260916/evidence_manifest.json',
     'reports/week8/phase2_training_smoke/target-20260916/run-01/status.json',
     'reports/week8/requirements_source.json', 'reports/week8/requirements_snapshot.txt',
     'reports/week8/phase2_full_training_plan/trainer_step_calculation_source.py.txt',
     'reports/week8/phase2_full_training_plan/prepare_candidates.py'}
manifest = [{'path': p, 'sha256': sha(ROOT/p)} for p in sorted(manifest_paths)]
write(OUT/'input_manifest.json', {'purpose':'PLAN_ONLY_NOT_EXECUTION_AUTHORIZATION','files':manifest})
write(OUT/'config_overrides.json', overrides)
receipt = {
    'status':'FULL_TRAINING_CANDIDATES_VERIFIED_NOT_RELEASED', 'training_started':False,
    'execution_authorized':False, 'gpu_start_requested':False, 'full_training_allowed':False,
    'requirements_source_hashes_verified':True, 'frozen_inputs_verified':len(lock['files']),
    'candidate_configs_verified':4, 'input_manifest_sha256':sha(OUT/'input_manifest.json'),
    'sft':{'samples':1422,'validation':158,'num_train_epochs':5,'expected_max_steps':1775,
           'step_calculation':'Transformers 4.50.0 actual Trainer method, CPU DataLoader, no model',
           'save_steps':500,'save_total_limit':2,'eval_strategy':'epoch'},
    'dpo':{'samples':783,'validation':87,'max_steps':40,'warmup_steps':29,'eval_steps':20},
    'budget_proposal':{'price_ceiling_cny_hour':1.48,'wall_clock_hours':3,'compute_ceiling_cny':4.44,
                       'approved':False,'scope':'SFT+DPO+two merges+load checks; excludes formal benchmarks and API',
                       'rationale':'Planning allowance, not a runtime prediction; 3-step timings do not establish full-run time'},
    'resources':{'minimum_free_disk_gib':45,'two_merged_weight_gib':round(base_bytes*2/1024**3,3),
                  'memory_limit_gib':60,'last_observed_disk_used_percent':80.15,
                  'live_preflight_required':True,'delete_existing_models':False},
    'remaining':['implement hash-bound full-training entry and stage-aware acceptance',
                 'validate cold loading of each complete merge and capture identity',
                 'confirm formal-session spending/time ceiling before paid launch',
                 'recheck target capacity/processes/disk before launch'],
}
write(OUT/'verification.json', receipt)
print(json.dumps(receipt,ensure_ascii=False,indent=2))
