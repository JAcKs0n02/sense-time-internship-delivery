#!/usr/bin/env python3
"""Portable SFT -> merge -> DPO -> merge using the reviewed Week 8 inputs."""
import argparse
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, write_json
from train_pipeline import execute_train, export_model
from week8_clean_eval import check_data
from week8_full_training import load_configs, validate_configs, verify_formal_adapter


def prepare(data, run, base):
    # Admit reproduced outputs, not an editable status label from an early task.
    check_data(data)
    configs = validate_configs(load_configs(ROOT))
    if run == data or run.is_relative_to(data) or data.is_relative_to(run):
        raise ValueError('training output must not overlap prepared data')
    run.mkdir(parents=True, exist_ok=False)
    preference = run/'preference_data'
    preference.mkdir()
    source = ROOT/'deliverables/week4/day20/remediation/source/data'
    for name in ('dataset_info.json', 'week4_dpo_train_v2.json', 'week4_dpo_validation_v2.json'):
        shutil.copy2(source/name, preference/name)
    for stage in ('sft', 'dpo'):
        config = configs[stage]
        config.update(model_name_or_path=str(base if stage == 'sft' else run/'models/final_sft'),
                      dataset_dir=str(data if stage == 'sft' else preference),
                      output_dir=str(run/stage/'attempt-01/adapter'))
    plan = {'sft': configs['sft'], 'dpo': configs['dpo'],
            'expected_steps': {'sft': 1775, 'dpo': 40},
            'reference': 'DPO adapter-disabled reference = this run merged SFT'}
    write_json(run/'planned_configs.json', plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--base-model', type=Path, required=True)
    parser.add_argument('--cli', default='llamafactory-cli')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    data, run, base = (p.resolve() for p in (args.data_dir, args.run_dir, args.base_model))
    if not args.dry_run:
        if not (base/'config.json').is_file() or not list(base.glob('*.safetensors')):
            raise ValueError('base model must be a local complete model directory')
        if not shutil.which(args.cli):
            raise RuntimeError('llamafactory-cli unavailable')
    plan = prepare(data, run, base)
    write_json(run/'status.json', {'status': 'planned_only', 'trained': False})
    if args.dry_run:
        return
    try:
        (run/'models').mkdir()
        for stage in ('sft', 'dpo'):
            write_json(run/'status.json', {'status': 'running', 'stage': stage, 'trained': False})
            adapter = execute_train(plan[stage], run/stage, args.cli)
            source = plan[stage]['model_name_or_path']
            checked = verify_formal_adapter(adapter, plan['expected_steps'][stage], source, stage)
            write_json(run/stage/'verification.json', checked)
            export_model(source, adapter, run/'models'/('final_'+stage), args.cli)
        write_json(run/'status.json', {'status': 'completed', 'trained': True,
                   'final_sft': str(run/'models/final_sft'), 'final_dpo': str(run/'models/final_dpo')})
    except BaseException as exc:
        write_json(run/'status.json', {'status': 'failed', 'trained': False,
                                      'error_type': type(exc).__name__})
        raise


if __name__ == '__main__':
    main()
