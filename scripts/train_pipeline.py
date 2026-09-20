#!/usr/bin/env python3
"""Sequential LLaMA-Factory SFT/DPO and export, with bounded OOM retry."""
import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import subprocess
from common import ROOT, write_json


def reduce_batch(config):
    config = copy.deepcopy(config)
    batch = int(config.get('per_device_train_batch_size', 1))
    accumulation = int(config.get('gradient_accumulation_steps', 1))
    if batch <= 1:
        raise ValueError('OOM at batch size 1: stop; no silent change of sequence length or epochs')
    smaller = max(d for d in range(1, batch) if batch % d == 0)
    config['per_device_train_batch_size'] = smaller
    config['gradient_accumulation_steps'] = accumulation * (batch // smaller)
    return config


def execute_train(config, target, cli, max_attempts=3):
    import yaml
    current = copy.deepcopy(config)
    for attempt in range(1, max_attempts+1):
        folder = target/f'attempt-{attempt:02d}'
        folder.mkdir(parents=True, exist_ok=False)
        current['output_dir'] = str(folder/'adapter')
        path = folder/'train.yaml'
        path.write_text(yaml.safe_dump(current, allow_unicode=True, sort_keys=False))
        with (folder/'train.log').open('w') as log:
            result = subprocess.run([cli, 'train', str(path)], stdout=log, stderr=subprocess.STDOUT)
        write_json(folder/'status.json', {'returncode': result.returncode, 'attempt': attempt})
        if result.returncode == 0:
            adapter = folder/'adapter'
            if not (adapter/'adapter_config.json').is_file() or not any(adapter.glob('adapter_model.*')):
                raise RuntimeError('training exited 0 but adapter artifacts are missing')
            return adapter
        log_text = (folder/'train.log').read_text(errors='replace').lower()
        if not any(term in log_text for term in ['cuda out of memory', 'cuda error: out of memory', 'torch.outofmemoryerror']):
            raise RuntimeError(f'training failed without OOM: {folder}/train.log')
        if attempt == max_attempts:
            raise RuntimeError('OOM retry budget exhausted')
        current = reduce_batch(current)
    raise RuntimeError('unreachable')


def export_model(base, adapter, output, cli):
    import yaml
    config = {'model_name_or_path': str(base), 'adapter_name_or_path': str(adapter), 'template': 'qwen', 'finetuning_type': 'lora', 'export_dir': str(output), 'export_size': 5, 'export_device': 'cpu', 'export_legacy_format': False}
    path = output.parent/(output.name+'_export.yaml')
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    with path.with_suffix('.log').open('w') as log:
        subprocess.run([cli, 'export', str(path)], stdout=log, stderr=subprocess.STDOUT, check=True)
    if not (output/'config.json').is_file() or not list(output.glob('*.safetensors')):
        raise RuntimeError('export artifacts incomplete')


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--base-model', required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--cli', default='llamafactory-cli')
    args = parser.parse_args()
    args.data_dir = args.data_dir.resolve(); args.run_dir = args.run_dir.resolve()
    stats = json.loads((args.data_dir/'statistics.json').read_text())
    if stats['mode'] == 'protocol_formal_tokenizer':
        raise ValueError('task3 data still requires content review, actual loader verification and final release before training')
    if stats['mode'] != 'formal_tokenizer':
        raise ValueError('training refuses quick smoke data; prepare with the real tokenizer')
    args.run_dir.mkdir(parents=True, exist_ok=False)
    sft = yaml.safe_load((ROOT/'deliverables/week3/day11/configs/experiments/epoch-e5.yaml').read_text())
    sft.update(model_name_or_path=args.base_model, dataset='week8_train', eval_dataset='week8_validation', dataset_dir=str(args.data_dir), eval_strategy='epoch', output_dir=str(args.run_dir/'sft/adapter'))
    sft.pop('max_samples', None)
    # Separate immutable data directory for DPO registration.
    pref_dir = args.run_dir/'preference_data'; pref_dir.mkdir()
    pref_source = ROOT/'deliverables/week4/day20/remediation/source/data'
    for name in ['week4_dpo_train_v2.json', 'week4_dpo_validation_v2.json', 'dataset_info.json']:
        shutil.copy2(pref_source/name, pref_dir/name)
    dpo = yaml.safe_load((ROOT/'deliverables/week4/day20/remediation/configs/reward_corrective_40step.yaml').read_text())
    dpo.update(model_name_or_path=str(args.run_dir/'models/final_sft'), dataset_dir=str(pref_dir), output_dir=str(args.run_dir/'dpo/adapter'))
    write_json(args.run_dir/'planned_configs.json', {'sft': sft, 'dpo': dpo, 'reference': 'DPO adapter-disabled implicit reference = freshly merged SFT', 'lineage_warning': 'new 9:1 split differs from historical 4999-row training; historical quality is not inherited'})
    if args.dry_run:
        write_json(args.run_dir/'status.json', {'status': 'planned_only', 'trained': False})
        return
    if not shutil.which(args.cli):
        raise RuntimeError('llamafactory-cli unavailable in current environment')
    try:
        model_dir = args.run_dir/'models'; model_dir.mkdir()
        adapter = execute_train(sft, args.run_dir/'sft', args.cli)
        export_model(args.base_model, adapter, model_dir/'final_sft', args.cli)
        adapter = execute_train(dpo, args.run_dir/'dpo', args.cli)
        export_model(model_dir/'final_sft', adapter, model_dir/'final_dpo', args.cli)
        write_json(args.run_dir/'status.json', {'status': 'completed', 'final_sft': str(model_dir/'final_sft'), 'final_dpo': str(model_dir/'final_dpo')})
    except Exception as exc:
        write_json(args.run_dir/'status.json', {'status': 'failed', 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
