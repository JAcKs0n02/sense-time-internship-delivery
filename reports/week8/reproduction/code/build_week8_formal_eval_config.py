#!/usr/bin/env python3
"""Build a weight-loading evaluation config; never authorize or execute it."""
import argparse
import json
from pathlib import Path
from common import ROOT, sha256, write_json


def build_config(root, model_path):
    from mmengine.config import Config
    root = Path(root).resolve()
    model_path = Path(model_path)
    if not model_path.is_absolute():
        raise ValueError('model path must be absolute and bound to the target host')
    evidence = root / 'reports/week8/phase2_prompt_fix/runtime-03'
    receipt = json.loads((evidence / 'verification.json').read_text())
    if receipt['status'] != 'ACTUAL_OC_PROMPT_RUNTIME_PASS_NO_INFERENCE':
        raise ValueError('prompt verification is incomplete')
    checked = {
        'expanded_config_sha256': evidence / 'expanded_audit_config.py',
        'custom_template_sha256': root / 'scripts/week8_prompt_template.py',
        'expected_rows_sha256': root / 'reports/week8/phase2_eval_hardening_v2/prompt_lengths.json',
    }
    for key, path in checked.items():
        if sha256(path) != receipt[key]:
            raise ValueError(f'checked prompt artifact changed: {key}')
    base = root / 'reports/week8/phase2_preflight'
    lock = json.loads((base / 'benchmark_input_lock.json').read_text())
    manifest = base / 'local_data_manifest.json'
    if sha256(manifest) != lock['local_data_manifest_sha256']:
        raise ValueError('input manifest changed')
    dependencies = [*checked.values(), manifest, base / 'benchmark_input_lock.json',
                    root / 'scripts/step3_eval.py', root / 'scripts/__init__.py', Path(__file__).resolve()]
    for entry in json.loads(manifest.read_text()):
        path = base / entry['path']
        if sha256(path) != entry['sha256']:
            raise ValueError(f'benchmark data changed: {path}')
        dependencies.append(path)
    records = base / 'benchmark_records.json'
    if sha256(records) != lock['benchmark_records_sha256']:
        raise ValueError('benchmark records changed')
    cfg = Config.fromfile(str(evidence / 'expanded_audit_config.py')).to_dict()
    for dataset in cfg['datasets']:
        benchmark = dataset['abbr'].split('-')[0]
        dataset['path'] = str(base / 'local_data' / benchmark)
        if dataset['infer_cfg']['ice_template']['type'] != 'scripts.week8_prompt_template.NonRecursivePromptTemplate':
            raise ValueError('unexpected prompt template')
    tokenizer = root / 'logs/week8-real-loader-20260915/attempt-02/tokenizer'
    dependencies.extend(p for p in sorted(tokenizer.iterdir()) if p.is_file())
    cfg['models'] = [dict(
        type='opencompass.models.huggingface_above_v4_33.HuggingFacewithChatTemplate',
        abbr='week8-bound-model', path=str(model_path), tokenizer_path=str(tokenizer),
        tokenizer_only=False, max_seq_len=2048, max_out_len=32, batch_size=1,
        tokenizer_kwargs=dict(local_files_only=True, trust_remote_code=False),
        model_kwargs=dict(local_files_only=True, trust_remote_code=False,
                          torch_dtype='bfloat16', device_map='auto'),
        generation_kwargs=dict(do_sample=False, num_beams=1),
        run_cfg=dict(num_gpus=1),
    )]
    cfg['seed'] = 42
    artifact = lambda path: dict(path=str(path.relative_to(root)), sha256=sha256(path))
    binding = dict(verified=False, model_path=str(model_path), records=artifact(records),
                   dependencies=[artifact(p) for p in dict.fromkeys(dependencies)], judge=None,
                   pending=['target Linux prompt recheck', 'model identity and forward verification',
                            'judge exact identity and endpoint', 'explicit runtime release'])
    return cfg, binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    config, binding = build_config(ROOT, args.model)
    output = args.output_dir.resolve()
    output.relative_to(ROOT)
    output.mkdir(parents=True, exist_ok=False)
    from mmengine.config import Config
    path = output / 'opencompass_formal.py'
    Config(config).dump(str(path))
    binding['config'] = dict(path=str(path.relative_to(ROOT)), sha256=sha256(path))
    write_json(output / 'runtime_lock_candidate.json', binding)
    print(f'CONFIG_BUILT_NOT_RELEASED {path}')


if __name__ == '__main__':
    main()
