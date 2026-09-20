#!/usr/bin/env python3
"""Day42 sequence-level distillation, with explicit teacher/student stages."""
import argparse
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from common import ROOT, read_records, sha256, write_json

TEACHER_MANIFEST_FILE_SHA256 = '39d6abfaa8d260be36524c10def770fd1381a6ad12d14f1cd145f220b39964a7'
TEACHER_FILE_LIST_SHA256 = 'aca9a2932ed427f8fc6e9962e70274eb286a09c446d177af2bc00f6df1675e4c'


def verify_teacher_identity(args):
    from compare_distillation import verify_model_manifest
    manifest = getattr(args, 'teacher_manifest', None)
    if not args.teacher or not manifest:
        raise ValueError('--teacher and --teacher-manifest are required')
    if sha256(manifest) != TEACHER_MANIFEST_FILE_SHA256:
        raise ValueError('teacher manifest is not the archived Week4 corrective DPO')
    return verify_model_manifest(args.teacher, manifest)


def verify_target_review(targets, receipt_path, review_path, config, curation_path=None):
    if not receipt_path or not review_path:
        raise ValueError('--generation-receipt and --quality-review are required')
    receipt_path = Path(receipt_path).resolve()
    receipt = json.loads(receipt_path.read_text())
    review = json.loads(Path(review_path).read_text())
    if receipt.get('status') != 'GENERATED_PENDING_QUALITY_REVIEW' or receipt.get('config') != config:
        raise ValueError('generation receipt or experiment configuration differs')
    files = receipt.get('files', [])
    if not files or len({e['path'] for e in files}) != len(files):
        raise ValueError('generation artifact bindings missing or duplicated')
    bound = {}
    for entry in files:
        path = (receipt_path.parent / entry['path']).resolve()
        if not path.is_relative_to(receipt_path.parent) or sha256(path) != entry['sha256']:
            raise ValueError('generation artifact changed or escaped output directory')
        bound[path] = entry['sha256']
    requested_targets = Path(targets).resolve()
    targets = receipt_path.parent/'teacher_targets.json' if curation_path else requested_targets
    if targets not in bound:
        raise ValueError('training targets are not bound to this generation')
    rows = read_records(targets)
    ids = [r.get('sample_id') for r in rows]
    if len(rows) < 10 or len(rows) != receipt.get('accepted') or any(not x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError('accepted target identity/count mismatch')
    selected_path = receipt_path.parent/'selected_prompts.json'
    teacher_path = receipt_path.parent/'teacher_identity.json'
    raw_paths = sorted(receipt_path.parent.glob('teacher_raw/*.json'))
    required = {targets, selected_path, teacher_path, *raw_paths}
    if set(bound) != required or not raw_paths:
        raise ValueError('generation must bind selected prompts, teacher identity and all raw answers')
    teacher_sha = TEACHER_MANIFEST_FILE_SHA256
    if sha256(teacher_path) != teacher_sha or receipt.get('teacher_manifest_sha256') != teacher_sha:
        raise ValueError('generation teacher differs from Week4 archive')
    if json.loads(teacher_path.read_text()).get('manifest_sha256') != TEACHER_FILE_LIST_SHA256:
        raise ValueError('teacher historical file-list fingerprint differs')
    selected = read_records(selected_path)
    if len(selected) != receipt.get('generated') or len(raw_paths) != len(selected):
        raise ValueError('generated prompt/raw answer count mismatch')
    selected_ids = [r.get('sample_id') for r in selected]
    if len(set(selected_ids)) != len(selected_ids) or any(not s for s in selected_ids):
        raise ValueError('selected prompt identities missing or duplicated')
    reconstructed = []
    for prompt, path in zip(selected, raw_paths):
        raw = json.loads(path.read_text())
        if (raw.get('sample_id') != prompt['sample_id'] or raw.get('messages') != teacher_messages(prompt)
                or type(raw.get('truncated')) is not bool or not isinstance(raw.get('answer'), str)):
            raise ValueError('teacher answer does not match its selected prompt')
        if raw['answer'].strip() and not raw['truncated']:
            reconstructed.append({**{k: prompt[k] for k in ('sample_id','instruction','input','system','history') if k in prompt},
                                  'output': raw['answer']})
    if rows != reconstructed:
        raise ValueError('student targets differ from complete teacher answers')
    if curation_path:
        verify_curated_subset(requested_targets, curation_path, receipt_path, selected, raw_paths, rows)
        targets = requested_targets
        rows = read_records(targets)
        ids = [row['sample_id'] for row in rows]
        checked_ids = review.get('reviewed_sample_ids', [])
        if (review.get('curation_manifest_sha256') != sha256(curation_path)
                or not isinstance(checked_ids, list) or len(checked_ids) != len(ids)
                or set(checked_ids) != set(ids)):
            raise ValueError('curated subset requires a current complete quality review')
        receipt = dict(receipt, curated_count=len(rows), curation_manifest_sha256=sha256(curation_path))
    checked = review.get('reviewed_sample_ids', [])
    if (review.get('status') != 'APPROVED_FOR_STUDENT_TRAINING'
            or review.get('generation_receipt_sha256') != sha256(receipt_path)
            or review.get('targets_sha256') != sha256(targets)
            or not isinstance(checked, list) or len(set(checked)) < 10 or not set(checked) <= set(ids)
            or not isinstance(review.get('notes'), str) or not review['notes'].strip()
            or review.get('unresolved_issues') != []):
        raise ValueError('teacher target review is missing, stale or has unresolved issues')
    return receipt


def verify_curated_subset(targets, manifest_path, receipt_path, selected, raw_paths, original_rows):
    """Allow removal only, with decisions bound to every original generated answer."""
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    def bound_path(field, expected):
        value = manifest.get(field)
        if not isinstance(value, str):
            raise ValueError('curation artifact path missing')
        path = (manifest_path.parent/value).resolve()
        if not path.is_relative_to(manifest_path.parent) or path != expected.resolve():
            raise ValueError('curation artifact path differs or escapes directory')
    if (manifest.get('schema_version') != 1
            or manifest.get('status') not in ('CURATION_COMPLETE_TRAINING_GATE_PENDING', 'CURATION_COMPLETE')
            or manifest.get('generation_receipt_sha256') != sha256(receipt_path)
            or manifest.get('original_targets_sha256') != sha256(receipt_path.parent/'teacher_targets.json')
            or manifest.get('curated_targets_sha256') != sha256(targets)):
        raise ValueError('curation identity, state or hashes differ')
    bound_path('generation_receipt_path', receipt_path)
    bound_path('original_targets_path', receipt_path.parent/'teacher_targets.json')
    bound_path('curated_targets_path', targets)
    decisions = manifest.get('decisions')
    if not isinstance(decisions, list) or len(decisions) != len(selected):
        raise ValueError('curation must review every generated answer')
    original = {row['sample_id']:row for row in original_rows}
    retained = []
    truncated_count = 0
    for index, (prompt, raw_path, decision) in enumerate(zip(selected, raw_paths, decisions)):
        if not isinstance(decision, dict):
            raise ValueError('invalid curation decision')
        raw = json.loads(raw_path.read_text())
        truncated_count += int(raw['truncated'])
        if (decision.get('index') != index or decision.get('sample_id') != prompt['sample_id']
                or decision.get('raw_sha256') != sha256(raw_path)
                or decision.get('decision') not in ('KEEP', 'EXCLUDE')
                or not isinstance(decision.get('notes'), str) or not decision['notes'].strip()
                or not isinstance(decision.get('reason_code'), str) or not decision['reason_code'].strip()):
            raise ValueError('curation decision is missing, duplicated or not bound to raw answer')
        raw_name = decision.get('raw_path')
        if not isinstance(raw_name, str):
            raise ValueError('curation raw path missing')
        resolved = (manifest_path.parent/raw_name).resolve()
        if not resolved.is_relative_to(manifest_path.parent) or resolved != raw_path:
            raise ValueError('curation raw path differs or escapes directory')
        if decision['decision'] == 'KEEP':
            if raw['truncated'] or prompt['sample_id'] not in original or decision['reason_code'] != 'KEEP':
                raise ValueError('curation retains an incomplete or excluded answer')
            retained.append(original[prompt['sample_id']])
        elif decision['reason_code'] == 'KEEP':
            raise ValueError('excluded answer has contradictory keep reason')
    expected_counts = dict(generated=len(selected), complete=len(original_rows), truncated=truncated_count,
                           retained=len(retained), excluded_complete=len(original_rows)-len(retained),
                           excluded_total=len(selected)-len(retained))
    if manifest.get('counts') != expected_counts:
        raise ValueError('curation counts do not match decisions')
    if len(retained) < 10 or read_records(targets) != retained:
        raise ValueError('curated targets must be the exact ordered subset of retained teacher answers')


def student_training_schedule(train_count):
    """Single-device batch=1, accumulation=8, verified on target Transformers 4.50."""
    if type(train_count) is not int or train_count < 8:
        raise ValueError('need at least eight student training examples')
    updates = math.ceil(train_count / 8)
    return {'train_count':train_count, 'gradient_accumulation_steps':8, 'steps_per_epoch':updates,
            'max_steps':2*updates, 'trainer_epoch_bound':math.ceil(2*updates / (train_count//8)),
            'sample_presentations':2*train_count}


def validate_student_state(state, schedule=None):
    from week8_full_training import validate_formal_state
    steps = state.get('max_steps')
    if type(steps) is not int or steps <= 0:
        raise ValueError('missing positive planned optimizer step count')
    loop_epochs = 2
    if schedule is not None:
        if schedule != student_training_schedule(schedule['train_count']) or steps != schedule['max_steps']:
            raise ValueError('optimizer steps differ from complete two-epoch schedule')
        loop_epochs = schedule['trainer_epoch_bound']
    epoch = state.get('epoch')
    if (state.get('num_train_epochs') != loop_epochs or type(epoch) not in (int, float)
            or not math.isfinite(epoch) or abs(epoch-2) > 1e-6):
        raise ValueError('student has not completed two epochs')
    rows = validate_formal_state(state, steps)
    evaluations = [r for r in state.get('log_history', []) if 'eval_loss' in r]
    if len(evaluations) != 2:
        raise ValueError('need validation after both student epochs')
    for i, row in enumerate(evaluations, 1):
        if (type(row.get('epoch')) not in (int, float) or not math.isfinite(row['epoch'])
                or abs(row['epoch']-i) > 1e-6 or type(row.get('eval_loss')) not in (int, float)
                or not math.isfinite(row['eval_loss'])):
            raise ValueError('student validation epoch or loss invalid')
    if not 0 < evaluations[0].get('step', 0) < evaluations[1].get('step', 0) == steps:
        raise ValueError('student validation step cadence invalid')
    if schedule is not None and evaluations[0]['step'] != schedule['steps_per_epoch']:
        raise ValueError('first epoch did not consume the planned training batches')
    return dict(global_step=steps, epochs=2, first_loss=rows[0]['loss'], last_loss=rows[-1]['loss'], validation=evaluations)


def teacher_messages(row):
    messages = []
    if row.get('system'):
        messages.append({'role': 'system', 'content': row['system']})
    for user, assistant in row.get('history', []):
        messages.extend([{'role': 'user', 'content': user}, {'role': 'assistant', 'content': assistant}])
    prompt = row['instruction'] + ('\n' + row['input'] if row.get('input') else '')
    messages.append({'role': 'user', 'content': prompt})
    return messages


def generate(args, config):
    teacher_identity = verify_teacher_identity(args)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError('teacher generation needs CUDA; no synthetic fallback')
    if not args.teacher:
        raise ValueError('--teacher must be the verified Week4 DPO model')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records = read_records(args.input)
    random.Random(config['seed']).shuffle(records)
    records = records[:config['max_samples']]
    if len(records) < 10:
        raise ValueError('need at least ten training prompts')
    ids = [r.get('sample_id') for r in records]
    if any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError('teacher prompts require unique lineage sample_id values')
    write_json(args.output_dir/'selected_prompts.json', records)
    shutil.copyfile(args.teacher_manifest, args.output_dir/'teacher_identity.json')
    tokenizer = AutoTokenizer.from_pretrained(args.teacher, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=torch.bfloat16, device_map='auto', local_files_only=True).eval()
    torch.manual_seed(config['seed'])
    generated = []
    for i, row in enumerate(records):
        prompt = row['instruction'] + ('\n'+row.get('input', '') if row.get('input') else '')
        messages = teacher_messages(row)
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt').to(model.device)
        with torch.inference_mode():
            tokens = model.generate(ids, max_new_tokens=config['max_new_tokens'], do_sample=True, temperature=config['temperature'], pad_token_id=tokenizer.eos_token_id)
        answer = tokenizer.decode(tokens[0, ids.shape[1]:], skip_special_tokens=True).strip()
        truncated = tokens[0, -1].item() != tokenizer.eos_token_id and tokens.shape[1]-ids.shape[1] >= config['max_new_tokens']
        write_json(args.output_dir/'teacher_raw'/f'{i:04d}.json', {'sample_id': row['sample_id'], 'prompt': prompt, 'messages': messages, 'answer': answer, 'truncated': truncated, 'input_tokens': ids.shape[1], 'output_tokens': tokens.shape[1]-ids.shape[1]})
        if answer and not truncated:
            generated.append({**{key: row[key] for key in ('sample_id', 'instruction', 'input', 'system', 'history') if key in row}, 'output': answer})
    if len(generated) < 10:
        raise RuntimeError('too few complete teacher answers; inspect generation audit')
    write_json(args.output_dir/'teacher_targets.json', generated)
    verify_teacher_identity(args)
    files = [args.output_dir/'selected_prompts.json', args.output_dir/'teacher_identity.json',
             args.output_dir/'teacher_targets.json', *sorted((args.output_dir/'teacher_raw').glob('*.json'))]
    write_json(args.output_dir/'generation_receipt.json', {'status': 'GENERATED_PENDING_QUALITY_REVIEW', 'teacher': args.teacher,
        'teacher_manifest_sha256': sha256(args.teacher_manifest), 'input_sha256': sha256(args.input), 'config': config,
        'generated': len(records), 'accepted': len(generated), 'method': config['method'],
        'files': [{'path': str(p.relative_to(args.output_dir)), 'sha256': sha256(p)} for p in files]})


def prepare_student_data(source, student, output, config):
    """Reuse cleaning with explicit distillation inputs, not the frozen 7B CLI."""
    from step1_data_prep import prepare
    return prepare(argparse.Namespace(
        input=Path(source), tokenizer=str(student), output_dir=Path(output),
        quick=False, seed=config['seed'], max_tokens=config['cutoff_len']))


def verify_student_artifacts(base, adapter, merged, output, timeout=900):
    from compare_distillation import run_bounded, verify_model_manifest
    from week8_full_training import verify_cold_receipt
    output, merged = Path(output).resolve(), Path(merged).resolve()
    deadline = time.monotonic() + timeout
    tensors = output/'student_tensor_verification.json'
    run_bounded([sys.executable, str(ROOT/'scripts/week8_distillation_verify.py'),
                 '--base', str(base), '--adapter', str(adapter), '--merged', str(merged),
                 '--output', str(tensors)], output/'student_tensor_verification.log', deadline-time.monotonic())
    manifest = verify_model_manifest(merged, tensors)
    if manifest.get('status') != 'TENSOR_MERGE_VERIFIED':
        raise ValueError('tensor merge verification did not pass')
    cold = output/'student_cold_load.json'
    run_bounded([sys.executable, str(ROOT/'scripts/week8_full_cold_load.py'),
                 '--model', str(merged), '--output', str(cold)],
                output/'student_cold_load.log', min(300, deadline-time.monotonic()))
    verify_cold_receipt(cold, merged)
    verify_model_manifest(merged, tensors)
    receipt = {'status':'STUDENT_ARTIFACTS_VERIFIED_COMPARISON_PENDING',
               'model_dir':str(merged), 'tensor_receipt_sha256':sha256(tensors),
               'cold_load_receipt_sha256':sha256(cold), 'quality_comparison':'pending'}
    write_json(output/'student_artifact_verification.json', receipt)
    return receipt


def train(args, config):
    generation = verify_target_review(args.input, getattr(args, 'generation_receipt', None),
                                      getattr(args, 'quality_review', None), config,
                                      curation_path=getattr(args, 'curation_manifest', None))
    import yaml
    from compare_distillation import verify_model_manifest
    from train_pipeline import execute_train, export_model
    if not args.student:
        raise ValueError('--student must point to a local Qwen2.5-0.5B-Instruct snapshot')
    if not getattr(args, 'student_manifest', None):
        raise ValueError('--student-manifest is required')
    student_identity = verify_model_manifest(args.student, args.student_manifest)
    model_config = json.loads((Path(args.student)/'config.json').read_text())
    if model_config.get('hidden_size') != 896 or model_config.get('num_hidden_layers') != 24:
        raise ValueError('student is not the expected Qwen2.5-0.5B architecture')
    if config['epochs'] != 2:
        raise ValueError('teacher requirement fixes this feasibility experiment at two epochs')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    stats = prepare_student_data(args.input, args.student, args.output_dir/'data', config)
    if stats['train_count'] < 8 or stats['validation_count'] < 1 or stats['prompt_overlap'] != 0:
        raise ValueError('insufficient or overlapping student train/validation data')
    from importlib.metadata import version
    if version('transformers') != '4.50.0':
        raise ValueError('student schedule requires the audited target Transformers 4.50.0')
    schedule = student_training_schedule(stats['train_count'])
    write_json(args.output_dir/'student_training_schedule.json', schedule)
    write_json(args.output_dir/'input_binding.json', {'generation_receipt_sha256': sha256(args.generation_receipt),
        'quality_review_sha256': sha256(args.quality_review), 'teacher_targets_sha256': sha256(args.input),
        'student_identity': student_identity, 'config': config, 'accepted_teacher_answers': generation['accepted'],
        'curated_teacher_answers': generation.get('curated_count'),
        'curation_manifest_sha256': generation.get('curation_manifest_sha256'), 'data_statistics': stats})
    effective = {
        'model_name_or_path': args.student, 'stage': 'sft', 'do_train': True,
        'finetuning_type': 'lora', 'lora_target': 'all', 'lora_rank': config['lora_rank'], 'lora_alpha': config['lora_alpha'],
        'template': 'qwen', 'dataset': 'week8_train', 'eval_dataset': 'week8_validation', 'dataset_dir': str((args.output_dir/'data').resolve()),
        'num_train_epochs': 2, 'max_steps':schedule['max_steps'], 'learning_rate': config['learning_rate'], 'per_device_train_batch_size': 1, 'gradient_accumulation_steps': 8,
        'cutoff_len': config['cutoff_len'], 'bf16': True, 'seed': config['seed'], 'logging_steps': 1, 'eval_strategy': 'epoch',
        'save_strategy': 'epoch', 'save_total_limit': 1, 'report_to': 'tensorboard', 'overwrite_output_dir': False,
    }
    (args.output_dir/'distill_train.yaml').write_text(yaml.safe_dump(effective, sort_keys=False))
    try:
        adapter = execute_train(effective, args.output_dir/'training', 'llamafactory-cli', max_attempts=1)
        state_path = adapter/'trainer_state.json'
        state = validate_student_state(json.loads(state_path.read_text()), schedule)
        state['trainer_state_sha256'] = sha256(state_path)
        write_json(args.output_dir/'student_state_verification.json', state)
        export_model(args.student, adapter, args.output_dir/'final_distilled', 'llamafactory-cli')
        verify_model_manifest(args.student, args.student_manifest)
        write_json(args.output_dir/'status.json', {'status': 'EXPORTED_PENDING_TENSOR_AND_COLD_LOAD_VERIFICATION',
            'training_state_verified': True, 'training_verified': False, 'epochs': 2,
            'method': config['method'], 'quality_comparison': 'pending'})
        artifacts = verify_student_artifacts(args.student, adapter, args.output_dir/'final_distilled', args.output_dir)
        verify_model_manifest(args.student, args.student_manifest)
        write_json(args.output_dir/'status.json', {'status':'STUDENT_TRAINING_VERIFIED_COMPARISON_PENDING',
            'training_state_verified':True, 'training_verified':True, 'epochs':2, 'method':config['method'],
            'artifact_verification':artifacts, 'quality_comparison':'pending'})
    except Exception as exc:
        write_json(args.output_dir/'status.json', {'status': 'FAILED', 'training_verified': False, 'error': str(exc)})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['generate', 'train'])
    p.add_argument('--config', type=Path, default=ROOT/'configs/distillation.json')
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--teacher'); p.add_argument('--student')
    p.add_argument('--teacher-manifest', type=Path)
    p.add_argument('--student-manifest', type=Path)
    p.add_argument('--generation-receipt', type=Path)
    p.add_argument('--quality-review', type=Path)
    p.add_argument('--curation-manifest', type=Path)
    args = p.parse_args(); args.output_dir = args.output_dir.resolve()
    config = json.loads(args.config.read_text())
    generate(args, config) if args.stage=='generate' else train(args, config)


if __name__ == '__main__':
    main()
