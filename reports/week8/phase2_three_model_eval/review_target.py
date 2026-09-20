"""Review downloaded CPU audit without altering historical candidates."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
receipt_path = HERE / 'target_audit.json'
receipt = read(receipt_path)
assert receipt['status'] == 'THREE_MODEL_TARGET_CPU_AUDIT_PASS'
assert receipt['host'] == 'autodl-container-be044ebe99-be706b14'
assert receipt['audit_script_sha256'] == sha(HERE / 'target_audit.py')
assert receipt['gpu_inference'] is False and receipt['judge_api_calls'] == 0
models = receipt['models']
assert set(models) == {'original_base', 'final_sft', 'final_dpo'}
spec = read(ROOT / 'deliverables/week3/day14/source/data/evaluation_questions.json')
ids = [q['id'] for q in spec['questions']]
assert len(ids) == len(set(ids)) == 20
baseline = models['original_base']
checked = 0
for name, model in models.items():
    candidate_path = HERE / name / 'runtime_lock_candidate.json'
    lock = read(candidate_path)
    assert sha(candidate_path) == model['candidate_sha256']
    assert model['model_path'] == lock['model_path']
    identity = read(HERE / name / 'model_identity.json')
    assert model['identity_files_verified'] == len(identity['files'])
    assert model['dependencies_verified'] == len(lock['dependencies'])
    rows = model['custom20_token_ids']
    assert [row['id'] for row in rows] == ids
    assert rows == baseline['custom20_token_ids']
    assert all(type(row['tokens']) is int and row['tokens'] > 0 for row in rows)
    assert model['generation_config_effective'] == baseline['generation_config_effective']
    assert model['generation_config_effective']['do_sample'] is False
    assert model['generation_config_effective']['max_new_tokens'] == 512
    if name != 'original_base':
        assert model['tokenizer_config_differences'] == {'padding_side': {'frozen': 'right', 'model': 'left'}}
    checked += len(identity['files'])
review = {'status': 'THREE_MODEL_CPU_AUDIT_LOCAL_REVIEW_PASS',
          'target_receipt': {'path': str(receipt_path.relative_to(ROOT)), 'sha256': sha(receipt_path)},
          'model_identity_files_verified_on_target': checked,
          'custom20_input_comparisons': 60,
          'effective_generation_parameters_equal': True,
          'merged_tokenizer_difference': 'padding_side only; no padding in current single-question custom20',
          'review_script_sha256': sha(Path(__file__)),
          'formal_scores_available': False,
          'next': 'Install model-specific locks and verify actual entrypoint before GPU inference.'}
(HERE / 'target_review.json').write_text(json.dumps(review, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(review, ensure_ascii=False, indent=2))
