"""Reproduce three model-bound evaluation candidates; no GPU or API calls."""
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
from mmengine.config import Config
from step3_eval import load_judge_profile

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads((ROOT / path).read_text())

def ref(path):
    return {'path': str(path.relative_to(ROOT)), 'sha256': digest(path)}

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

base_lock_path = ROOT / 'reports/week8/phase2_final_release/runtime_lock.json'
assert digest(base_lock_path) == '4986a43e393ebd2017a7e2fad7f8e8fcd063e8811e1b07852828c27b0e0968ec'
base_lock = json.loads(base_lock_path.read_text())
for item in base_lock['dependencies'] + [base_lock['records'], base_lock['judge_profile'], base_lock['release_review']]:
    assert digest(ROOT / item['path']) == item['sha256'], item['path']
load_judge_profile(base_lock)
review_path = ROOT / 'reports/week8/phase2_full_training_target/target_review.json'
assert digest(review_path) == '8ed9ef2be4df463191fe823e09ccd965c2e893f4c7c06098954781171c6efc76'
review = json.loads(review_path.read_text())
assert review['status'] == 'FORMAL_TRAINING_PASS_LOCAL_REVIEWED'
source_config = ROOT / 'reports/week8/phase2_final_release/remote_opencompass_formal.py'
assert digest(source_config) == base_lock['config']['sha256']
base_config = Config.fromfile(str(source_config)).to_dict()
assert len(base_config['datasets']) == 119
run = ROOT / 'reports/week8/phase2_full_training_target/evidence/logs/full-run-01'
tokenizer = ROOT / 'logs/week8-real-loader-20260915/attempt-02/tokenizer'
models = {}
for name in ('original_base', 'final_sft', 'final_dpo'):
    if name == 'original_base':
        source = ROOT / 'reports/week8/phase2_gpu_320/expected_base_identity.json'
        identity = json.loads(source.read_text())['files']
        model_path = base_lock['model_path']
    else:
        stage = name.removeprefix('final_')
        source = run / f'{stage}_merge_verification.json'
        receipt = json.loads(source.read_text())
        identity = receipt['identity']
        model_path = receipt['model_path']
        assert model_path.endswith('/models/' + name)
        hashes = {v['path']: v['sha256'] for v in identity}
        adapter = run / stage / 'attempt-01/adapter'
        for filename in ('tokenizer.json', 'tokenizer_config.json'):
            if filename == 'tokenizer.json':
                assert digest(adapter / filename) == hashes[filename]
            assert json.loads((adapter / filename).read_text()) == json.loads((tokenizer / filename).read_text())
        for filename in ('vocab.json', 'added_tokens.json', 'special_tokens_map.json'):
            assert digest(tokenizer / filename) == hashes[filename]
    assert len({v['path'] for v in identity}) == len(identity)
    folder = OUT / name
    folder.mkdir(exist_ok=True)
    identity_path = folder / 'model_identity.json'
    write(identity_path, {'model_path': model_path, 'files': identity, 'source': ref(source),
                          'target_recheck_required': True})
    config = copy.deepcopy(base_config)
    config['models'][0]['path'] = model_path
    config_path = folder / 'opencompass_formal.py'
    Config(config).dump(str(config_path))
    reparsed = Config.fromfile(str(config_path)).to_dict()
    reparsed['models'][0]['path'] = base_lock['model_path']
    assert reparsed == base_config, name
    lock = copy.deepcopy(base_lock)
    lock.pop('release_review')
    lock.update(verified=False, model_path=model_path, config=ref(config_path),
                model_identity=ref(identity_path), training_review=ref(review_path),
                pending=['Rehash all model files on instance 320; verify exact file set.',
                         'Validate all three custom20 tokenizers against frozen tokenizer: merged tokenizer_config hashes differ from checkpoint; merged files unavailable locally.',
                         'Validate target dependencies and model-specific config with evaluation entrypoint.',
                         'Bind fresh target receipt and release before inference.'])
    # The existing entrypoint checks every dependency, including absolute model files.
    lock['dependencies'] += [ref(identity_path)] + [
        {'path': str(Path(model_path) / item['path']), 'sha256': item['sha256']}
        for item in identity]
    lock_path = folder / 'runtime_lock_candidate.json'
    write(lock_path, lock)
    models[name] = {'model_path': model_path, 'identity_files': len(identity),
                    'config': ref(config_path), 'lock': ref(lock_path)}
records = read(base_lock['records']['path'])
counts = {dataset: sum(r['benchmark'] == dataset and r['split'] == split for r in records)
          for dataset, split in [('ceval', 'val'), ('cmmlu', 'test')]}
assert counts == {'ceval': 1346, 'cmmlu': 11582}
write(OUT / 'verification.json', {
    'status': 'THREE_MODEL_CANDIDATES_LOCAL_PASS_NOT_RELEASED',
    'models': models, 'shared_dependencies_rechecked': len(base_lock['dependencies']),
    'benchmark_questions_per_model': counts, 'custom_questions_per_model': 20,
    'total_benchmark_questions': 3 * sum(counts.values()), 'total_custom_questions': 60,
    'config_equivalent_except_model_path': True,
    'merged_tokenizer_json_sha256_equal_to_frozen': True,
    'checkpoint_tokenizer_config_semantically_equal_to_frozen': True,
    'merged_tokenizer_config_semantic_equivalence': 'PENDING_TARGET_CHECK',
    'training_review': ref(review_path), 'base_lock': ref(base_lock_path),
    'preparation_script': ref(Path(__file__).resolve()),
    'gpu_started': False, 'api_calls': 0, 'execution_enabled': False,
    'limitations': ['Remote model weights have not been rehashed in this evaluation stage.',
                    'No fresh benchmark or custom20 scores yet.',
                    'All three custom20 tokenizer configurations require target semantic comparison; merged config hashes differ from archived checkpoint configs.']})
print('PASS: 3 independent candidates, 119 subjects/model, 38,784 benchmark answers + 60 custom answers planned; not released.')
