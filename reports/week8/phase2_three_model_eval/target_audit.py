"""Read-only CPU audit of all three model identities and custom20 inputs."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sys

os.environ['USE_TORCH'] = '0'
os.environ['USE_TF'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
ROOT = Path('/root/autodl-tmp/week8-judge-runtime-20260916')
HERE = Path(__file__).resolve().parent
OUT = HERE / 'target_audit.json'
assert not OUT.exists(), 'audit output already exists'
assert socket.gethostname() == 'autodl-container-be044ebe99-be706b14'
from transformers import AutoTokenizer, GenerationConfig
from mmengine.config import Config
sys.path.insert(0, str(ROOT / 'scripts'))
from step3_eval import load_judge_profile

def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()

def read(path):
    return json.loads(path.read_text())

frozen_path = ROOT / 'logs/week8-real-loader-20260915/attempt-02/tokenizer'
frozen = AutoTokenizer.from_pretrained(frozen_path, local_files_only=True)
frozen_config = read(frozen_path / 'tokenizer_config.json')
spec = read(ROOT / 'deliverables/week3/day14/source/data/evaluation_questions.json')
assert len(spec['questions']) == 20
base_config = Config.fromfile(str(ROOT / 'target-check/formal-config/opencompass_formal.py')).to_dict()
results = {}
base_generation = None
for name in ('original_base', 'final_sft', 'final_dpo'):
    folder = HERE / name
    lock = read(folder / 'runtime_lock_candidate.json')
    identity_path = folder / 'model_identity.json'
    identity = read(identity_path)
    assert sha(identity_path) == lock['model_identity']['sha256']
    assert sha(folder / 'opencompass_formal.py') == lock['config']['sha256']
    model_path = Path(lock['model_path'])
    assert identity['model_path'] == str(model_path)
    actual = {str(p.relative_to(model_path)) for p in model_path.rglob('*') if p.is_file()}
    expected = {v['path'] for v in identity['files']}
    assert actual == expected, (name, 'file set', actual ^ expected)
    for item in identity['files']:
        p = model_path / item['path']
        assert p.stat().st_size == item['bytes'] and sha(p) == item['sha256'], str(p)
    for item in lock['dependencies']:
        # This manifest was verified above; it has not been installed into ROOT.
        p = identity_path if item['path'] == lock['model_identity']['path'] else ROOT / item['path']
        if p.is_absolute() and str(p).startswith(str(model_path) + '/'):
            continue  # already streamed and verified, do not read 15 GB twice
        assert sha(p) == item['sha256'], str(p)
    assert sha(ROOT / lock['records']['path']) == lock['records']['sha256']
    load_judge_profile(lock)
    config = Config.fromfile(str(folder / 'opencompass_formal.py')).to_dict()
    assert config['models'][0]['path'] == str(model_path)
    config['models'][0]['path'] = base_config['models'][0]['path']
    assert config == base_config
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer_config = read(model_path / 'tokenizer_config.json')
    differences = {k: {'frozen': frozen_config.get(k), 'model': tokenizer_config.get(k)}
                   for k in sorted(set(frozen_config) | set(tokenizer_config))
                   if frozen_config.get(k) != tokenizer_config.get(k)}
    assert tokenizer.chat_template == frozen.chat_template
    assert tokenizer.get_vocab() == frozen.get_vocab()
    rows = []
    for q in spec['questions']:
        messages = [{'role': 'system', 'content': spec['system_message']}] + q['messages']
        expected_ids = frozen.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        actual_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        assert actual_ids == expected_ids, (name, q['id'])
        rows.append({'id': q['id'], 'tokens': len(actual_ids),
                     'token_ids_sha256': hashlib.sha256(json.dumps(actual_ids).encode()).hexdigest()})
    generation = GenerationConfig.from_pretrained(model_path, local_files_only=True).to_dict()
    # Version/provenance has no inference effect; all real generation fields must match.
    effective = {k: v for k, v in generation.items() if k not in ('transformers_version', '_from_model_config', '_commit_hash')}
    effective.update(do_sample=False, max_new_tokens=512)
    if base_generation is None:
        base_generation = effective
    assert effective == base_generation, (name, 'effective generation config differs')
    results[name] = {'identity_files_verified': len(expected), 'dependencies_verified': len(lock['dependencies']),
                     'model_path': str(model_path), 'candidate_sha256': sha(folder / 'runtime_lock_candidate.json'),
                     'tokenizer_config_differences': differences, 'custom20_token_ids': rows,
                     'generation_config_raw': read(model_path / 'generation_config.json'),
                     'generation_config_effective': effective,
                     'tokenizer_config_raw': tokenizer_config}
    print('PASS', name, len(expected), 'identity files; 20 identical custom20 inputs', flush=True)

receipt = {'status': 'THREE_MODEL_TARGET_CPU_AUDIT_PASS', 'host': socket.gethostname(),
           'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'models': results, 'audit_script_sha256': sha(Path(__file__)),
           'disk_free_bytes': shutil.disk_usage('/root/autodl-tmp').free,
           'custom20_batching': 'one question, no padding', 'model_weight_loading': False,
           'gpu_inference': False, 'judge_api_calls': 0, 'evaluation_released': False}
OUT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
print('RECEIPT_SHA256', sha(OUT), flush=True)
