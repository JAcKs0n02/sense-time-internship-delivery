"""Bind reviewed evidence and check real entrypoint; never run fresh()."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import socket
import sys
from types import SimpleNamespace

ROOT = Path('/root/autodl-tmp/week8-judge-runtime-20260916')
HERE = Path(__file__).resolve().parent
DEST = ROOT / 'reports/week8/phase2_three_model_eval'
assert socket.gethostname() == 'autodl-container-be044ebe99-be706b14'
assert not DEST.exists(), 'refuse to overwrite deployed locks'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
assert sha(HERE / 'target_audit.json') == '7b897741b926728fa3989213d5d5a80bf823927ec957aa5348cdbcb65f7e1525'
assert sha(HERE / 'target_review.json') == '80acf6dddecb6670642e2fb1d110827401d4f68de2d3148fbc52638f23891d6a'
assert read(HERE / 'target_review.json')['status'] == 'THREE_MODEL_CPU_AUDIT_LOCAL_REVIEW_PASS'
DEST.mkdir(parents=True)
for filename in ('target_audit.json', 'target_review.json'):
    shutil.copy2(HERE / filename, DEST / filename)
sys.path.insert(0, str(ROOT / 'scripts'))
from step3_eval import require_evaluation_lock

preflight_path = ROOT / 'configs/week8_evaluation_preflight.json'
original = preflight_path.read_bytes()
(DEST / 'preflight_before_release.json').write_bytes(original)
results = {}
try:
    for name in ('original_base', 'final_sft', 'final_dpo'):
        shutil.copytree(HERE / name, DEST / name)
        lock = read(DEST / name / 'runtime_lock_candidate.json')
        lock['verified'] = True
        lock['pending'] = []
        def reference(filename):
            p = DEST / filename
            return {'path': str(p.relative_to(ROOT)), 'sha256': sha(p)}
        lock['release_review'] = reference('target_review.json')
        lock['target_audit'] = reference('target_audit.json')
        lock['dependencies'] += [lock['release_review'], lock['target_audit']]
        path = DEST / name / 'runtime_lock.json'
        path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + '\n')
        preflight = json.loads(original)
        preflight['runtime_lock'] = lock
        preflight_path.write_text(json.dumps(preflight, ensure_ascii=False, indent=2) + '\n')
        require_evaluation_lock(SimpleNamespace(model=lock['model_path'], judge_model='deepseek-flash', judge_url='https://api.deepseek.com'))
        results[name] = {'runtime_lock_sha256': sha(path), 'dependencies_checked': len(lock['dependencies']),
                         'model_path': lock['model_path'], 'entrypoint_gate': 'PASS'}
        print('ENTRYPOINT_PASS', name, flush=True)
finally:
    preflight_path.write_bytes(original)
assert preflight_path.read_bytes() == original
receipt = {'status': 'THREE_MODEL_EVALUATION_LOCKS_RELEASED_ENTRYPOINT_PASS',
           'host': socket.gethostname(), 'models': results,
           'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'script_sha256': sha(Path(__file__)), 'target_root': str(ROOT),
           'preflight_restored': True, 'gpu_inference': False, 'judge_api_calls': 0,
           'session_execution_started': False}
path = HERE / 'release_receipt.json'
path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
import tarfile
with tarfile.open(HERE / 'released_locks.tar.gz', 'w:gz') as archive:
    for name in results:
        archive.add(DEST / name / 'runtime_lock.json', arcname=name + '/runtime_lock.json')
    archive.add(path, arcname='release_receipt.json')
print('ARCHIVE_SHA256', sha(HERE / 'released_locks.tar.gz'), flush=True)
