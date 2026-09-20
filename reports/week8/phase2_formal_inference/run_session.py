"""One sequential, bounded session using the three released evaluation locks."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

ROOT = Path('/root/autodl-tmp/week8-judge-runtime-20260916')
OUT = ROOT / 'logs/formal-eval-20260916-01'
LOCKS = ROOT / 'reports/week8/phase2_three_model_eval'
DEADLINE = datetime.datetime.fromisoformat('2026-09-17T10:10:00+00:00').timestamp()
EXPECTED = {
    'original_base': 'ddf8cf644799a6e5fbe2946c911bad6424dcef14148411c927e44269cf2856e0',
    'final_sft': 'a108d93844830964731ac191f2a87a1a506da4c98431fa9c4ae4a48cac66fc6c',
    'final_dpo': 'e385687c3d6775324fdbb792b6a25d375417f0356049120543181d85aaac0fe6',
}
sys.path.insert(0, str(ROOT / 'scripts'))
from common import write_json, sha256

assert socket.gethostname() == 'autodl-container-be044ebe99-be706b14'
assert time.time() < DEADLINE - 3600
guard = (ROOT / 'logs/formal-eval-session.lock').open('a')
fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not OUT.exists(), 'refuse duplicate session or overwrite'
assert shutil.disk_usage('/root/autodl-tmp').free > 10 * 1024**3
for name, digest in EXPECTED.items():
    assert sha256(LOCKS / name / 'runtime_lock.json') == digest, name
gpu_processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
assert not gpu_processes, 'another GPU process is running'
import torch
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
assert torch.cuda.device_count() == 1
OUT.mkdir()
write_json(OUT / 'launch.json', {'host': socket.gethostname(), 'pid': os.getpid(),
    'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'deadline_utc': datetime.datetime.fromtimestamp(DEADLINE,datetime.timezone.utc).isoformat(),
    'locks': EXPECTED, 'gpu': torch.cuda.get_device_name(0),
    'disk_free_bytes': shutil.disk_usage('/root/autodl-tmp').free,
    'script_sha256': sha256(Path(__file__)), 'mode': 'fresh', 'attempts_per_model': 1})
preflight = ROOT / 'configs/week8_evaluation_preflight.json'
original = preflight.read_bytes()
(OUT / 'preflight_before.json').write_bytes(original)
env = os.environ.copy()
env.update(PATH='/root/autodl-tmp/conda/envs/llm_exp/bin:' + env.get('PATH',''),
           OMP_NUM_THREADS='14', MKL_NUM_THREADS='14', TOKENIZERS_PARALLELISM='false',
           PYTHONPATH=str(ROOT) + os.pathsep + env.get('PYTHONPATH',''))
env.pop('USE_TORCH', None)
child = None
completed = []
try:
    for name in EXPECTED:
        remaining = DEADLINE - time.time()
        assert remaining > 60, 'session deadline reached'
        lock = json.loads((LOCKS / name / 'runtime_lock.json').read_text())
        config = json.loads(original)
        config['runtime_lock'] = lock
        write_json(preflight, config)
        write_json(OUT / 'status.json', {'status': 'RUNNING', 'current_model': name, 'completed_models': completed})
        command = [sys.executable, str(ROOT / 'scripts/step3_eval.py'), '--mode', 'fresh',
                   '--model', lock['model_path'], '--judge-model', 'deepseek-flash',
                   '--judge-url', 'https://api.deepseek.com', '--output-dir', str(OUT / name)]
        write_json(OUT / (name + '_command.json'), command)
        with (OUT / (name + '.log')).open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            write_json(OUT / (name + '_process.json'), {'pid': child.pid, 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
            code = child.wait(timeout=remaining)
            child = None
        assert code == 0, f'{name} exit={code}; preserve evidence, no automatic retry'
        result = json.loads((OUT / name / 'status.json').read_text())
        assert result['status'] == 'completed' and result['fresh_model_inference'] is True
        completed.append(name)
    write_json(OUT / 'status.json', {'status': 'COMPLETED_PENDING_INDEPENDENT_REVIEW', 'completed_models': completed,
                                   'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
except BaseException as exc:
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
    write_json(OUT / 'status.json', {'status': 'FAILED_OR_TIMEOUT', 'completed_models': completed,
                                   'error': str(exc), 'stopped_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
    raise
finally:
    preflight.write_bytes(original)
    write_json(OUT / 'session_exit.json', {'preflight_restored': preflight.read_bytes() == original,
                                         'completed_models': completed})
