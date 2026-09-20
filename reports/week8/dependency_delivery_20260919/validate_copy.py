"""Bounded CPU checks in a newly installed environment, with source/network guards."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
AUDIT = Path(__file__).resolve().parent
DEST = ROOT / 'outputs/week8-submission-20260919/repository'
PYTHON = '/tmp/week8-delivery-fresh-20260919/bin/python'
home = Path('/tmp/week8-delivery-empty-home-20260919')
home.mkdir(exist_ok=True)
env = {'PATH': str(Path(PYTHON).parent) + ':/usr/bin:/bin',
       'HOME': str(home), 'PYTHONPATH': '/tmp/week8-release-guard-20260919',
       'RELEASE_CANDIDATE_ROOT': str(DEST), 'HF_HUB_OFFLINE': '1',
       'TRANSFORMERS_OFFLINE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
       'PYTHONNOUSERSITE': '1', 'PIPELINE_PYTHON': PYTHON,
       'TMPDIR': '/tmp', 'LANG': 'en_US.UTF-8', 'TOKENIZERS_PARALLELISM': 'false',
       'EVAL_MODEL_PATH': '/models/final_dpo'}
probe = """import socket,sys
from pathlib import Path
assert sys.prefix != sys.base_prefix
assert Path('run_pipeline.sh').is_file()
try: Path('/Users/yifanren/Documents/商汤科技实习/README.md').read_bytes()
except PermissionError: pass
else: raise AssertionError('source guard inactive')
s=socket.socket()
try: s.connect(('192.0.2.1',443))
except PermissionError: pass
else: raise AssertionError('network guard inactive')
finally: s.close()
print('SOURCE_AND_NETWORK_GUARDS_ACTIVE')
"""
commands = [
    ('guard', [PYTHON, '-c', probe]),
    ('pip_check', [PYTHON, '-m', 'pip', 'check']),
    ('imports', [PYTHON, '-c', 'import torch,transformers,accelerate,datasets,opencompass,mmengine; from opencompass.models import HuggingFaceCausalLM; print("EVALUATION_IMPORTS_OK; cuda_available=",torch.cuda.is_available())']),
    ('quick', ['bash', 'run_pipeline.sh', '--quick', '--run-dir', 'logs/delivery-fresh-quick']),
    ('scores', ['bash', 'run_pipeline.sh', '--score-only', 'all', '--run-dir', 'logs/delivery-fresh-scores']),
    ('data_eval_plan', ['bash', 'run_pipeline.sh', '--skip-train', '--dry-run', '--eval-limit-per-subject', '1', '--run-dir', 'logs/delivery-fresh-plan']),
    ('entry_tests', [PYTHON, '-m', 'pytest', '-q', 'tests/test_week8_delivery_entrypoints.py', '-p', 'no:cacheprovider']),
]
runs = []
for name, command in commands:
    start = time.monotonic()
    with (AUDIT / (name + '.log')).open('w') as stream:
        result = subprocess.run(command, cwd=DEST, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, timeout=600)
    runs.append({'name': name, 'command': command, 'returncode': result.returncode,
                 'elapsed_seconds': round(time.monotonic()-start, 2)})
    (AUDIT/'validation_runs.json').write_text(json.dumps(runs, indent=2)+'\n')
    print(name, result.returncode, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
reference = json.loads((ROOT/'reports/week8/release_candidate_20260919/formal_reproduction_comparison.json').read_text())
for row in reference['files']:
    path = DEST/'logs/delivery-fresh-plan/data'/row['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], row['path']
(AUDIT/'formal_reproduction_comparison.json').write_text(json.dumps(reference, indent=2)+'\n')
freeze = subprocess.check_output([PYTHON, '-m', 'pip', 'freeze'], text=True, cwd=DEST, env=env)
(AUDIT/'environment_freeze_macos.txt').write_text(freeze)
print('FORMAL_DATA_15_FILES_BYTE_IDENTICAL', flush=True)
