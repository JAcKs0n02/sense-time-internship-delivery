"""Reproduce the smoke input snapshot; does not launch or approve full training."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
from common import sha256,write_json
from week8_training_smoke import REQUIRED,DATA,verify_files,admit
out=ROOT/'reports/week8/phase2_training_smoke'
review=json.loads((ROOT/'reports/week8/phase2_final_release/review.json').read_text())
verify_files(ROOT,review['target_receipts_rechecked'])
checks=json.loads((ROOT/'reports/week8/phase1_task4/verification.json').read_text())
assert checks['blocking_issues']==0, checks['blocking_issues']
loader=json.loads((ROOT/'reports/week8/phase2_gpu_monitor/observed_receipt.json').read_text())
assert loader['loader_exit_code']==0
stats=json.loads((ROOT/DATA/'statistics.json').read_text())
for name,digest in stats['outputs'].items():
    assert sha256(ROOT/DATA/name)==digest,name
paths=set(REQUIRED)|{x['path'] for x in review['target_receipts_rechecked']}|{
    'reports/week8/phase1_task4/verification.json',
    'reports/week8/phase1_task4/final_checks.json',
    'reports/week8/phase1_task6/final_checks.json',
    'reports/week8/phase2_gpu_monitor/observed_receipt.json'}
runtime=json.loads((ROOT/'reports/week8/phase2_final_release/runtime_lock.json').read_text())
lock={'scope':'SFT3_MERGE_DPO3_ONLY','full_training_allowed':False,
      'base_model':runtime['model_path'],
      'files':[{'path':p,'sha256':sha256(ROOT/p)} for p in sorted(paths)]}
write_json(out/'smoke_lock.json',lock)
digest=sha256(out/'smoke_lock.json')
admit(ROOT,out/'smoke_lock.json',digest)
(out/'smoke_lock.sha256').write_text(digest+'\n')
print(f'PASS: {len(paths)} inputs pinned; smoke lock {digest}; no training started')
