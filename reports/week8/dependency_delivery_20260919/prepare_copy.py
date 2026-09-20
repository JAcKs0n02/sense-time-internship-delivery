"""Continue from the reviewed, redacted candidate; preserve source and history."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[3]
AUDIT = Path(__file__).resolve().parent
SEED = ROOT / 'outputs/week8-clean-eval-release-20260919/repository'
DEST = ROOT / 'outputs/week8-submission-20260919/repository'
assert not DEST.exists(), 'Do not overwrite an existing delivery copy'
shutil.copytree(SEED, DEST)
# Inherited receipts remain explicitly historical, not a verdict on this copy.
meta = DEST / 'release_metadata'
meta.rename(DEST / 'previous_candidate_metadata')
meta.mkdir()
(DEST / 'previous_candidate_metadata').rename(meta / 'previous_candidate')
overlays = [
    'README.md', 'configs/README.md', 'configs/requirements-evaluation.txt',
    'configs/requirements-training-cuda.txt', 'run_pipeline.sh',
    'scripts/step2_train.sh', 'scripts/pipeline/step2_train.py',
    'scripts/pipeline/step3_eval.py', 'tests/test_week8_delivery_entrypoints.py',
    'docs/week8_teacher_scope_audit_20260919.md',
]
# Include successful scoped GPU evaluation and shutdown receipts, not transfers.
evidence = ROOT / 'reports/week8/delivery_eval_20260919'
overlays += [str(p.relative_to(ROOT)) for p in evidence.rglob('*')
             if p.is_file() and p.suffix in {'.py', '.json', '.jsonl', '.csv', '.log', '.out', '.txt', '.md', '.yaml', '.yml', '.sh'}
             and '__pycache__' not in p.parts]
rows = []
for name in sorted(set(overlays)):
    source, target = ROOT / name, DEST / name
    assert source.is_file() and not source.is_symlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    rows.append({'path': name, 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
(AUDIT / 'overlay.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
(DEST / 'RELEASE_VALIDATION.md').write_text(
    '# 本地交付副本\n\n本副本正在进行现行依赖与独立运行检查。旧验证记录位于'
    '`release_metadata/previous_candidate/`，不代表本副本已验收或已发布。\n'
    '原仓库及旧候选保持原样；最终报告同步、Week1–8导航整理和Git发布仍待后续完成。\n')
print(json.dumps({'copy': str(DEST), 'overlaid_files': len(rows)}, ensure_ascii=False))
