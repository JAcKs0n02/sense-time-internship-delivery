"""Record the checked delivery state without changing original experiment files."""
from pathlib import Path
import hashlib
import json
import re
import shutil

ROOT = Path(__file__).resolve().parents[3]
AUDIT = Path(__file__).resolve().parent
DEST = ROOT/'outputs/week8-submission-20260919/repository'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
verification = json.loads((AUDIT/'verification.json').read_text())
assert verification['status'] == 'PASS_FRESH_MACOS_ENVIRONMENT_AND_INDEPENDENT_COPY'
for name in ['README.md', 'configs/README.md', 'docs/week8_teacher_scope_audit_20260919.md',
             'docs/week8_dependency_delivery.md']:
    target = DEST/name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT/name, target)
# Keep generated validation runs with the audit, not mixed into delivery inputs.
for name in ['delivery-fresh-quick', 'delivery-fresh-scores', 'delivery-fresh-plan',
             'delivery-fresh-cached-evaluation']:
    target = AUDIT/'validated_runs'/name
    target.parent.mkdir(parents=True, exist_ok=True)
    assert not target.exists()
    shutil.move(DEST/'logs'/name, target)
for p in AUDIT.iterdir():
    if p.is_file():
        target = DEST/'reports/week8/dependency_delivery_20260919'/p.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
shutil.copy2(AUDIT/'verification.json', DEST/'release_metadata/verification.json')
(DEST/'RELEASE_VALIDATION.md').write_text(
    '# 本地交付副本\n\n现行评估依赖已在空白macOS Python3.11环境安装，pip check通过。'
    '隔离原仓库与外网后，正式数据15个产物逐字节一致，评估配置、已有评分核验和6项入口测试通过。'
    '最新GPU评估的366个生成文件、305个依赖与20题评分已只读核验。\n\n'
    '没有新增GPU推理或API请求；不代表从零安装Linux CUDA通过。已完成评分的目录迁移后'
    '不能直接重跑付费入口，只读核验见[说明](docs/week8_dependency_delivery.md)。\n\n'
    '当前收据：release_metadata/verification.json。旧候选的收据在'
    'release_metadata/previous_candidate/，只描述旧版本。最终报告同步、全部周次导航整理'
    '与Git发布仍待完成；原仓库和检查点未删除。\n')
redactions = json.loads((ROOT/'reports/week8/clean_eval_release_20260919/redaction_mapping.json').read_text())
for row in redactions['files']:
    assert sha(ROOT/row['path']) == row['source_sha256'], 'original redaction source changed'
    assert sha(DEST/row['path']) == row['export_sha256'], 'redacted data changed'
patterns = [re.compile(r'(?<![A-Za-z0-9_-])(?:sk-[A-Za-z0-9]{20,}|AIza[A-Za-z0-9_-]{35}|AQ\.[A-Za-z0-9_-]{30,}|gh[pousr]_[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16})(?![A-Za-z0-9_-])'),
            re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')]
files = [p for p in sorted(DEST.rglob('*')) if p.is_file()]
for p in files:
    assert not p.is_symlink() and '.git' not in p.relative_to(DEST).parts
    assert not p.name.startswith('.env')
    try:
        content = p.read_text()
    except UnicodeError:
        original = ROOT/'outputs/week8-clean-eval-release-20260919/repository'/p.relative_to(DEST)
        assert original.is_file() and sha(p) == sha(original), 'new unreviewed binary'
        continue
    assert not any(pattern.search(content) for pattern in patterns), 'credential pattern in '+str(p.relative_to(DEST))
# Keep the explicit file allowlist approach; include additions/moved metadata.
ignore = DEST/'.gitignore'
known = ignore.read_text()
additions = []
for p in files:
    rule = '!/'+str(p.relative_to(DEST))
    if rule not in known.splitlines():
        for parent in reversed(p.relative_to(DEST).parents):
            if str(parent) != '.': additions.append('!/'+str(parent)+'/')
        additions.append(rule)
ignore.write_text(known+'\n# Current delivery additions\n'+'\n'.join(dict.fromkeys(additions))+'\n')
sum_path = DEST/'RELEASE_SHA256SUMS.txt'
files = [p for p in sorted(DEST.rglob('*')) if p.is_file() and p != sum_path]
sum_path.write_text(''.join(sha(p)+'  '+str(p.relative_to(DEST))+'\n' for p in files))
for line in sum_path.read_text().splitlines():
    digest, name = line.split('  ', 1)
    assert sha(DEST/name) == digest
result = {'status':'PASS_COPY_INTEGRITY', 'files':len(files)+1,
          'bytes':sum(p.stat().st_size for p in files)+sum_path.stat().st_size,
          'checksums_sha256':sha(sum_path), 'credential_pattern_hits':0,
          'original_sensitive_sources_unchanged':len(redactions['files']),
          'historical_training_requirements_unchanged':sha(ROOT/'configs/requirements-training.txt') == sha(DEST/'configs/requirements-training.txt'),
          'scan_limits':'Known patterns in text; binary documents unchanged from reviewed seed, not an exhaustive personal-data audit.'}
(AUDIT/'copy_integrity.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(result, ensure_ascii=False))
