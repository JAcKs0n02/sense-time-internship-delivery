"""Archive the accepted phase1 evidence; never changes data or training gates."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[3]
DEST = ROOT / 'deliverables/week8/phase1'
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    if DEST.exists(): raise ValueError('snapshot exists; do not overwrite a frozen release')
    protocol = read(ROOT/'configs/week8_data_protocol.json')
    paths = {}
    def add(p, role):
        p = Path(p)
        if not p.is_absolute(): p = ROOT/p
        if not p.is_file(): raise FileNotFoundError(p)
        rel = str(p.relative_to(ROOT))
        paths.setdefault(rel, set()).add(role)
    for e in [protocol['source'], protocol['lineage'], protocol['historical_group_links']] + protocol['protected_files'] + protocol['tokenizer_files']:
        assert sha(ROOT/e['path']) == e['sha256'], e['path']
        add(e['path'], e.get('role', 'frozen_tokenizer'))
    for p in (ROOT/'logs/week8-protocol-data-20260914/run-a').glob('*.json'): add(p, 'accepted_data_and_audit')
    for directory in ['phase1', 'phase1_task3', 'phase1_task4', 'phase1_task5', 'phase1_task6']:
        for p in (ROOT/'reports/week8'/directory).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.name not in ['final_checks.json']:
                add(p, 'task_evidence')
    for p in (ROOT/'logs/week8-real-loader-20260915/task6-recheck').glob('*'):
        if p.is_file(): add(p, 'fresh_real_loader_evidence')
    for name in ['scripts/common.py', 'scripts/week8_data.py', 'scripts/step1_data_prep.py', 'scripts/train_pipeline.py', 'scripts/check_week8_loader.py', 'run_pipeline.sh', 'tests/test_week8_pipeline.py', 'tests/test_week8_protocol_data.py', 'tests/test_week8_loader.py', 'configs/week8_data_protocol.json', 'configs/requirements-data.txt', 'configs/requirements-loader-cpu.txt', 'deliverables/week1/day3/source/config/config.json', 'deliverables/week3/day11/configs/experiments/epoch-e5.yaml', 'docs/week8_data_input_decision.md', 'docs/week8_data_protocol.md', 'docs/week8_phase1_data_readiness.md', 'docs/week8_phase1_task3_result.md', 'docs/week8_phase1_task4_result.md', 'docs/week8_phase1_task5_result.md', 'docs/week8_phase1_task6_result.md']:
        add(name, 'implementation_configuration_or_document')
    for e in read(ROOT/'reports/week8/phase1_task6/next_stage_plan.json')['dpo']['candidate_files']:
        assert sha(ROOT/e['path']) == e['sha256']
        add(e['path'], 'next_stage_candidate_not_accepted_by_sft_audit')
    entries=[]
    for source, roles in sorted(paths.items()):
        src=ROOT/source; target=DEST/'files'/source
        h=sha(src); target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src,target)
        assert sha(target)==h and sha(src)==h, source
        entries.append({'source_path':source,'archive_path':str(target.relative_to(DEST)),'sha256':h,'bytes':target.stat().st_size,'roles':sorted(roles)})
    manifest={'schema_version':1,'status':'PHASE1_DATA_PREPARATION_PASS_TRAINING_BLOCKED','date':'2026-09-15','protocol_sha256':sha(ROOT/'configs/week8_data_protocol.json'),'file_count':len(entries),'total_bytes':sum(e['bytes'] for e in entries),'files':entries,'scope':'local repository evidence snapshot; no full model weights, no training release, no external publication','portability':'source scripts resolve repository paths; this is not a standalone runnable training environment','upstream_dependencies':'historical parent/review dependencies remain referenced by original path and SHA-256 in files/reports/week8/phase1/input_inventory.json'}
    (DEST/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    checks=[f"{e['sha256']}  {e['archive_path']}" for e in entries]
    checks.append(f"{sha(DEST/'manifest.json')}  manifest.json")
    (DEST/'SHA256SUMS.txt').write_text('\n'.join(checks)+'\n')
    print(json.dumps({k:manifest[k] for k in ['status','file_count','total_bytes']},indent=2))

if __name__=='__main__': main()
