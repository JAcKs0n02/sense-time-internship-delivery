from pathlib import Path
import json,hashlib,random
ROOT=Path(__file__).resolve().parents[3];out=Path(__file__).resolve().parent
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
source=ROOT/'logs/week8-protocol-data-20260914/run-a/train_alpaca.json';validation=source.with_name('validation_alpaca.json');stats=json.loads(source.with_name('statistics.json').read_text())
for f in [source,validation]:assert digest(f)==stats['outputs'][f.name]
a=json.loads(source.read_text());v=json.loads(validation.read_text());assert len(a)==1422 and len(v)==158
lineage=json.loads(source.with_name('lineage.json').read_text());assert digest(source.with_name('lineage.json'))==stats['outputs']['lineage.json']
ids={r['row_index']:r['sample_id'] for r in lineage if r['split']=='train'}
assert len(ids)==len(a)
a=[dict(r,sample_id=ids[i]) for i,r in enumerate(a)]
random.Random(42).shuffle(a);selected=a[:200];assert len({r['sample_id'] for r in selected})==200
key=lambda r:(r['instruction'],r.get('input',''))
assert not {key(r) for r in selected}&{key(r) for r in v}
rows=[{k:r[k] for k in ('sample_id','system','history','instruction','input') if k in r} for r in selected]
(out/'candidate_prompts.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
teacher=json.loads((ROOT/'Submission/Week4/Day21_Weekly_Report_and_Final_Model_Archive/Final_DPO_Model_Archive.json').read_text())['merged_dpo']
plan={'status':'LOCAL_PREPARATION_PASS_NOT_GPU_RELEASE','teacher':{'required_lineage':'Week4 reward_corrective_40step_merged','historical_path':teacher['path'],'historical_manifest_sha256':teacher['manifest_sha256'],'current_remote_identity_verified':False},'student':{'model_id':'Qwen/Qwen2.5-0.5B-Instruct','revision':None,'local_path':None,'current_identity_verified':False},'training':json.loads((ROOT/'configs/distillation.json').read_text()),'prompts':{'source':str(source.relative_to(ROOT)),'source_sha256':digest(source),'selection':'seed42 shuffle then first 200; prompt-only, original answers removed','candidate_path':str((out/'candidate_prompts.json').relative_to(ROOT)),'candidate_sha256':digest(out/'candidate_prompts.json'),'count':200,'source_records':1422,'validation_records_excluded':158,'validation_exact_prompt_overlap':0},'required_before_gpu':['Verify current Week4 teacher and student snapshot file identities on 320','Replace shorthand OpenCompass comparison with explicit model-bound CEval config and frozen gold validation','Bind generation source, config and model manifests; inspect complete teacher targets before training','Verify two-epoch trainer state, merged tensors and independent model reload','New bounded session deadline, logs and shutdown plan; old inference session remains closed'],'gpu_started':False,'api_calls':0}
(out/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
print('200 prompt-only candidates; source hash checked; zero exact overlap with 158 validation prompts')
