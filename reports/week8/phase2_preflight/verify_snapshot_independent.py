from pathlib import Path
import csv,hashlib,io,json,zipfile,collections
import pyarrow.parquet as pq
R=Path.cwd();O=R/'reports/week8/phase2_preflight';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_text())
for e in read(O/'download_manifest.json')['files']:assert sha(O/e['path'])==e['sha256']
expected={}
for p in (O/'snapshots/ceval/ceval-exam').glob('*/*.parquet'):
 split=p.name.split('-')[0]
 for i,x in enumerate(pq.read_table(p).to_pylist()):expected[f'ceval:{p.parent.name}:{split}:{i}']=(x['question'],{k:x[k] for k in 'ABCD'},x['answer'])
 target=O/'local_data/ceval'/split/(p.parent.name+'_'+split+'.csv')
 converted=list(csv.DictReader(target.open(newline='')))
 original=pq.read_table(p).to_pylist()
 assert len(converted)==len(original)
 for a,b in zip(converted,original):assert a=={k:str(v) for k,v in b.items()}
with zipfile.ZipFile(O/'snapshots/haonan-li/cmmlu/cmmlu_v1_0_1.zip') as z:
 for n in z.namelist():
  if not n.endswith('.csv'):continue
  assert (O/'local_data/cmmlu'/n).read_bytes()==z.read(n)
  p=Path(n)
  for i,x in enumerate(csv.DictReader(io.StringIO(z.read(n).decode('utf-8-sig')))):expected[f'cmmlu:{p.stem}:{p.parts[0]}:{i}']=(x['Question'],{k:x[k] for k in 'ABCD'},x['Answer'])
actual=read(O/'benchmark_records.json');assert len(actual)==len(expected)==25865
for x in actual:assert expected.pop(x['id'])==(x['question'],x['options'],x['answer'])
assert not expected
for e in read(O/'local_data_manifest.json'):assert sha(O/e['path'])==e['sha256']
overlap=read(O/'benchmark_overlap_verification.json');assert overlap['matched_pairs']==0
for n,h in overlap['source_hashes'].items():assert sha(R/overlap['source_data_dir']/n)==h
for e in read(R/'deliverables/week8/phase1/manifest.json')['files']:assert sha(R/'deliverables/week8/phase1'/e['archive_path'])==e['sha256']
v={'status':'PASS_INDEPENDENT_BENCHMARK_SNAPSHOT_AUDIT','normalized_records_verified':len(actual),'official_download_files_verified':len(read(O/'download_manifest.json')['files']),'local_data_files_verified':len(read(O/'local_data_manifest.json')),'scored_questions':sum(x['role']=='scored' for x in actual),'fewshot_questions':sum(x['role']=='fewshot' for x in actual),'unscored_ceval_test_questions':sum(x['role']=='unscored_protected' for x in actual),'source_and_phase1_archive_unchanged':True,'overlap_pairs':overlap['matched_pairs'],'actual_opencompass_runtime_executed':False}
(O/'independent_verification.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');print(json.dumps(v,indent=2))
