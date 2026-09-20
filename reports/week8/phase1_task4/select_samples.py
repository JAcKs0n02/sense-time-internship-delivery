"""Deterministic risk-stratified content review selection; no source mutation."""
from pathlib import Path
import json,hashlib,re
R=Path(__file__).resolve().parents[3];D=R/'logs/week8-protocol-data-20260914/run-a';O=Path(__file__).resolve().parent
load=lambda p:json.loads(p.read_text())
rows=load(D/'lineage.json');data={s:load(D/f'{s}_sharegpt.json') for s in ['train','validation']}
for r in rows:
 r['record']=data[r['split']][r['row_index']];r['turns']=sum(c['from']=='gpt' for c in r['record']['conversations'])
 r['text']='\n'.join(c['value'] for c in r['record']['conversations'])
 r['tags']=[]
 if r['turns']>1:r['tags'].append('multiturn')
 if re.search(r'```|\bdef\b|\bSELECT\b|\bfunction\b|\bclass\b|编写.*代码|Python|JavaScript|C\+\+',r['text'],re.I):r['tags'].append('code_candidate')
 if re.search(r'计算|方程|分数|概率|数学|函数|求解|三角|面积|百分比',r['text']):r['tags'].append('math_candidate')
selected={}
def take(pool,n,reason):
 count=0
 for r in pool:
  if r['sample_id'] in selected:continue
  selected[r['sample_id']]={k:v for k,v in r.items() if k!='text'};selected[r['sample_id']]['selection_reason']=reason;count+=1
  if count==n:break
order=lambda rr:sorted(rr,key=lambda r:hashlib.sha256(('week8-content-review:42:'+r['sample_id']).encode()).hexdigest())
take(order([r for r in rows if r['turns']>1]),13,'all_multiturn')
take(sorted(rows,key=lambda r:(-r['tokens'],r['sample_id'])),8,'longest_remaining')
take(sorted(rows,key=lambda r:(r['tokens'],r['sample_id'])),6,'shortest_remaining')
take(order([r for r in rows if 'code_candidate'in r['tags']]),6,'code_stratum')
take(order([r for r in rows if 'math_candidate'in r['tags']]),6,'math_stratum')
for split in ['validation','train']:
 for source in ['alpaca_gpt4_zh','coig_pc','sharegpt_zh']:
  need=max(0,3-sum(r['split']==split and r['source']==source for r in selected.values()))
  if need:take(order([r for r in rows if r['split']==split and r['source']==source]),min(need,50-len(selected)),f'coverage_{split}_{source}')
if len(selected)<50:take(order(rows),50-len(selected),'deterministic_fill')
assert len(selected)==50
result=list(selected.values())
for n,r in enumerate(result,1):r['review_id']=f'R{n:02d}'
(O/'selected_samples.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
(O/'selection_manifest.json').write_text(json.dumps({'method':'all13multiturn_then8longest6shortest6code6math_then_source_split_coverage_then_hash_fill','sample_count':50,'pool':1580,'representative_random_sample':False,'input_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [D/'lineage.json',D/'train_sharegpt.json',D/'validation_sharegpt.json']},'counts':{k:sum(k in r['tags'] for r in result) for k in ['multiturn','code_candidate','math_candidate']},'split_counts':{s:sum(r['split']==s for r in result) for s in ['train','validation']}},ensure_ascii=False,indent=2)+'\n')
print((O/'selection_manifest.json').read_text())
