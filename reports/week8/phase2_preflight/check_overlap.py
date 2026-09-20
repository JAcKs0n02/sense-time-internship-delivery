"""Apply frozen all-turn lexical rules to pinned benchmark stems and options."""
from pathlib import Path
import hashlib,json,sys,unicodedata,time
R=Path(__file__).resolve().parents[3];O=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'scripts'))
from week8_data import match_text
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def norm(s):return ''.join(unicodedata.normalize('NFKC',s).split()).casefold()
def grams(s):return frozenset(s[i:i+5] for i in range(len(s)-4))
def match(a,b,ga,gb):
 if not a or not b:return None
 if a==b:return {'kind':'exact','score':1.0}
 if min(len(a),len(b))<32:return None
 if a in b or b in a:return {'kind':'containment','score':1.0}
 if min(len(ga),len(gb))<.8*max(len(ga),len(gb)):return None
 shared=len(ga&gb);union=len(ga)+len(gb)-shared
 if shared>=.8*union:return {'kind':'char5_jaccard','score':shared/union}
def main():
 lock=json.loads((O/'benchmark_input_lock.json').read_text());assert sha(O/'benchmark_records.json')==lock['benchmark_records_sha256']
 rows=json.loads((O/'benchmark_records.json').read_text());targets={}
 for row in rows:
  for field,text in [('question',row['question']),('question_with_options',row['question']+'\n'+'\n'.join(k+'. '+row['options'][k] for k in 'ABCD'))]:
   key=norm(text);targets.setdefault(key,[]).append({'id':row['id'],'role':row['role'],'field':field})
 prepared=[(t,grams(t),refs) for t,refs in targets.items()]
 data=R/'logs/week8-protocol-data-20260914/run-a';lineage=json.loads((data/'lineage.json').read_text());inputs=[]
 for split in ['train','validation']:
  values=json.loads((data/f'{split}_sharegpt.json').read_text());orig=sorted([x for x in lineage if x['split']==split],key=lambda x:x['row_index'])
  for row,origin in zip(values,orig):
   turns=([{'from':'system','value':row['system']}] if row['system'] else [])+row['conversations']
   for i,turn in enumerate(turns):inputs.append((norm(turn['value']),origin['sample_id'],split,i,turn['from']))
 # Verify the independent optimized predicate against the frozen implementation,
 # including exact, short non-match, containment, near-duplicate and unrelated inputs.
 pairs=[('短题','短题'),('短题','短题延长'),('a'*40,'prefix'+'a'*40),(''.join(chr(0x4e00+i) for i in range(100)),''.join(chr(0x4e00+i) for i in range(99))+'变'),('a'*40,'b'*40)]
 pairs += [(a[0],b[0]) for a in inputs[:20] for b in prepared[:200]]
 for a,b in pairs:assert match(a,b,grams(a),grams(b))==match_text(a,b)
 hits=[];start=time.monotonic()
 for index,(a,sid,split,turn,role) in enumerate(inputs):
  ga=grams(a)
  for b,gb,refs in prepared:
   result=match(a,b,ga,gb)
   if result:
    hits.append({'sample_id':sid,'split':split,'turn_index':turn,'role':role,'match':result,'benchmark_matches':refs,'normalized_source':a,'normalized_benchmark':b})
  if index%400==0:print('checked_turns',index,'hits',len(hits),flush=True)
 (O/'benchmark_overlap_hits.json').write_text(json.dumps(hits,ensure_ascii=False,indent=2)+'\n')
 receipt={'status':'PASS_NO_LEXICAL_HITS' if not hits else 'BLOCKED_BENCHMARK_OVERLAP_REQUIRES_REVIEW','source_data_dir':str(data.relative_to(R)),'source_hashes':{p.name:sha(p) for p in data.glob('*.json')},'benchmark_records_sha256':sha(O/'benchmark_records.json'),'benchmark_records':len(rows),'benchmark_unique_comparison_texts':len(prepared),'source_records':len(lineage),'all_source_turns_checked':len(inputs),'predicate_comparison_cases':len(pairs),'matched_pairs':len(hits),'matched_sample_ids':sorted({h['sample_id'] for h in hits}),'checked_roles':['scored','fewshot','unscored_protected'],'comparison_fields':['question','question_with_options'],'elapsed_seconds':round(time.monotonic()-start,2),'semantic_leakage_proven_absent':False,'data_mutated':False,'training_allowed':False}
 (O/'benchmark_overlap_verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k!='source_hashes'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
