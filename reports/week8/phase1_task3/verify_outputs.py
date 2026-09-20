"""Independent exported-data audit. No imports from the preparation implementation."""
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import unicodedata
ROOT=Path(__file__).resolve().parents[3]
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(t): return ''.join(unicodedata.normalize('NFKC',t).split()).casefold()
@lru_cache(maxsize=20000)
def grams(t): return set(t[n:n+5] for n in range(len(t)-4))
def similar(a,b):
    if not a or not b: return False
    if a==b: return True
    if len(a)<32 or len(b)<32: return False
    if a in b or b in a: return True
    x,y=grams(a),grams(b)
    if min(len(x),len(y)) < .8*max(len(x),len(y)): return False
    return len(x&y)/len(x|y)>=.8

def verify(folder,repeat):
    stats=read(folder/'statistics.json');protocol=read(folder/'protocol.json')
    for name,expected in stats['outputs'].items():
        if sha(folder/name)!=expected: raise ValueError('output hash mismatch: '+name)
    for e in read(folder/'input_receipt.json'):
        if sha(ROOT/e['path'])!=e['sha256']: raise ValueError('source hash changed: '+e['path'])
    if sha(ROOT/'configs/week8_data_protocol.json')!=stats['protocol_sha256']: raise ValueError('protocol differs')
    raw=[json.loads(x) for x in (ROOT/protocol['source']['path']).read_text().splitlines()]
    source_lineage=[json.loads(x) for x in (ROOT/protocol['lineage']['path']).read_text().splitlines()]
    original={x['sample_id']:row for x,row in zip(source_lineage,raw)}
    aligned=read(folder/'lineage.json');excluded=read(folder/'exclusions.json')
    if len({r['sample_id'] for r in aligned+excluded})!=len(raw) or len(aligned)+len(excluded)!=len(raw): raise ValueError('source partition incomplete or duplicate')
    sets={};texts={};multi={};payload={}
    for split in ['train','validation']:
        sg=read(folder/f'{split}_sharegpt.json');al=read(folder/f'{split}_alpaca.json')
        records=sorted([r for r in aligned if r['split']==split],key=lambda r:r['row_index'])
        if len(sg)!=len(al) or len(records)!=len(sg): raise ValueError('format row counts mismatch')
        if [r['row_index'] for r in records]!=list(range(len(sg))): raise ValueError('invalid row indices')
        sets[split]={r['sample_id'] for r in records};texts[split]=[];multi[split]=0
        for s,a,l in zip(sg,al,records):
            body=s['conversations'];src=original[l['sample_id']]
            if body!=src['conversations'] or s['system']!=src.get('system',''): raise ValueError('source content changed')
            rebuilt=[]
            for u,v in a['history']: rebuilt.extend([{'from':'human','value':u},{'from':'gpt','value':v}])
            rebuilt.extend([{'from':'human','value':a['instruction']},{'from':'gpt','value':a['output']}])
            if rebuilt!=body or a['system']!=s['system'] or a['input']!='': raise ValueError('Alpaca lost content')
            if len(body)%2 or any(c['from']!=('human' if n%2==0 else 'gpt') or not c['value'].strip() for n,c in enumerate(body)): raise ValueError('invalid roles/content')
            multi[split]+=len(body)>2
            texts[split].extend(norm(c['value']) for c in body)
            if s['system']: texts[split].append(norm(s['system']))
            payload[l['sample_id']]=s
    if sets['train']&sets['validation']: raise ValueError('IDs cross split')
    if any(similar(a,b) for a in texts['train'] for b in texts['validation']): raise ValueError('lexical relationship crosses split')
    question_texts=[];coverage=[]
    for e in protocol['protected_files']:
        if e['role']=='custom20_fixed_rubric': continue
        f=ROOT/e['path'];v=[json.loads(x) for x in f.read_text().splitlines()] if f.suffix=='.jsonl' else read(f)
        if isinstance(v,dict):
            keys=[k for k in ['items','questions','prompts','records'] if isinstance(v.get(k),list)]
            if len(keys)!=1: raise ValueError('unknown eval schema')
            v=v[keys[0]]
        if len(v)!=e['records']: raise ValueError('eval source count mismatch')
        before=len(question_texts)
        for item in v:
            if 'messages' in item: qs=[m['content'] for m in item['messages'] if m['role']=='user']
            elif 'conversations' in item: qs=[m['value'] for m in item['conversations'] if m['from']=='human']
            elif 'text' in item: qs=[item['text']]
            else: raise ValueError('unknown question schema')
            if not qs: raise ValueError('empty question')
            question_texts.extend(norm(t) for t in qs)
        coverage.append({'path':e['path'],'records':len(v),'questions':len(question_texts)-before})
    question_texts=set(question_texts)
    if any(similar(a,b) for a in texts['train']+texts['validation'] for b in question_texts): raise ValueError('protected question contamination')
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained((ROOT/protocol['tokenizer_files'][0]['path']).parent,local_files_only=True,use_fast=True)
    lengths=[]
    for row in aligned:
        s=payload[row['sample_id']]
        msgs=([{'role':'system','content':s['system']}] if s['system'] else [])+[{'role':('user' if c['from']=='human' else 'assistant'),'content':c['value']} for c in s['conversations']]
        length=len(tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=False))
        if length!=row['tokens'] or length>2048: raise ValueError('token length mismatch')
        lengths.append(length)
    names=sorted(p.name for p in folder.glob('*.json'))
    if names!=sorted(p.name for p in repeat.glob('*.json')): raise ValueError('repeat file list mismatch')
    hashes={name:sha(folder/name) for name in names}
    if any(sha(repeat/name)!=value for name,value in hashes.items()): raise ValueError('nondeterministic outputs')
    if Counter(r['split'] for r in aligned)!=Counter({'train':stats['train_count'],'validation':stats['validation_count']}): raise ValueError('reported counts incorrect')
    return {'status':'PASS_TASK3_INDEPENDENT_EXPORT_AUDIT','train':len(sets['train']),'validation':len(sets['validation']),'preserved_records':len(aligned),'exclusions':len(excluded),'multiturn':multi,'protected_unique_questions':len(question_texts),'cross_split_lexical_hits':0,'protected_lexical_hits':0,'max_tokens':max(lengths),'matching_repeat_files':len(names),'output_hashes':hashes,'coverage':coverage,'actual_lf_loader_verified':False,'semantic_leakage_proven_absent':False,'benchmark_snapshots_locked':False}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('repeat',type=Path);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
    result=verify(a.folder,a.repeat)
    a.receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['output_hashes','coverage']},ensure_ascii=False,indent=2))
