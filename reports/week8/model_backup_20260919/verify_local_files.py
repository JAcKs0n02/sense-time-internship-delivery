from pathlib import Path
import os,json,hashlib,struct,datetime,time
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
root=Path('/Users/yifanren/Documents/商汤科技实习');out=root/'reports/week8/model_backup_20260919';dest=root/'models/backups/week8-20260919'
plan=json.loads((root/'reports/week8/release_audit_20260919/expected_model_backups.json').read_text());results=[]
for m in plan['models']:
 folder=dest/m['role'];verified=[];keys=set()
 assert {p.name for p in folder.iterdir() if p.is_file()}=={f['path'] for f in m['expected_files']}
 source=json.loads((root/m['source_evidence']).read_text())
 for f in m['expected_files']:
  p=folder/f['path']; assert p.is_file() and p.stat().st_size==f['bytes'],str(p)
  h=hashlib.sha256()
  with p.open('rb') as stream:
   for b in iter(lambda:stream.read(8*1024**2),b''):h.update(b)
  assert h.hexdigest()==f['sha256'],str(p)
  verified.append(f)
  if p.suffix=='.safetensors':
   with p.open('rb') as stream:
    n=struct.unpack('<Q',stream.read(8))[0];assert 0<n<50*1024**2;header=json.loads(stream.read(n))
   data_size=p.stat().st_size-8-n
   for name,value in header.items():
    if name=='__metadata__':continue
    a,b=value['data_offsets'];assert 0<=a<=b<=data_size;assert name not in keys;keys.add(name)
   idx=folder/'model.safetensors.index.json'
   if idx.exists():
    mapping=json.loads(idx.read_text())['weight_map']
    for name in header:
     if name!='__metadata__':assert mapping[name]==p.name
    for name,shard in mapping.items():
     if shard==p.name:assert name in header
 idx=folder/'model.safetensors.index.json'
 if idx.exists():assert keys==set(json.loads(idx.read_text())['weight_map'])
 assert len(keys)==source.get('tensors',source.get('merged_tensors'))
 results.append({'role':m['role'],'files':len(verified),'bytes':sum(f['bytes'] for f in verified),'tensor_count':len(keys),'sha256_and_index':'PASS'})
(out/'independent_file_verification.json').write_text(json.dumps({'status':'PASS','verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'models':results},indent=2)+'\n')
print('INDEPENDENT_FILE_HASH_AND_INDEX_PASS',flush=True)
