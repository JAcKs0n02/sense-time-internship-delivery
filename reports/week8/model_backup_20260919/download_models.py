from pathlib import Path
import urllib.request,urllib.parse,json,hashlib,time,os,concurrent.futures,datetime
os.umask(0o077)
ROOT=Path('/Users/yifanren/Documents/商汤科技实习');DEST=ROOT/'models/backups/week8-20260919';OUT=ROOT/'reports/week8/model_backup_20260919';DEST.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
BASE='https://a1087293-be99-be706b14.nmb2.seetacloud.com:8443/jupyter'
TOKEN=os.environ['WEEK8_BACKUP_JUPYTER_TOKEN']
plan=json.loads((ROOT/'reports/week8/release_audit_20260919/expected_model_backups.json').read_text());start=time.time()
def req(url,extra=None):return urllib.request.Request(url,headers={'Authorization':'token '+TOKEN,**(extra or {})})
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
remote=[];jobs=[]
for m in plan['models']:
 rel=m['remote_source'].removeprefix('/root/')
 with urllib.request.urlopen(req(BASE+'/api/contents/'+urllib.parse.quote(rel)+'?content=1'),timeout=30) as r:listing=json.load(r)
 observed={x['name']:x for x in listing['content']}
 for f in m['expected_files']:
  assert f['path'] in observed and observed[f['path']]['size']==f['bytes'],(m['role'],f['path'])
  jobs.append((m['role'],rel,f))
 remote.append({'role':m['role'],'source':m['remote_source'],'file_count':len(observed),'expected_files_present_sizes_match':True})
(OUT/'remote_inventory.json').write_text(json.dumps(remote,indent=2)+'\n')
def download(job):
 role,rel,f=job;folder=DEST/role;folder.mkdir(exist_ok=True);p=folder/f['path'];partial=p.with_name(p.name+'.partial')
 if p.exists():
  assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256'];return {'role':role,**f,'state':'VERIFIED_EXISTING'}
 offset=partial.stat().st_size if partial.exists() else 0
 url=BASE+'/files/'+urllib.parse.quote(rel+'/'+f['path'])
 def fetch_range(bounds):
  begin,end=bounds
  for attempt in range(4):
   try:
    with urllib.request.urlopen(req(url,{'Range':f'bytes={begin}-{end}'}),timeout=45) as response:
     assert (response.status==206 and response.headers.get('Content-Range')==f'bytes {begin}-{end}/{f["bytes"]}') or (response.status==200 and begin==0 and end+1==f['bytes']), 'invalid range response'
     data=response.read(end-begin+2)
     assert len(data)==end-begin+1,'incomplete range body'
    return begin,end,data
   except Exception:
    if attempt==3:raise
    time.sleep(2)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ranges:
  while offset < f['bytes']:
   bounds=[(begin,min(begin+8*1024**2,f['bytes'])-1) for begin in range(offset,min(offset+16*1024**2,f['bytes']),8*1024**2)]
   for begin,end,data in ranges.map(fetch_range,bounds):
    assert begin==offset,'non-contiguous range'
    with partial.open('ab') as target:target.write(data)
    offset=end+1
 assert partial.stat().st_size==f['bytes'],(role,f['path'],'size')
 assert sha(partial)==f['sha256'],(role,f['path'],'hash')
 partial.rename(p)
 result={'role':role,**f,'state':'DOWNLOADED_SHA256_VERIFIED'}
 print(json.dumps({'file':role+'/'+f['path'],'bytes':f['bytes'],'elapsed_seconds':round(time.time()-start)}),flush=True)
 return result
results=[]
try:
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
  for r in pool.map(download,jobs):
   results.append(r);(OUT/'progress.json').write_text(json.dumps({'completed_files':results,'status':'COPYING'},indent=2)+'\n')
 (OUT/'transfer.json').write_text(json.dumps({'status':'ALL_FILES_DOWNLOADED_HASH_VERIFIED','files':results,'total_bytes':sum(r['bytes'] for r in results),'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
 print('ALL_FILES_DOWNLOADED_HASH_VERIFIED',flush=True)
except Exception as e:
 (OUT/'transfer_error.json').write_text(json.dumps({'type':type(e).__name__,'message':str(e).replace(TOKEN,'[REDACTED]'),'status':'INCOMPLETE_KEEP_PARTIAL_FILES'},indent=2)+'\n');raise
