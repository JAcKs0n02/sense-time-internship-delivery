"""Download official immutable benchmark snapshots; no model or user data upload."""
from pathlib import Path
import concurrent.futures,hashlib,json,urllib.request,time
OUT=Path(__file__).resolve().parent
jobs=[]
for prefix,repo in [('ceval','ceval/ceval-exam'),('haonan-li','haonan-li/cmmlu')]:
 m=json.loads((OUT/(prefix+'_hub_metadata.json')).read_text()); revision=m['sha']
 for e in m['siblings']:
  name=e['rfilename']
  if name=='.gitattributes':continue
  url=f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}'
  jobs.append((repo,revision,name,url))
def download(job):
 repo,rev,name,url=job;p=OUT/'snapshots'/repo/name;p.parent.mkdir(parents=True,exist_ok=True)
 for attempt in range(3):
  try:
   with urllib.request.urlopen(url,timeout=45) as f:payload=f.read()
   p.write_bytes(payload)
   return {'repo':repo,'revision':rev,'remote_file':name,'url':url,'path':str(p.relative_to(OUT)),'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}
  except Exception:
   if attempt==2:raise
   time.sleep(1)
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:records=list(ex.map(download,jobs))
 (OUT/'download_manifest.json').write_text(json.dumps({'status':'OFFICIAL_DOWNLOADS_COMPLETE_NOT_EVALUATED','files':records},ensure_ascii=False,indent=2)+'\n')
 print('downloaded',len(records),'files',sum(x['bytes'] for x in records),'bytes')
