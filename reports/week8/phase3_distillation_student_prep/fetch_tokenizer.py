from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import urllib.request,json,hashlib
m=json.load(open('reports/week8/phase3_distillation_target/retrieved/student_manifest.json'))
out=Path('/tmp/week8-student-tokenizer-7ae557');out.mkdir(exist_ok=True)
files=[f for f in m['files'] if f['path'] in ['config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt']]
def get(f):
 url=f"https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/{m['revision']}/{f['path']}"
 p=out/f['path']
 if not p.exists():
  with urllib.request.urlopen(url,timeout=60) as res:b=res.read()
  assert hashlib.sha256(b).hexdigest()==f['sha256'];p.write_bytes(b)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256']
 return {'path':str(p),'sha256':f['sha256'],'source_url':url}
with ThreadPoolExecutor(max_workers=5) as ex:verified=list(ex.map(get,files))
receipt={'status':'TOKENIZER_FILES_MATCH_PINNED_STUDENT_MANIFEST','revision':m['revision'],'model_id':m['model_id'],'tokenizer_dir':str(out),'student_manifest_sha256':hashlib.sha256(Path('reports/week8/phase3_distillation_target/retrieved/student_manifest.json').read_bytes()).hexdigest(),'files':verified}
Path('reports/week8/phase3_distillation_student_prep/tokenizer_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('Verified',len(verified),'files',out)
