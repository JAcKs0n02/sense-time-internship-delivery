"""Run frozen assertions inside resource-bounded Docker containers, without host mounts."""
from pathlib import Path
import sys,json,subprocess,uuid,hashlib
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'deliverables/week3/day14/source/scripts'))
from run_code_checks import extract_python_code,build_test_program
questions=json.loads((ROOT/'deliverables/week3/day14/source/data/evaluation_questions.json').read_text())['questions'];qs={q['id']:q for q in questions}
image=subprocess.check_output(['docker','image','inspect','python:3.11-slim-bookworm','--format','{{.Id}}'],text=True).strip()
results=[]
for line in (OUT/'private/responses.jsonl').read_text().splitlines():
 row=json.loads(line);q=qs[row['question_id']]
 if q['automatic_check']['type']!='python_tests':continue
 code=extract_python_code(row['raw_response']);name='week8-check-'+uuid.uuid4().hex[:12]
 program=build_test_program(code or '',q['automatic_check']['assertions'])
 cmd=['docker','run','--rm','--name',name,'--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--pids-limit','32','--memory','128m','--cpus','1','--user','65534:65534','-i',image,'python','-I','-B','-']
 try:
  proc=subprocess.run(cmd,input=program,text=True,capture_output=True,timeout=12)
  result={'status':'passed' if proc.returncode==0 and 'DAY14_TESTS_PASSED' in proc.stdout else 'failed','exit_code':proc.returncode,'stdout':proc.stdout[-4000:],'stderr':proc.stderr[-4000:]}
 except subprocess.TimeoutExpired:
  subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=10)
  result={'status':'timeout'}
 results.append({'model':row['candidate_id'],'question_id':q['id'],'answer_sha256':hashlib.sha256(row['raw_response'].encode()).hexdigest(),'program_sha256':hashlib.sha256(program.encode()).hexdigest(),**result})
 print(row['candidate_id'],q['id'],result['status'],flush=True)
with (OUT/'code_execution.json').open('x') as f:json.dump({'image_id':image,'network':'none','host_mounts':[],'read_only':True,'memory':'128m','cpus':1,'timeout_seconds':12,'ranking_input':False,'records':results},f,ensure_ascii=False,indent=2);f.write('\n')
