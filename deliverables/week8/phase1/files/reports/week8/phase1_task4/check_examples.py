"""Execute only explicitly inspected, local, bounded examples from the 50-row review."""
from pathlib import Path
import json,re,subprocess,tempfile,hashlib,sqlite3,sys
O=Path(__file__).resolve().parent
rows={r['review_id']:r for r in json.loads((O/'selected_samples.json').read_text())}
checks=[]
def answers(rid):return [c['value'] for c in rows[rid]['record']['conversations'] if c['from']=='gpt']
def blocks(text):return re.findall(r'```(?:python|javascript|cpp|c\+\+)?\n(.*?)```',text,re.S|re.I)
def py(rid,code,assertions,case):
 result=subprocess.run([sys.executable,'-I','-c',code+'\n'+assertions],capture_output=True,text=True,timeout=10)
 if result.returncode:raise RuntimeError(rid+result.stderr)
 checks.append({'review_id':rid,'case':case,'status':'PASS','snippet_sha256':hashlib.sha256(code.encode()).hexdigest(),'stdout':result.stdout})
with tempfile.TemporaryDirectory() as td:
 d=Path(td)
 code=blocks(answers('R05')[1])[0];(d/'count.cpp').write_text(code)
 p=subprocess.run(['/usr/bin/clang++','-std=c++17',str(d/'count.cpp'),'-o',str(d/'count')],capture_output=True,text=True,timeout=30)
 if p.returncode:raise RuntimeError(p.stderr)
 p=subprocess.run([str(d/'count')],capture_output=True,text=True,timeout=5);assert p.returncode==0 and p.stdout=='1\n2\n3\n4\n5\n'
 checks.append({'review_id':'R05','case':'compile_and_count_1_to_5','status':'PASS','snippet_sha256':hashlib.sha256(code.encode()).hexdigest(),'stdout':p.stdout})
for n in [0,1]:
 code=blocks(answers('R10')[n])[0]
 test="assert [fibonacci(i) for i in range(10)] == [0,1,1,2,3,5,8,13,21,34]"
 if n==1:test+="\nassert generate_fibonacci_sequence(0)==[]\nassert generate_fibonacci_sequence(10)==[0,1,1,2,3,5,8,13,21,34]\nassert generate_fibonacci_sequence(5)==[0,1,1,2,3]"
 py('R10',code,test,'recursive_version_'+str(n+1))
code=answers('R28')[0].split('SELECT',1)[1];code='SELECT'+code
con=sqlite3.connect(':memory:');con.execute('CREATE TABLE customers(id integer,name text)');con.executemany('INSERT INTO customers VALUES (?,?)',[(1,'John'),(2,'Jane'),(3,'John')]);assert con.execute(code).fetchall()==[(1,'John'),(3,'John')];con.close();checks.append({'review_id':'R28','case':'sqlite_named_schema_fixture','status':'PASS','snippet_sha256':hashlib.sha256(code.encode()).hexdigest(),'result_ids':[1,3]})
py('R29',blocks(answers('R29')[0])[0],'assert nums == [7,4,3,2,1]','descending_sort')
code=blocks(answers('R30')[0])[0];p=subprocess.run(['/opt/homebrew/bin/node','-e',code+"\nif(JSON.stringify(arr)!=='[1,2,3]')throw Error('bad reversal');"],capture_output=True,text=True,timeout=5);assert p.returncode==0,p.stderr;checks.append({'review_id':'R30','case':'javascript_reverse_mutates_array','status':'PASS','snippet_sha256':hashlib.sha256(code.encode()).hexdigest(),'stdout':p.stdout})
py('R31',blocks(answers('R31')[0])[0],'assert sum==30','addition')
py('R32',blocks(answers('R32')[0])[0],'assert intersection == set()','disjoint_sets')
py('R33',blocks(answers('R33')[0])[0],"a=Node(1);b=Node(2);assert a.get_value()==1 and a.get_next_node() is None;a.set_next_node(b);assert a.get_next_node() is b",'possible_linked_node_example')
py('R34',answers('R34')[0],'assert x == 720','price_expression')
assert sum([8,7,19,33])==67 and '67' in answers('R35')[0];checks.append({'review_id':'R35','case':'independent_integer_sum','status':'PASS','result':67})
text=answers('R38')[0];code=text[text.index('# 首先'):text.index('print(centroid)')+len('print(centroid)')]
py('R38','points=[[1,2,3],[2,3,4],[3,4,5],[4,5,6]]\n'+code,'assert centroid == [2.5,3.5,4.5]','centroid_with_prompt_input_bound')
code=next(line for line in answers('R47')[0].splitlines() if line.startswith('result ='))
py('R47',code,'assert result == [1,2,3,4,5,6]','list_concatenation')
(O/'example_checks.json').write_text(json.dumps({'status':'PASS','checks':checks,'count':len(checks),'reviewed_examples_only':True,'production_framework_loader_test':False},ensure_ascii=False,indent=2)+'\n')
print('PASS',len(checks),'bounded example checks')
