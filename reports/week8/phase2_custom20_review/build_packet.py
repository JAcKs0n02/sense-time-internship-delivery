"""Reproduce a read-only evidence review and blank human review packet."""
from pathlib import Path
import sys,json,csv,hashlib,ast
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'deliverables/week3/day14'))
sys.path.insert(0,str(ROOT/'deliverables/week3/day14/source/scripts'))
from eval_harness import load_questions,automatic_check
from prepare_blind_review import prepare_blind_records,write_reviewer_csv
from run_code_checks import extract_python_code
OUT=Path(__file__).resolve().parent
questions=load_questions(ROOT/'deliverables/week3/day14/source/data/evaluation_questions.json')
rows=[];responses=[];sources={}
for model in ['original_base','final_sft','final_dpo']:
    directory=ROOT/f'reports/week8/phase2_gemini_score_{model}'
    verification=json.loads((directory/'verification.json').read_text())
    assert verification['status']=='VERIFIED_COMPLETE'
    for name,h in verification['files'].items():
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==h
    sources[model]=hashlib.sha256((directory/'verification.json').read_bytes()).hexdigest()
    plan=json.loads((directory/'plan.json').read_text())
    for q,req in zip(questions,plan['requests']):
        inp=json.loads(req['body']['contents'][0]['parts'][0]['text'])
        assert inp['question']==q and req['question_id']==q['id']
        answer=inp['candidate_answer']
        evidence=automatic_check(q,answer)
        flags=[]
        if q['automatic_check']['type']=='numeric' and not evidence['passed']:flags.append('numeric_final_mismatch')
        if q['automatic_check']['type']=='required_terms' and not evidence['passed']:flags.append('keyword_miss_requires_semantic_review')
        if q['id']=='MATH-05':flags.append('ambiguous_question_do_not_rank_from_this_item')
        if q['automatic_check']['type']=='python_tests':
            code=extract_python_code(answer)
            if code is not None:
                try:ast.parse(code);evidence['syntax_status']='valid_not_execution'
                except SyntaxError as e:
                    evidence['syntax_status']='invalid';evidence['syntax_error']={'type':type(e).__name__,'line':e.lineno,'message':e.msg};flags.append('invalid_python_syntax')
            else:flags.append('no_python_function_extracted')
            if evidence.get('code_execution_status')=='disabled_no_sandbox':flags.append('code_behavior_not_executed')
        result=json.loads((directory/'judge'/q['id']/'result.json').read_text())
        rows.append({'model':model,'question_id':q['id'],'check_type':q['automatic_check']['type'],'automatic_evidence':evidence,'review_flags':flags,'ai_score':result['weighted_score']})
        responses.append({'candidate_id':model,'question_id':q['id'],'messages':q['messages'],'raw_response':answer,'status':'completed'})
assert len(rows)==60 and len({(x['model'],x['question_id']) for x in rows})==60
(OUT/'automatic_review.json').write_text(json.dumps({'ranking_input':False,'source_receipts':sources,'records':rows},ensure_ascii=False,indent=2)+'\n')
with (OUT/'private/responses.jsonl').open('x') as f:
    for row in responses:f.write(json.dumps(row,ensure_ascii=False)+'\n')
blind,mapping=prepare_blind_records(responses,seed=20260918,question_metadata={q['id']:q for q in questions})
for reviewer in ['reviewer_1','reviewer_2']:write_reviewer_csv(OUT/f'reviewers/{reviewer}.csv',blind,reviewer)
(OUT/'private/blind_mapping.json').write_text(json.dumps(mapping,ensure_ascii=False,indent=2)+'\n')
(OUT/'reviewers/rubric.json').write_bytes((ROOT/'deliverables/week3/day14/source/data/evaluation_rubric.json').read_bytes())
with (OUT/'review_flags.csv').open('x',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=['model','question_id','check_type','ai_score','review_flags','automatic_evidence']);w.writeheader()
    for row in rows:w.writerow({**row,'review_flags':';'.join(row['review_flags']),'automatic_evidence':json.dumps(row['automatic_evidence'],ensure_ascii=False)})
for model in sources:
    subset=[r for r in rows if r['model']==model]
    print(model,'numeric',sum(r['automatic_evidence'].get('passed') is True for r in subset if r['check_type']=='numeric'),'/7','syntax_invalid',[r['question_id'] for r in subset if 'invalid_python_syntax' in r['review_flags']])
