"""Validate immutable calibration and final answer bindings in the submission."""
import json
from pathlib import Path
from common import ROOT
from week8_score_recovery_audit import digest, read, require
from week8_gemini_judge import MODEL, validate_response, usage as gemini_usage


def validate_calibration(calibration, repo=ROOT):
    calibration, repo = Path(calibration), Path(repo)
    review = read(calibration/'verification.json')
    require(review['status']=='SMALL_BATCH_PASS_NOT_FORMAL_RELEASE' and review['valid_responses']==17 and not review['failures'], 'calibration failed')
    for name,h in review['files'].items():
        require(Path(name).name==name and digest(calibration/name)==h, 'calibration changed')
    for name,h in read(calibration/'manifest.json').items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and digest(repo/name)==h, 'frozen dependency changed')
    profile=read(calibration/'judge_profile_candidate.json')
    require(profile['verification_sha256']==digest(calibration/'verification.json'), 'receipt mismatch')
    require(profile['model']==MODEL, 'wrong profile model')
    template={'systemInstruction':profile['systemInstruction'],'generationConfig':profile['generationConfig']}
    cases=read(calibration/'cases.json');require(len(cases)==17, 'case count')
    scores={};totals={}
    for c in cases:
        body=read(calibration/(c['id']+'-request.json'))
        require({k:body[k] for k in template}==template, 'calibration protocol mismatch')
        inp=json.loads(body['contents'][0]['parts'][0]['text'])
        require(inp=={'question':c['question'],'candidate_answer':c['answer'],'rubric':profile['rubric']}, 'calibration input mismatch')
        raw=read(calibration/(c['id']+'-response.raw'));score=validate_response(raw)
        for k,t in c['checks'].items():
            d,op=k.rsplit('_',1);require(score[d]>=t if op=='min' else score[d]<=t, 'score check failed')
        scores[c['id']]=score
        for k,v in gemini_usage(raw).items():totals[k]=totals.get(k,0)+v
    require(totals==review['usage'], 'usage mismatch')
    for p in read(calibration/'plan.json')['pair_checks']:
        for k in scores[p['left']]:require(abs(scores[p['left']][k]-scores[p['right']][k])<=(p['max_weighted_delta'] if k=='weighted_score' else p['max_dimension_delta'])+1e-9, 'pair failed')
    return profile


def validate_plan(model, plan):
    """Recheck original score inputs without requiring old failed-job directories."""
    calibration = ROOT/plan['calibration']
    require(digest(calibration/'verification.json') == plan['calibration_receipt_sha256'], 'calibration receipt changed')
    require(digest(calibration/'judge_profile_candidate.json') == plan['profile_sha256'], 'calibration profile changed')
    profile = validate_calibration(calibration)
    for name, expected in plan['dependencies'].items():
        require(digest(ROOT/name) == expected, 'frozen scoring dependency changed')
    answers_path = ROOT/'data/evaluation/answers'/f'{model}.jsonl'
    require(digest(answers_path) == plan['answers_sha256'], 'final answers changed')
    answers = [json.loads(line) for line in answers_path.read_text().splitlines()]
    questions = read(ROOT/'deliverables/week3/day14/source/data/evaluation_questions.json')['questions']
    require(len(answers) == len(questions) == len(plan['requests']) == 20, 'question count changed')
    template = {key:profile[key] for key in ('systemInstruction', 'generationConfig')}
    for question, answer, request in zip(questions, answers, plan['requests']):
        require(question['id'] == answer['id'] == request['question_id'], 'question order changed')
        body = request['body']
        require({key:body[key] for key in template} == template, 'scoring template changed')
        payload = json.loads(body['contents'][0]['parts'][0]['text'])
        require(payload == {'question':question, 'candidate_answer':answer['answer'], 'rubric':profile['rubric']}, 'answer or rubric binding changed')
    return plan
