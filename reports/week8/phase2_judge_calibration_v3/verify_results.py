"""Revalidate original responses, frozen inputs, retry sequence and totals offline."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
import week8_deepseek_judge as judge
manifest=json.loads((OUT/'manifest.json').read_text())
for name,digest in manifest['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
cases=json.loads((OUT/'cases.json').read_text());rows=[];total={k:0 for k in ('prompt_tokens','completion_tokens','total_tokens')};all_attempts=0;invalid=0;results={}
for c in cases:
    path=OUT/c['id'];result=json.loads((path/'result.json').read_text());results[c['id']]=result
    assert 1<=result['attempts']<=2
    usage={k:0 for k in total}
    for n in range(1,result['attempts']+1):
        f=path/f'attempt-{n}.json';audit=json.loads(f.read_text());raw=audit['response'];all_attempts+=1
        assert hashlib.sha256(audit['request']['messages'][0]['content'].encode()).hexdigest()==manifest['prompt_sha256']
        payload=json.loads(audit['request']['messages'][1]['content']);assert payload['question']==c['question'] and payload['candidate_answer']==c['answer']
        try:scores=judge.validate_response(raw,judge.MODEL)
        except ValueError as exc:
            invalid+=1;scores=None
            assert audit['validation_error']==str(exc)
            if n<result['attempts']:assert isinstance(exc,judge.RetryableJudgmentError)
        if scores is not None:assert n==result['attempts'],'valid response must never trigger retry'
        for k in usage:usage[k]+=raw['usage'][k]
    assert result['scores']==scores and result['usage']==usage and result['usage_complete']
    checks={}
    if scores:
        for key,t in c['checks'].items():
            dim,op=key.rsplit('_',1);checks[key]=scores[dim]>=t if op=='min' else scores[dim]<=t
    expected={'checks':checks,'passed':scores is not None and all(checks.values())}
    assert expected==json.loads((OUT/(c['id']+'-checks.json')).read_text())
    for k in total:total[k]+=usage[k]
    rows.append({'id':c['id'],'passed':expected['passed'],'attempts':result['attempts'],'scores':scores})
repeats=[]
for c in cases:
    if 'repeat_of' in c:
        a=results[c['id']]['scores'];b=results[c['repeat_of']]['scores']
        delta={k:abs(a[k]-b[k]) for k in a}
        passed=delta['weighted_score']<=0.5 and all(v<=1 for k,v in delta.items() if k!='weighted_score')
        repeats.append({'id':c['id'],'deltas':delta,'passed':passed})
assert all_attempts<=12
summary={'status':'SMALL_BATCH_PASS' if all(r['passed'] for r in rows) and all(r['passed'] for r in repeats) else 'NOT_APPROVED','formal_scoring_approved':False,'cases':rows,'repeats':repeats,'api_calls':all_attempts,'invalid_attempts':invalid,'usage':total,'balance':json.loads((OUT/'latest_balance.json').read_text()),'limitations':['Six regression cases are not an independent grading accuracy estimate.','One repeated pair is insufficient to establish general repeatability.','Formal entrypoint and runtime lock are not yet integrated.']}
(OUT/'verification.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,ensure_ascii=False,indent=2))
