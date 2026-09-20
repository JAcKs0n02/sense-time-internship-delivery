"""Fresh evaluation must bind new evidence and never silently rebill."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))


def module():
    import week8_clean_eval
    return week8_clean_eval


@pytest.fixture
def clean_binding():
    """Build config for this checkout; never reuse another machine's paths."""
    from build_week8_formal_eval_config import build_config
    from mmengine.config import Config
    w=module()
    with tempfile.TemporaryDirectory(prefix='clean-eval-config-test-',dir=ROOT/'logs') as temp:
        directory=Path(temp)
        config,binding=build_config(ROOT,str(ROOT/'models/backups/week8-20260919/week8_final_dpo'))
        path=directory/'opencompass.py';Config(config).dump(str(path))
        binding['config']={'path':str(path.relative_to(ROOT)),'sha256':w.sha(path)}
        target=directory/'binding.json';target.write_text(json.dumps(binding))
        yield target


def fixture_answers():
    spec = json.loads((ROOT/'deliverables/week3/day14/source/data/evaluation_questions.json').read_text())
    return [{'id': q['id'], 'answer': 'new answer '+q['id'], 'finish_reason': 'eos',
             'generated_tokens': 30} for q in spec['questions']]


def test_new_plan_uses_fresh_answers_and_native_gemini_protocol():
    w = module()
    plan = w.bind_score_plan(fixture_answers(), 'a'*64, 'b'*64)
    assert plan['model'] == 'gemini-3.1-pro-preview'
    assert len(plan['requests']) == plan['max_calls'] == 20
    for row, answer in zip(plan['requests'], fixture_answers()):
        body = row['body']
        assert 'messages' not in body
        payload = json.loads(body['contents'][0]['parts'][0]['text'])
        assert payload['candidate_answer'] == answer['answer']
    assert plan['scope'] == 'clean_eval_custom20'
    other = w.bind_score_plan(fixture_answers(), 'c'*64, 'b'*64)
    assert plan['identity'] != other['identity']


@pytest.mark.parametrize('change', ['duplicate', 'missing', 'truncated', 'empty', 'long', 'order'])
def test_bad_answers_rejected_before_judge(change):
    w = module(); answers = fixture_answers()
    if change == 'duplicate': answers[-1] = answers[0]
    if change == 'missing': answers.pop()
    if change == 'truncated': answers[0]['finish_reason'] = 'length'
    if change == 'empty': answers[0]['answer'] = ' '
    if change == 'long': answers[0]['generated_tokens'] = 513
    if change == 'order': answers.reverse()
    with pytest.raises(ValueError): w.bind_score_plan(answers, 'a'*64, 'b'*64)


def test_candidate_cannot_launch_and_has_no_side_effects(tmp_path):
    w = module()
    with patch.object(w, 'run_child', side_effect=AssertionError('must not run')):
        with pytest.raises(ValueError, match='not released'):
            w.generate({'release_state':'CANDIDATE_NOT_RELEASED'}, 'a'*64, tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_changed_saved_plan_rejected_before_paid_runner(tmp_path):
    w=module(); path=tmp_path/'plan.json'; path.write_text('{}')
    with patch.object(w.scoring, 'run', side_effect=AssertionError('paid call')):
        with pytest.raises(ValueError, match='plan hash'):
            w.load_bound_plan(path, '0'*64)


def test_new_scoring_recomputes_csv_and_completed_rerun_never_bills(tmp_path):
    w = module(); plan = w.bind_score_plan(fixture_answers(), 'a'*64, 'b'*64)
    raw=json.loads((ROOT/'reports/week8/phase2_gemini_calibration_v2/EQ-TEXT-response.raw').read_text())
    calls=[]
    def fake(body): calls.append(body); return copy.deepcopy(raw)
    out=tmp_path/'out'
    w.scoring.run(plan, out, tmp_path/'claims', fake)
    result=w.verify_completed(plan,out)
    assert result['questions']==20
    w.scoring.run(plan, out, tmp_path/'claims', fake)
    assert len(calls)==20


def test_unknown_paid_request_stops_without_retry(tmp_path):
    w=module(); plan=w.bind_score_plan(fixture_answers(),'a'*64,'b'*64); calls=[]
    def fail(body): calls.append(body); raise TimeoutError('unknown paid attempt')
    with pytest.raises(TimeoutError): w.scoring.run(plan,tmp_path/'out',tmp_path/'claims',fail)
    with pytest.raises(ValueError): w.scoring.run(plan,tmp_path/'out',tmp_path/'claims',fail)
    assert len(calls)==1


def test_pipeline_rejects_incompatible_clean_flags_before_data(tmp_path):
    proc=subprocess.run(['bash',str(ROOT/'run_pipeline.sh'),'--clean-eval','generate','--quick',
        '--eval-plan','unused','--eval-plan-sha256','0'*64,'--run-dir',str(tmp_path/'out')],capture_output=True,text=True)
    assert proc.returncode!=0
    assert not (tmp_path/'out').exists()


def test_child_timeout_terminates_process_group(tmp_path):
    w=module()
    with pytest.raises(subprocess.TimeoutExpired):
        w.run_child([sys.executable,'-c','import time; time.sleep(60)'],tmp_path/'child.log',0.05)


def test_missing_benchmark_never_becomes_zero_score(tmp_path):
    w=module()
    lock={'records':{'path':'reports/week8/phase2_preflight/benchmark_records.json'}}
    with pytest.raises(ValueError,match='coverage'):
        w.ev.validate_benchmarks(tmp_path,lock,'test-model')


def test_candidate_missing_dependencies_is_rejected(clean_binding):
    w=module()
    plan=w.candidate(clean_binding)
    del plan['dependencies']['scripts/week8_clean_eval.py']
    with pytest.raises(ValueError,match='required dependency'):
        w.validate_plan(plan)


def test_source_manifest_cannot_omit_or_add_artifacts(tmp_path):
    w=module(); (tmp_path/'answers.jsonl').write_text('original')
    receipt={'files':{'answers.jsonl':w.sha(tmp_path/'answers.jsonl')}}
    (tmp_path/'generation_receipt.json').write_text('{}')
    w.verify_source_files(tmp_path,receipt)
    (tmp_path/'unexpected.json').write_text('{}')
    with pytest.raises(ValueError,match='coverage'):
        w.verify_source_files(tmp_path,receipt)


def test_stale_target_audit_cannot_bind_new_candidate():
    w=module()
    with pytest.raises(ValueError):
        w.validate_target_audit({'target_audit':None})


def test_timeout_kills_grandchild_that_ignores_sigterm(tmp_path):
    import os, signal, time
    w=module(); marker=tmp_path/'pid'
    child_code='import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)'
    parent_code=("import subprocess,sys,time,pathlib; p=subprocess.Popen([sys.executable,'-c',"+
                 repr(child_code)+"]); pathlib.Path("+repr(str(marker))+").write_text(str(p.pid)); time.sleep(60)")
    pid=None
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            w.run_child([sys.executable,'-c',parent_code],tmp_path/'tree.log',0.5)
        pid=int(marker.read_text())
        for _ in range(50):
            state=subprocess.run(['ps','-o','stat=','-p',str(pid)],capture_output=True,text=True).stdout.strip()
            if not state or state.startswith('Z'): break
            time.sleep(.02)
        assert not state or state.startswith('Z'), 'grandchild survived timeout'
    finally:
        if pid:
            try: os.kill(pid,signal.SIGKILL)
            except ProcessLookupError: pass


def test_missing_gpu_release_blocks_paid_call(tmp_path):
    w=module()
    with pytest.raises(ValueError,match='GPU release'):
        w.validate_gpu_release(None,'a'*64)


def test_score_produces_csv_and_rejects_tamper_without_calls(tmp_path):
    w=module(); plan=w.bind_score_plan(fixture_answers(),'a'*64,'b'*64)
    plan.update(generation_source='source',gpu_release={'path':'release.json','sha256':'c'*64})
    source=tmp_path/'source'; source.mkdir()
    (source/'benchmarks.json').write_text(json.dumps([
        {'model':'model','dataset':'ceval','score':50,'questions':1346},
        {'model':'model','dataset':'cmmlu','score':60,'questions':11582}]))
    raw=json.loads((ROOT/'reports/week8/phase2_gemini_calibration_v2/EQ-TEXT-response.raw').read_text())
    output=tmp_path/'logs/clean-eval-scores/run';calls=[]
    def fake(body): calls.append(body);return copy.deepcopy(raw)
    w.scoring.run(plan,output,tmp_path/'claims',fake)
    with patch.object(w,'ROOT',tmp_path),patch.object(w,'prepare_score',return_value=plan):
        w.score(plan,output)
        import csv
        rows=list(csv.DictReader((output/'pipeline_summary.csv').open()))
        assert [x['dataset'] for x in rows]==['ceval','cmmlu','custom20']
        assert len(calls)==20
        (output/'pipeline_summary.csv').write_text('wrong\nvalue\n')
        with pytest.raises(ValueError,match='CSV differs'):w.score(plan,output)


def test_different_plans_cannot_acquire_instance_guard_concurrently(tmp_path):
    w=module()
    with patch.object(w,'ROOT',tmp_path),w.gpu_guard():
        with pytest.raises(BlockingIOError):
            with w.gpu_guard(): pass


def test_gpu_release_wrong_generation_or_running_state_rejected(tmp_path):
    w=module(); (tmp_path/'evidence.txt').write_text('test evidence, not actual platform state')
    release={'instance':'320','state':'running','generation_receipt_sha256':'a'*64,
             'observed_utc':'2026-09-19T00:00:00+00:00',
             'evidence':{'path':'evidence.txt','sha256':w.sha(tmp_path/'evidence.txt')}}
    path=tmp_path/'release.json'; path.write_text(json.dumps(release))
    with patch.object(w,'ROOT',tmp_path),pytest.raises(ValueError,match='state or generation'):
        w.validate_gpu_release({'path':'release.json','sha256':w.sha(path)},'a'*64)
    release['state']='stopped';path.write_text(json.dumps(release))
    with patch.object(w,'ROOT',tmp_path),pytest.raises(ValueError,match='state or generation'):
        w.validate_gpu_release({'path':'release.json','sha256':w.sha(path)},'b'*64)


def test_prepare_score_replays_all_source_checks_without_model_or_network(tmp_path,clean_binding):
    """Synthetic generation uses archived benchmark files; never a fresh GPU receipt."""
    import shutil
    w=module()
    # Source lives inside ROOT to exercise the actual portable source locator.
    with tempfile.TemporaryDirectory(prefix='clean-eval-test-',dir=ROOT/'logs') as temp:
        source=Path(temp)/'source';source.mkdir()
        plan=w.candidate(clean_binding)
        unsigned_hash=hashlib.sha256(w.scoring.canonical(plan)).hexdigest()
        environment=Path(temp)/'env.txt'; environment.write_text('synthetic test only')
        audit_path=Path(temp)/'target.json'
        audit_path.write_text(json.dumps({'status':'CLEAN_EVAL_TARGET_PASS','host':w.HOST,
            'candidate_sha256':unsigned_hash,'runtime_versions':w.RUNTIME_VERSIONS,
            'deadline_utc':'2026-09-19T05:00:00+00:00','platform_shutdown_utc':'2026-09-19T05:05:00+00:00',
            'environment_freeze':{'path':str(environment.relative_to(ROOT)),'sha256':w.sha(environment)},**dict.fromkeys(['pip_check','cuda_bf16','model_forward','prompt_check','shutdown_confirmed'],True)}))
        plan.update(release_state='TARGET_AUDITED_RELEASED',deadline_utc='2026-09-19T05:00:00+00:00',
            platform_shutdown_utc='2026-09-19T05:05:00+00:00',target_audit={
            'path':str(audit_path.relative_to(ROOT)),'sha256':w.sha(audit_path)})
        (source/'generation_plan.json').write_text(json.dumps(plan))
        data=ROOT/'reports/week8/release_candidate_20260919/validated_runs/release-formal-data-001'
        shutil.copytree(data,source/'data')
        oc=ROOT/'reports/week8/phase2_trained_generation/retrieved/logs/trained-generation-20260917/final_dpo/opencompass'
        for p in oc.glob('*/results/*/*.json'):
            dest=source/'opencompass'/p.relative_to(oc);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
        (source/'benchmarks.json').write_text(json.dumps(w.ev.validate_benchmarks(source/'opencompass',plan,plan['model_path'])))
        (source/'answers.jsonl').write_text('\n'.join(json.dumps(x) for x in fixture_answers())+'\n')
        (source/'launch.json').write_text('{}')
        (source/'pip_check.log').write_text('synthetic test only')
        (source/'environment_freeze.txt').write_text('synthetic test only')
        receipt={'status':'GENERATED_PENDING_LOCAL_SCORING','judge_calls':0,
                 'generation_plan_sha256':w.sha(source/'generation_plan.json'),
                 'files':{str(p.relative_to(source)):w.sha(p) for p in source.rglob('*') if p.is_file()}}
        (source/'generation_receipt.json').write_text(json.dumps(receipt))
        receipt_hash=w.sha(source/'generation_receipt.json')
        with patch.object(w,'validate_gpu_release'):
            new=w.prepare_score(source,receipt_hash,{'path':'test-only','sha256':'0'*64})
            assert new['answers_sha256']==w.sha(source/'answers.jsonl')
            assert len(new['requests'])==20
            original=json.loads((source/'benchmarks.json').read_text());original[0]['score']=0
            (source/'benchmarks.json').write_text(json.dumps(original))
            with pytest.raises(ValueError,match='artifact changed'):
                w.prepare_score(source,receipt_hash,{'path':'test-only','sha256':'0'*64})


def test_paid_execution_uses_bounded_supervised_worker(tmp_path):
    w=module(); plan=w.bind_score_plan(fixture_answers(),'a'*64,'b'*64)
    plan.update(generation_source='source',gpu_release={'path':'release','sha256':'c'*64})
    with patch.object(w,'ROOT',tmp_path),patch.object(w,'prepare_score',return_value=plan),\
         patch.object(w,'run_child',side_effect=subprocess.TimeoutExpired('score',7200)) as child,\
         patch.object(w.scoring,'run',side_effect=AssertionError('paid API in parent')):
        with pytest.raises(subprocess.TimeoutExpired):
            w.score(plan,tmp_path/'logs/clean-eval-scores/run',execute=True)
    assert child.call_args.args[2]==7200
    assert 'score-worker' in child.call_args.args[0]


def test_terminating_supervisor_stops_worker_group(tmp_path):
    import os, signal, time
    marker=tmp_path/'worker.pid'
    worker_code=('import os,pathlib,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); '
                 'pathlib.Path('+repr(str(marker))+').write_text(str(os.getpid())); time.sleep(60)')
    parent_code=('import sys,pathlib;sys.path.insert(0,'+repr(str(ROOT/'scripts'))+');'
                 'from week8_clean_eval import run_child; run_child([sys.executable,"-c",'+repr(worker_code)+'],pathlib.Path('+repr(str(tmp_path/'worker.log'))+'),60)')
    parent=subprocess.Popen([sys.executable,'-c',parent_code],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    worker=None
    try:
        for _ in range(150):
            if marker.exists(): break
            time.sleep(.02)
        assert marker.exists()
        worker=int(marker.read_text());parent.terminate();parent.wait(timeout=25)
        for _ in range(50):
            state=subprocess.run(['ps','-o','stat=','-p',str(worker)],capture_output=True,text=True).stdout.strip()
            if not state or state.startswith('Z'):break
            time.sleep(.02)
        assert not state or state.startswith('Z'),'paid worker survives supervisor SIGTERM'
    finally:
        if parent.poll() is None:parent.kill();parent.wait()
        if worker:
            try:os.killpg(worker,signal.SIGKILL)
            except ProcessLookupError:pass


def test_supervised_child_preserves_inherited_pythonpath(tmp_path):
    import os
    w=module(); inherited=str(tmp_path/'site-guard')
    with patch.dict(os.environ,{'PYTHONPATH':inherited}):
        w.run_child([sys.executable,'-c','import os; print(os.environ["PYTHONPATH"])'],tmp_path/'env.log',10)
    assert inherited in (tmp_path/'env.log').read_text().strip().split(os.pathsep)
