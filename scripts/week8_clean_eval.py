"""New data + inference + Gemini evaluation; historical releases stay immutable.

GPU generation and local scoring are separate resumable stages. A candidate is
never a runtime release. All paid work uses the existing durable claim runner.
"""
import argparse
import copy
from contextlib import contextmanager, ExitStack
import datetime as dt
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import step3_eval as ev
import week8_gemini_score_only as scoring
from week8_generation_only import sha, read, check
from week8_pipeline_score import load_release, verify_completed

ROOT = Path(__file__).resolve().parents[1]
HOST = 'autodl-container-be044ebe99-be706b14'
RUNTIME_VERSIONS = {'torch':'2.5.1+cu121','transformers':'4.50.0','opencompass':'0.5.3','mmengine':'0.10.7'}
QUESTIONS = 'deliverables/week3/day14/source/data/evaluation_questions.json'
DATA_PROTOCOL = 'configs/week8_data_protocol.json'
BACKUP = 'reports/week8/model_backup_20260919/backup_files_manifest.json'
DATA_EXPECTED = 'reports/week8/release_candidate_20260919/formal_reproduction_comparison.json'


def safe_path(root, relative):
    p = Path(relative)
    check(not p.is_absolute() and '..' not in p.parts, 'unsafe artifact path')
    result = Path(root)/p
    check(result.resolve().is_relative_to(Path(root).resolve()), 'artifact escapes root')
    check(not result.is_symlink(), 'symlink artifact')
    return result


def load_bound_plan(path, expected):
    check(sha(path) == expected, 'plan hash mismatch')
    return read(path)


def save(path, value):
    scoring.save(Path(path), value)


def candidate(binding_path):
    binding = read(binding_path)
    check(binding['verified'] is False, 'expected unreleased config binding')
    # Rebuild the expected configuration, so a different prompt/model cannot
    # acquire legitimacy merely by having its hash copied into a candidate.
    from build_week8_formal_eval_config import build_config
    from mmengine.config import Config
    expected, _ = build_config(ROOT, binding['model_path'])
    check(sha(safe_path(ROOT,binding['config']['path'])) == binding['config']['sha256'], 'config hash mismatch')
    check(Config.fromfile(str(ROOT/binding['config']['path'])).to_dict() == expected,
          'config differs from reviewed builder')
    artifacts = binding['dependencies'] + [binding['config'], binding['records']]
    dependencies = {a['path']: a['sha256'] for a in artifacts}
    names = [str(p.relative_to(ROOT)) for p in (ROOT/'scripts').glob('*.py')]
    names += ['run_pipeline.sh', DATA_PROTOCOL, BACKUP, DATA_EXPECTED,
              'deliverables/week2/day7/clean_pipeline.py', QUESTIONS,
              'configs/requirements-training.txt']
    def collect(value):
        if isinstance(value, dict):
            if 'path' in value and 'sha256' in value:
                dependencies[value['path']] = value['sha256']
            for item in value.values(): collect(item)
        elif isinstance(value, list):
            for item in value: collect(item)
    protocol = read(ROOT/DATA_PROTOCOL)
    collect(protocol)
    dependencies[protocol['rules_document']] = protocol['rules_sha256']
    for name in names: dependencies[name] = sha(ROOT/name)
    for name, h in dependencies.items(): check(sha(safe_path(ROOT, name)) == h, 'input changed: '+name)
    files = [dict(path=Path(r['path']).name, bytes=r['bytes'], sha256=r['sha256'])
             for r in read(ROOT/BACKUP)['files'] if '/week8_final_dpo/' in r['path']]
    check(len(files) == 14, 'final DPO manifest incomplete')
    return {'schema': 1, 'mode': 'clean_eval_generation', 'release_state': 'CANDIDATE_NOT_RELEASED',
            'target_root':str(ROOT), 'target_host': HOST, 'model_name': 'final_dpo', 'model_path': binding['model_path'],
            'model_files': files, 'config': binding['config'], 'records': binding['records'],
            'dependencies': dependencies, 'benchmark_timeout_seconds': 14400,
            'maximum_session_seconds': 21600, 'judge_calls': 0,
            'target_audit': None, 'deadline_utc': None, 'platform_shutdown_utc': None}


def validate_plan(plan, *, model_files=False):
    check(plan['schema'] == 1 and plan['mode'] == 'clean_eval_generation', 'unexpected plan mode')
    check(plan['model_name'] == 'final_dpo' and plan['target_host'] == HOST and plan['judge_calls'] == 0,
          'unexpected model/host/judge policy')
    check(plan['benchmark_timeout_seconds'] == 14400 and plan['maximum_session_seconds'] == 21600,
          'unexpected time policy')
    check(Path(plan['model_path']).is_absolute(), 'model path must be absolute')
    required = {str(p.relative_to(ROOT)) for p in (ROOT/'scripts').glob('*.py')}
    required.update({'run_pipeline.sh',DATA_PROTOCOL,BACKUP,DATA_EXPECTED,QUESTIONS,
                     'deliverables/week2/day7/clean_pipeline.py','configs/requirements-training.txt'})
    check(required <= set(plan['dependencies']), 'required dependency missing')
    for name, h in plan['dependencies'].items():
        check(sha(safe_path(ROOT, name)) == h, 'dependency changed: '+name)
    for key in ('config','records'):
        a=plan[key]
        check(plan['dependencies'].get(a['path']) == a['sha256'], 'unbound '+key)
    from build_week8_formal_eval_config import build_config
    from mmengine.config import Config
    expected_config, expected_binding = build_config(ROOT,plan['model_path'])
    for a in expected_binding['dependencies'] + [expected_binding['records']]:
        check(plan['dependencies'].get(a['path']) == a['sha256'], 'required dependency changed')
    actual = Config.fromfile(str(safe_path(ROOT,plan['config']['path']))).to_dict()
    for dataset in actual['datasets']:
        benchmark = dataset['abbr'].split('-')[0]
        suffix = 'reports/week8/phase2_preflight/local_data/'+benchmark
        check(dataset['path'] == str(Path(plan['target_root'])/suffix), 'dataset path not bound')
        dataset['path'] = str(ROOT/suffix)
    suffix = 'logs/week8-real-loader-20260915/attempt-02/tokenizer'
    check(actual['models'][0]['tokenizer_path'] == str(Path(plan['target_root'])/suffix), 'tokenizer path not bound')
    actual['models'][0]['tokenizer_path'] = str(ROOT/suffix)
    check(actual == expected_config, 'config/model differs from reviewed protocol')
    for key in ('config', 'records'):
        a = plan[key]
        check(plan['dependencies'].get(a['path']) == a['sha256'], 'unbound '+key)
    expected = [dict(path=Path(r['path']).name, bytes=r['bytes'], sha256=r['sha256'])
                for r in read(ROOT/BACKUP)['files'] if '/week8_final_dpo/' in r['path']]
    check(plan['model_files'] == expected, 'model manifest differs from backup')
    if model_files:
        model = Path(plan['model_path'])
        check(not model.is_symlink() and not any(p.is_symlink() for p in model.rglob('*')), 'model symlink')
        check({str(p.relative_to(model)) for p in model.rglob('*') if p.is_file()} ==
              {r['path'] for r in expected}, 'model file coverage')
        for row in expected:
            p = model/row['path']
            check(p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], 'model bytes changed')
    return plan


def require_release(plan):
    check(plan.get('release_state') == 'TARGET_AUDITED_RELEASED', 'generation not released')
    validate_plan(plan, model_files=True)
    check(socket.gethostname() == HOST and str(ROOT) == plan['target_root'], 'wrong instance or repository root')
    deadline = dt.datetime.fromisoformat(plan['deadline_utc']).timestamp()
    shutdown = dt.datetime.fromisoformat(plan['platform_shutdown_utc']).timestamp()
    check(60 < deadline-time.time() <= 21600 and 0 < shutdown-deadline <= 600, 'invalid or expired window')
    validate_target_audit(plan)
    return deadline


def validate_target_audit(plan):
    a = plan.get('target_audit')
    check(isinstance(a, dict) and sha(safe_path(ROOT, a['path'])) == a['sha256'], 'target audit missing/changed')
    audit = read(ROOT/a['path'])
    check(audit['status'] == 'CLEAN_EVAL_TARGET_PASS' and audit['host'] == HOST,
          'target audit not passed')
    # The release review records the candidate with mutable release fields reset.
    unsigned = copy.deepcopy(plan)
    unsigned.update(release_state='CANDIDATE_NOT_RELEASED', target_audit=None,
                    deadline_utc=None, platform_shutdown_utc=None)
    check(audit['candidate_sha256'] == hashlib.sha256(scoring.canonical(unsigned)).hexdigest(), 'audit binds another candidate')
    check(audit.get('deadline_utc') == plan['deadline_utc'] and audit.get('platform_shutdown_utc') == plan['platform_shutdown_utc']
          and isinstance(plan['deadline_utc'],str) and isinstance(plan['platform_shutdown_utc'],str),
          'target audit does not bind shutdown window')
    check(all(audit.get(k) is True for k in ('pip_check', 'cuda_bf16', 'model_forward', 'prompt_check', 'shutdown_confirmed')),
          'target checks incomplete')

    check(audit.get('runtime_versions') == RUNTIME_VERSIONS, 'target runtime versions missing/changed')
    environment = audit.get('environment_freeze')
    check(isinstance(environment,dict) and sha(safe_path(ROOT,environment['path'])) == environment['sha256'],
          'target environment freeze missing/changed')


def stop_process_group(child):
    if child is None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=20)
    except subprocess.TimeoutExpired:
        pass
    # The leader may already be gone while OpenCompass workers remain alive.
    # Always address the group, including after a successful leader exit.
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait()


@contextmanager
def gpu_guard():
    # The repository lock also coordinates with the historical session runner;
    # /tmp prevents two clean-eval copies on the same instance from racing.
    with ExitStack() as stack:
        for path in (Path('/tmp/week8-clean-eval-gpu.lock'), ROOT/'logs/formal-eval-session.lock'):
            path.parent.mkdir(parents=True, exist_ok=True)
            guard = stack.enter_context(path.open('a'))
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def run_child(command, log_path, timeout):
    check(timeout > 0, 'session deadline reached')
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT)+os.pathsep+env.get('PYTHONPATH',''), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false', PATH=str(Path(sys.executable).parent)+os.pathsep+env.get('PATH',''))
    env.pop('USE_TORCH', None)
    child = None
    previous = {}
    def interrupted(signum, frame):
        raise RuntimeError('supervised stage interrupted')
    for sig in (signal.SIGTERM,signal.SIGINT):
        previous[sig]=signal.signal(sig,interrupted)
    try:
        with Path(log_path).open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            code = child.wait(timeout=timeout)
            check(code == 0, 'evaluation stage failed; no retry')
    finally:
        for sig in previous: signal.signal(sig,signal.SIG_IGN)
        try:
            stop_process_group(child)
        finally:
            for sig,handler in previous.items(): signal.signal(sig,handler)


def check_data(directory):
    expected = read(ROOT/DATA_EXPECTED)
    check(expected['status'] == 'ALL_15_OUTPUTS_BYTE_IDENTICAL', 'missing data reference')
    for row in expected['files']:
        check(sha(safe_path(directory, row['path'])) == row['sha256'], 'formal data reproduction differs')


def custom_worker(plan, output):
    require_release(plan)
    check(read(output/'launch.json')['pid'] == os.getppid(), 'worker requires live generation parent')
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    check(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'CUDA BF16 unavailable')
    tokenizer = AutoTokenizer.from_pretrained(plan['model_path'], local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(plan['model_path'], local_files_only=True,
                trust_remote_code=False, torch_dtype=torch.bfloat16, device_map='auto').eval()
    spec = read(ROOT/QUESTIONS); torch.manual_seed(42)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    with (output/'answers.jsonl').open('x') as stream:
        for q in spec['questions']:
            ids = tokenizer.apply_chat_template([{'role':'system','content':spec['system_message']}] + q['messages'],
                    tokenize=True, add_generation_prompt=True, return_tensors='pt').to(model.device)
            check(ids.shape[1]+512 <= 2048, 'custom question exceeds reviewed context')
            with torch.inference_mode():
                tokens = model.generate(ids, max_new_tokens=512, do_sample=False)[0, ids.shape[1]:]
            reason = 'eos' if len(tokens) and int(tokens[-1]) in eos else 'length'
            answer = tokenizer.decode(tokens, skip_special_tokens=True)
            check(reason == 'eos' and answer.strip(), 'custom answer truncated or empty')
            stream.write(json.dumps({'id':q['id'], 'answer':answer, 'finish_reason':reason,
                                     'generated_tokens':len(tokens)}, ensure_ascii=False)+'\n')
            stream.flush(); os.fsync(stream.fileno())


def generate(plan, plan_hash, output):
    check(plan.get('release_state') == 'TARGET_AUDITED_RELEASED', 'generation not released')
    with gpu_guard():
        return generate_locked(plan, plan_hash, output)


def generate_locked(plan, plan_hash, output):
    deadline = require_release(plan)
    output = Path(output).resolve()
    check(not output.exists(), 'output exists; no automatic replay')
    check(output.is_relative_to((ROOT/'logs').resolve()), 'generation output must be under logs')
    check(not os.environ.get('DATA_INPUT') and not os.environ.get('TOKENIZER_PATH'), 'unset data overrides')
    check(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True).strip(), 'GPU occupied')
    # Exclusive permanent claim prevents accidental repetition in another directory.
    save(ROOT/'logs/clean-eval-claims'/f'{plan_hash}.json', {'output':str(output)})
    output.mkdir(parents=True)
    save(output/'generation_plan.json', plan)
    save(output/'launch.json', {'pid':os.getpid(), 'plan_sha256':plan_hash, 'host':HOST, 'judge_calls':0})
    old = {}
    def interrupted(signum, frame): raise RuntimeError('generation interrupted')
    for sig in (signal.SIGTERM, signal.SIGINT): old[sig] = signal.signal(sig, interrupted)
    try:
        check(sys.version_info[:2] == (3,10), 'target Python must be 3.10')
        check({k:importlib.metadata.version(k) for k in RUNTIME_VERSIONS} == RUNTIME_VERSIONS, 'runtime package versions changed')
        run_child([sys.executable,'-m','pip','check'], output/'pip_check.log', min(120,deadline-time.time()))
        run_child([sys.executable,'-m','pip','freeze'], output/'environment_freeze.txt', min(120,deadline-time.time()))
        audit=read(ROOT/plan['target_audit']['path'])
        check(sha(output/'environment_freeze.txt') == audit['environment_freeze']['sha256'], 'runtime environment changed since target audit')
        run_child([sys.executable,str(ROOT/'scripts/step1_data_prep.py'),'--protocol',str(ROOT/DATA_PROTOCOL),
                   '--output-dir',str(output/'data')], output/'data_prep.log', min(1800,deadline-time.time()))
        check_data(output/'data')
        run_child(ev.opencompass_command(ROOT/plan['config']['path'],output/'opencompass'),
                  output/'opencompass.log', min(14400,deadline-time.time()))
        rows = ev.validate_benchmarks(output/'opencompass',plan,plan['model_path'])
        save(output/'benchmarks.json',rows)
        run_child([sys.executable,str(Path(__file__).resolve()),'worker','--plan',str(output/'generation_plan.json'),
                   '--plan-sha256',sha(output/'generation_plan.json'),'--output',str(output)],
                  output/'custom.log',deadline-time.time())
        answers = [json.loads(line) for line in (output/'answers.jsonl').read_text().splitlines()]
        validate_answers(answers)
        files = {str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
        save(output/'generation_receipt.json', {'status':'GENERATED_PENDING_LOCAL_SCORING', 'files':files,
             'generation_plan_sha256':sha(output/'generation_plan.json'), 'judge_calls':0})
    except BaseException as exc:
        save(output/'failure.json', {'status':'FAILED_NO_RETRY','error_type':type(exc).__name__,'judge_calls':0})
        raise
    finally:
        for sig, handler in old.items(): signal.signal(sig,handler)


def validate_answers(answers):
    spec = read(ROOT/QUESTIONS)
    check(len(answers) == 20 and [a['id'] for a in answers] == [q['id'] for q in spec['questions']], 'answer coverage/order mismatch')
    check(all(isinstance(a['answer'],str) and a['answer'].strip() and a.get('finish_reason') == 'eos'
              and type(a.get('generated_tokens')) is int and 0 < a['generated_tokens'] <= 512 for a in answers),
          'empty, truncated or invalid answer')


def bind_score_plan(answers, answers_hash, receipt_hash):
    validate_answers(answers)
    base = load_release('original_base')  # Revalidates the 17-case calibration and frozen rubric.
    requests = copy.deepcopy(base['requests'])
    check([r['question_id'] for r in requests] == [a['id'] for a in answers], 'calibrated question mismatch')
    for row, answer in zip(requests,answers):
        part = row['body']['contents'][0]['parts'][0]
        payload = json.loads(part['text']); payload['candidate_answer'] = answer['answer']
        part['text'] = json.dumps(payload,ensure_ascii=False)
    plan = {k:copy.deepcopy(base[k]) for k in ('schema','calibration','calibration_receipt_sha256',
        'profile_sha256','dependencies','max_calls','max_estimated_usd','historical_scores_reused','generation_reused','model')}
    template = {k:requests[0]['body'][k] for k in ('systemInstruction','generationConfig')}
    plan.update(scope='clean_eval_custom20', model_name='final_dpo', requests=requests,
                answers_sha256=answers_hash, generation_receipt_sha256=receipt_hash,
                identity=hashlib.sha256(scoring.canonical({'answers':answers_hash,'protocol':template,'model':base['model']})).hexdigest())
    plan['dependencies']['scripts/week8_clean_eval.py'] = sha(Path(__file__))
    plan['dependencies']['scripts/week8_pipeline_score.py'] = sha(ROOT/'scripts/week8_pipeline_score.py')
    return plan


def verify_source_files(source, receipt):
    check(not source.is_symlink() and not any(p.is_symlink() for p in source.rglob('*')), 'symlink generation evidence')
    actual = {str(p.relative_to(source)) for p in source.rglob('*') if p.is_file()}
    check(actual == set(receipt['files']) | {'generation_receipt.json'}, 'generation manifest coverage mismatch')
    for name,h in receipt['files'].items():
        check(sha(safe_path(source,name)) == h, 'generation artifact changed')


def validate_gpu_release(artifact, receipt_hash):
    check(isinstance(artifact,dict), 'GPU release receipt required')
    path = safe_path(ROOT,artifact['path'])
    check(sha(path) == artifact['sha256'], 'GPU release receipt changed')
    release = read(path)
    check(release.get('instance') == '320' and release.get('state') == 'stopped'
          and release.get('generation_receipt_sha256') == receipt_hash,
          'GPU release state or generation binding mismatch')
    observed = dt.datetime.fromisoformat(release['observed_utc'])
    check(observed.tzinfo is not None and observed.timestamp() <= time.time()+60,
          'GPU release observation time invalid')
    evidence = release['evidence']
    check(sha(safe_path(ROOT,evidence['path'])) == evidence['sha256'], 'GPU release evidence changed')


def prepare_score(source, receipt_hash, gpu_release):
    validate_gpu_release(gpu_release,receipt_hash)
    source = Path(source).resolve()
    check(sha(source/'generation_receipt.json') == receipt_hash, 'generation receipt mismatch')
    receipt = read(source/'generation_receipt.json')
    check(receipt['status'] == 'GENERATED_PENDING_LOCAL_SCORING' and receipt['judge_calls'] == 0, 'generation incomplete')
    check(not (source/'failure.json').exists(), 'failed generation')
    verify_source_files(source,receipt)
    required = {'generation_plan.json','answers.jsonl','benchmarks.json','launch.json','pip_check.log','environment_freeze.txt'}
    check(required <= set(receipt['files']), 'receipt coverage incomplete')
    plan = read(source/'generation_plan.json')
    check(sha(source/'generation_plan.json') == receipt['generation_plan_sha256'], 'generation plan changed')
    validate_plan(plan)
    check(plan['release_state'] == 'TARGET_AUDITED_RELEASED', 'source generation not released')
    validate_target_audit(plan)
    audit=read(ROOT/plan['target_audit']['path'])
    check(sha(source/'environment_freeze.txt') == audit['environment_freeze']['sha256'], 'generation environment differs from audited freeze')
    check_data(source/'data')
    rows = ev.validate_benchmarks(source/'opencompass',plan,plan['model_path'])
    check(rows == read(source/'benchmarks.json'), 'benchmark summary differs')
    answers = [json.loads(line) for line in (source/'answers.jsonl').read_text().splitlines()]
    result = bind_score_plan(answers,sha(source/'answers.jsonl'),receipt_hash)
    result['generation_source'] = str(source.relative_to(ROOT))
    result['gpu_release'] = gpu_release
    return result


def score(plan, output, *, execute=False, _worker=False):
    expected = prepare_score(ROOT/plan['generation_source'],plan['generation_receipt_sha256'],plan.get('gpu_release'))
    check(plan == expected, 'new scoring plan no longer matches generation/calibration/code')
    output = Path(output).resolve()
    check(output.is_relative_to((ROOT/'logs/clean-eval-scores').resolve()), 'score output must be under logs/clean-eval-scores')
    source = ROOT/plan['generation_source']
    check(not output.is_relative_to(source.resolve()) and not source.resolve().is_relative_to(output), 'score output overlaps generation')
    if execute:
        check(socket.gethostname() != HOST, 'paid scoring must run after retrieval on local host')
        if _worker:
            scoring.run(plan,output,ROOT/'reports/week8/score_only_claims')
        else:
            # Keep supervision evidence outside the runner's exclusive output.
            identity=hashlib.sha256(scoring.canonical(plan)).hexdigest()
            directory=ROOT/'logs/clean-eval-supervision'/identity
            directory.mkdir(parents=True,exist_ok=True)
            path=directory/'plan.json'
            if path.exists(): check(read(path)==plan,'supervised scoring plan changed')
            else: save(path,plan)
            run_child([sys.executable,str(Path(__file__).resolve()),'score-worker',
                       '--plan',str(path),'--plan-sha256',sha(path),'--output',str(output),
                       '--supervisor-pid',str(os.getpid())],
                      directory/(str(time.time_ns())+'.log'),7200)

    result = verify_completed(plan,output)
    rows = read(source/'benchmarks.json')
    rows += [{'model': 'final_dpo', 'dataset':'custom20', 'score':result['weighted_mean'],
              'questions':20, 'evidence_kind':'fresh_gemini_ai_judge_not_human'}]
    summary = output/'pipeline_summary.csv'
    if not summary.exists(): ev.write_csv(summary, rows)
    else:
        import csv
        fields=list(dict.fromkeys(k for row in rows for k in row))
        with summary.open() as stream:
            reader=csv.DictReader(stream); existing=list(reader)
            check(reader.fieldnames==fields,'CSV differs from source fields')
        expected_rows=[{k:str(row[k]) if k in row else '' for k in fields} for row in rows]
        check(existing==expected_rows, 'CSV differs from source')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['candidate','audit','generate','worker','prepare-score','score','verify-score','score-worker'])
    parser.add_argument('--plan',type=Path); parser.add_argument('--plan-sha256')
    parser.add_argument('--supervisor-pid',type=int)
    parser.add_argument('--binding',type=Path); parser.add_argument('--source',type=Path)
    parser.add_argument('--gpu-release',type=Path)
    parser.add_argument('--receipt-sha256'); parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.action == 'candidate':
        check(args.binding is not None,'binding required'); save(args.output,candidate(args.binding))
    elif args.action == 'prepare-score':
        check(args.source is not None and args.receipt_sha256,'source and receipt hash required')
        check(args.gpu_release is not None,'GPU release receipt required')
        artifact={'path':str(args.gpu_release.resolve().relative_to(ROOT)), 'sha256':sha(args.gpu_release)}
        save(args.output,prepare_score(args.source,args.receipt_sha256,artifact))
    else:
        check(args.plan is not None and args.plan_sha256,'plan and SHA256 required')
        plan=load_bound_plan(args.plan,args.plan_sha256)
        if args.action == 'audit': validate_plan(plan,model_files=True)
        elif args.action == 'generate': generate(plan,args.plan_sha256,args.output)
        elif args.action == 'worker': custom_worker(plan,args.output)
        else:
            if args.action=='score-worker':
                check(args.supervisor_pid==os.getppid(),'score worker requires live supervisor')
            score(plan,args.output,execute=args.action in ('score','score-worker'),_worker=args.action=='score-worker')
    print('CLEAN_EVAL_'+args.action.upper().replace('-','_')+'_OK')


if __name__ == '__main__': main()
