"""One bounded 320 session: cold loads, then teacher generation; never training."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path('/root/autodl-tmp/week8-distillation-gpu-20260918')
sys.path.insert(0, str(ROOT/'scripts'))
from common import sha256, write_json
from compare_distillation import run_bounded, verify_model_manifest


def main():
    plan = json.loads((ROOT/'execution_plan.json').read_text())
    deadline = datetime.datetime.fromisoformat(plan['deadline_utc'].replace('Z', '+00:00')).timestamp()
    assert socket.gethostname() == 'autodl-container-be044ebe99-be706b14'
    assert time.time() < deadline - 1800, 'insufficient session window'
    guard = (ROOT/'session.lock').open('a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = ROOT/'run-01'
    out.mkdir(exist_ok=False)
    completed = []
    try:
        for item in json.loads((ROOT/'upload_manifest.json').read_text()):
            assert sha256(ROOT/item['path']) == item['sha256'], item['path']
        assert shutil.disk_usage(ROOT).free > 8*1024**3
        assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip(), 'GPU occupied'
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,memory.total', '--format=csv,noheader'], text=True).strip()
        assert len(gpu.splitlines()) == 1 and 'RTX 3090' in gpu
        teacher = Path('/root/autodl-tmp/qwen25-week4/best_model/qwen25-7b-week4-dpo-corrective-merged')
        student = Path('/root/autodl-tmp/week8-distillation-preflight-20260918/student-Qwen2.5-0.5B-Instruct')
        write_json(out/'launch.json', {'host':socket.gethostname(), 'pid':os.getpid(), 'gpu':gpu,
            'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'deadline_utc':plan['deadline_utc'],
            'disk_free_bytes':shutil.disk_usage(ROOT).free, 'student_training_allowed':False})
        for name, model in [('student',student), ('teacher',teacher)]:
            write_json(out/'status.json', {'status':'RUNNING', 'stage':name+'_cold_load', 'completed':completed})
            verify_model_manifest(model, ROOT/(name+'_manifest.json'))
            receipt = out/(name+'_cold_load.json')
            run_bounded([sys.executable, str(ROOT/'scripts/week8_full_cold_load.py'), '--model',str(model),'--output',str(receipt)],
                        out/(name+'_cold_load.log'), min(300,deadline-time.time()))
            result = json.loads(receipt.read_text())
            assert result['status']=='COLD_LOAD_PASS' and result['finite_logits'] is True
            assert result['model_path']==str(model) and result['dtype']=='torch.bfloat16'
            assert result['device']=='cuda:0' and 1<=result['generated_tokens']<=2
            verify_model_manifest(model, ROOT/(name+'_manifest.json'))
            completed.append(name+'_cold_load')
        assert deadline-time.time()>1500, 'insufficient generation budget'
        write_json(out/'status.json', {'status':'RUNNING','stage':'teacher_generation','completed':completed})
        command=[sys.executable,str(ROOT/'scripts/distill.py'),'generate','--config',str(ROOT/'configs/distillation.json'),
            '--input',str(ROOT/'candidate_prompts.json'),'--output-dir',str(out/'generation'),
            '--teacher',str(teacher),'--teacher-manifest',str(ROOT/'teacher_manifest.json')]
        write_json(out/'generation_command.json',command)
        run_bounded(command,out/'generation.log',min(1500,deadline-time.time()))
        receipt=json.loads((out/'generation/generation_receipt.json').read_text())
        assert receipt['status']=='GENERATED_PENDING_QUALITY_REVIEW' and receipt['generated']==200
        assert receipt['input_sha256']==sha256(ROOT/'candidate_prompts.json')
        for item in receipt['files']:
            assert sha256(out/'generation'/item['path'])==item['sha256']
        completed.append('teacher_generation')
        write_json(out/'status.json',{'status':'GENERATED_PENDING_QUALITY_REVIEW','completed':completed,
            'generated':receipt['generated'],'accepted':receipt['accepted'],'student_training_started':False})
    except BaseException as exc:
        write_json(out/'status.json',{'status':'FAILED_NO_RETRY','completed':completed,'error_type':type(exc).__name__,'error':str(exc)})
        raise
    finally:
        write_json(out/'session_exit.json',{'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':completed})


if __name__=='__main__':
    main()
