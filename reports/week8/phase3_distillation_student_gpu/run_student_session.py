"""Single bounded session: identity, exact data reproduction, train, export and verify."""
import base64,datetime,fcntl,gzip,hashlib,json,os,shutil,socket,subprocess,sys,time
from pathlib import Path
from importlib.metadata import version
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
from common import sha256,write_json
from compare_distillation import run_bounded,verify_model_manifest
from distill import prepare_student_data,verify_target_review,student_training_schedule


def export_evidence(out):
    files=[]
    for p in sorted(out.rglob('*')):
        if p.is_file() and p.suffix in ('.json','.log','.yaml','.txt') and p.stat().st_size<4_000_000:
            b=p.read_bytes();files.append({'path':str(p.relative_to(out)),'sha256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode()})
    payload=json.dumps({'files':files},ensure_ascii=False).encode()
    (ROOT/'student_evidence.txt').write_text(json.dumps({'bundle_sha256':hashlib.sha256(payload).hexdigest(),'gzip_base64':base64.b64encode(gzip.compress(payload)).decode()}))
    write_json(ROOT/'evidence_summary.json',{'files':len(files),'bundle_sha256':hashlib.sha256(payload).hexdigest()})


def main():
    plan=json.loads((ROOT/'execution_plan.json').read_text())
    deadline=datetime.datetime.fromisoformat(plan['deadline_utc'].replace('Z','+00:00')).timestamp()
    assert socket.gethostname()=='autodl-container-be044ebe99-be706b14'
    assert time.time()<deadline-1800,'insufficient bounded session window'
    guard=(ROOT/'session.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=ROOT/'run-01';out.mkdir(exist_ok=False);completed=[]
    try:
        for item in json.loads((ROOT/'upload_manifest.json').read_text()):assert sha256(ROOT/item['path'])==item['sha256'],item['path']
        assert shutil.disk_usage(ROOT).free>5*1024**3
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU occupied'
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.total','--format=csv,noheader'],text=True).strip()
        assert len(gpu.splitlines())==1 and 'RTX 3090' in gpu
        expected={'transformers':'4.50.0','llamafactory':'0.9.3','peft':'0.15.1','torch':'2.5.1+cu121'}
        versions={k:version(k) for k in expected};assert versions==expected,versions
        student=Path('/root/autodl-tmp/week8-distillation-preflight-20260918/student-Qwen2.5-0.5B-Instruct')
        teacher=Path('/root/autodl-tmp/qwen25-week4/best_model/qwen25-7b-week4-dpo-corrective-merged')
        verify_model_manifest(student,ROOT/'student_manifest.json');verify_model_manifest(teacher,ROOT/'teacher_manifest.json')
        write_json(out/'launch.json',{'host':socket.gethostname(),'gpu':gpu,'versions':versions,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'deadline_utc':plan['deadline_utc'],'disk_free_bytes':shutil.disk_usage(ROOT).free})
        c=ROOT/'curation';config=json.loads((ROOT/'configs/distillation.json').read_text())
        verify_target_review(c/'curated_targets.json',c/'retrieved/generation/generation_receipt.json',ROOT/'quality_review.json',config,curation_path=c/'curation_manifest.json')
        stats=prepare_student_data(c/'curated_targets.json',student,out/'preflight_data',config)
        expected_stats=json.loads((ROOT/'expected_statistics.json').read_text())
        for key in ['raw_count','clean_count','train_count','validation_count','rejected','exact_duplicates','fuzzy_duplicates','prompt_overlap','max_length','outputs']:
            assert stats[key]==expected_stats[key],key
        schedule=student_training_schedule(stats['train_count']);assert schedule==json.loads((ROOT/'expected_schedule.json').read_text())
        completed.append('identity_and_exact_preparation');write_json(out/'status.json',{'status':'RUNNING','stage':'student_training','completed':completed})
        command=[sys.executable,str(ROOT/'scripts/distill.py'),'train','--config',str(ROOT/'configs/distillation.json'),'--input',str(c/'curated_targets.json'),'--output-dir',str(out/'student'),'--student',str(student),'--student-manifest',str(ROOT/'student_manifest.json'),'--generation-receipt',str(c/'retrieved/generation/generation_receipt.json'),'--quality-review',str(ROOT/'quality_review.json'),'--curation-manifest',str(c/'curation_manifest.json')]
        write_json(out/'training_command.json',command)
        run_bounded(command,out/'student_pipeline.log',min(1800,deadline-time.time()-60))
        result=json.loads((out/'student/status.json').read_text());assert result['status']=='STUDENT_TRAINING_VERIFIED_COMPARISON_PENDING'
        for name,digest in expected_stats['outputs'].items():assert sha256(out/'student/data'/name)==digest
        assert json.loads((out/'student/student_training_schedule.json').read_text())==schedule
        verify_model_manifest(student,ROOT/'student_manifest.json')
        completed.append('two_epoch_training_export_tensor_cold_verification')
        write_json(out/'status.json',{'status':'STUDENT_TRAINING_VERIFIED_COMPARISON_PENDING','completed':completed,'epochs':2,'optimizer_steps':36,'train_count':137,'validation_count':15,'comparison_pending':True})
    except BaseException as exc:
        write_json(out/'status.json',{'status':'FAILED_NO_RETRY','completed':completed,'error_type':type(exc).__name__,'error':str(exc)})
        raise
    finally:
        write_json(out/'session_exit.json',{'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':completed})
        export_evidence(out)

if __name__=='__main__':main()
