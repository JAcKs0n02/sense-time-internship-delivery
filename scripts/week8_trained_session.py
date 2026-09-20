"""Bounded SFT/DPO coordinator. Requires target release and shutdown receipt."""
import argparse,datetime,fcntl,json,os,signal,socket,subprocess,sys,time
from pathlib import Path
from week8_generation_only import ROOT,check,read,save,sha

def stop_group(child):
    if child is None:return
    try:os.killpg(child.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:child.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        child.wait()

def main():
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--release-sha256',required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    check(socket.gethostname()=='autodl-container-be044ebe99-be706b14','wrong host')
    check(sha(args.release)==args.release_sha256,'release mismatch');r=read(args.release)
    check(r['status']=='TARGET_AUDITED_RELEASED','not released');planpath=ROOT/r['plan_path'];check(sha(planpath)==r['plan_sha256'],'plan mismatch')
    check(sha(Path(__file__))==r['coordinator_sha256'],'coordinator mismatch')
    deadline=datetime.datetime.fromisoformat(r['deadline_utc']).timestamp();shutdown=datetime.datetime.fromisoformat(r['platform_shutdown_utc']).timestamp()
    check(time.time()<=datetime.datetime.fromisoformat(r['latest_start_utc']).timestamp(),'start window expired')
    check(0<shutdown-deadline<=600 and deadline-time.time()>=r['minimum_remaining_seconds'],'insufficient protected window')
    guard=(ROOT/'logs/formal-eval-session.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    check(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU occupied')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);save(out/'launch.json',{'release_sha256':args.release_sha256,'pid':os.getpid(),'deadline_utc':r['deadline_utc'],'judge_calls':0})
    child=None;done=[];env=os.environ.copy();env.update(PATH=str(Path(sys.executable).parent)+os.pathsep+env.get('PATH',''),PYTHONPATH=str(ROOT),OMP_NUM_THREADS='14',MKL_NUM_THREADS='14',TOKENIZERS_PARALLELISM='false');env.pop('USE_TORCH',None)
    def interrupted(signum,frame):raise RuntimeError('session interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        for name in ['final_sft','final_dpo']:
            remaining=deadline-time.time();check(remaining>60,'deadline reached')
            cmd=[sys.executable,str(ROOT/'scripts/week8_generation_only.py'),'--plan',str(planpath),'--plan-sha256',r['plan_sha256'],'--model-name',name,'--output',str(out/name)]
            save(out/(name+'-command.json'),cmd)
            with (out/(name+'.log')).open('x') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                code=child.wait(timeout=remaining)
                check(code==0,'model failed; no retry')
                child=None
            check(read(out/name/'status.json')['status']=='GENERATED_PENDING_REVIEW','incomplete result');done.append(name)
        save(out/'status.json',{'status':'GENERATED_PENDING_REVIEW','completed_models':done,'judge_calls':0})
    except BaseException as e:
        stop_group(child);save(out/'status.json',{'status':'FAILED_NO_RETRY','completed_models':done,'error_type':type(e).__name__});raise
    finally:
        # Platform timer is the final hard stop. Successful early runs should be
        # shut down by the caller after artifacts are verified and retrieved.
        save(out/'session_exit.json',{'completed_models':done,'stopped_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
if __name__=='__main__':main()
