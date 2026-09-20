"""Failure boundaries for formal training; tiny CPU tensors are not GPU evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))


def state(n):
    return {'global_step':n,'max_steps':n,'log_history':[
        {'step':i,'loss':1.,'grad_norm':.1,'learning_rate':1e-5} for i in range(1,n+1)]}


class FormalTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('week8_full_training'),'formal entry missing')
        import week8_full_training as m
        self.m=m

    def test_incomplete_duplicate_nonfinite_or_zero_updates_rejected(self):
        good=state(40)
        self.assertEqual(len(self.m.validate_formal_state(good,40)),40)
        bads=[]
        x=copy.deepcopy(good);x['global_step']=3;bads.append(x)
        x=copy.deepcopy(good);x['log_history'][-1]['step']=39;bads.append(x)
        x=copy.deepcopy(good);x['max_steps']=41;bads.append(x)
        for key,value in [('loss',float('nan')),('grad_norm',0.),('learning_rate',0.)]:
            x=copy.deepcopy(good)
            for r in x['log_history']:r[key]=value
            bads.append(x)
        for x in bads:
            with self.subTest(x=x),self.assertRaises(ValueError):self.m.validate_formal_state(x,40)

    def test_optimizer_and_adapter_must_both_have_real_finite_updates(self):
        import torch
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);(p/'checkpoint-40').mkdir()
            s=state(40);s['log_history'] += [{'step':i,'eval_loss':.6} for i in [20,40]]
            (p/'trainer_state.json').write_text(json.dumps(s))
            (p/'adapter_config.json').write_text(json.dumps({'peft_type':'LORA','r':8,'lora_alpha':16,'base_model_name_or_path':'/base'}))
            def put(step=40,b=.1,moment=.1):
                save_file({'layer.lora_A.weight':torch.ones(8,2),'layer.lora_B.weight':torch.full((2,8),b)},str(p/'adapter_model.safetensors'))
                torch.save({'state':{i:{'step':torch.tensor(step),'exp_avg':torch.tensor([moment]),'exp_avg_sq':torch.ones(1)} for i in range(2)}},p/'checkpoint-40/optimizer.pt')
            for step,b,moment in [(39,.1,.1),(40,0.,.1),(40,.1,float('inf'))]:
                put(step,b,moment)
                with self.assertRaises(ValueError):self.m.verify_formal_adapter(p,40,'/base','dpo')
            put();self.assertEqual(self.m.verify_formal_adapter(p,40,'/base','dpo')['optimizer_step'],40)
            import subprocess
            worker=subprocess.run([sys.executable,str(ROOT/'scripts/week8_full_verify.py'),'adapter','--path',str(p),'--output',str(p/'verified.json'),'--steps','40','--base','/base','--stage','dpo'],capture_output=True,text=True)
            self.assertEqual(worker.returncode,0,worker.stderr)
            self.assertEqual(json.loads((p/'verified.json').read_text())['optimizer_step'],40)
            s['log_history'][-1]['eval_loss']=float('nan');(p/'trainer_state.json').write_text(json.dumps(s))
            with self.assertRaises(ValueError):self.m.verify_formal_adapter(p,40,'/base','dpo')

    def test_failed_command_never_runs_later_command(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);marker=p/'later'
            with self.assertRaises(RuntimeError):
                self.m.run_command([sys.executable,'-c','raise SystemExit(7)'],p/'failed.log',p,time.monotonic()+10)
                self.m.run_command([sys.executable,'-c',f'open({str(marker)!r},"w").write("bad")'],p/'later.log',p,time.monotonic()+10)
            self.assertFalse(marker.exists())
            receipt=json.loads((p/'failed.log.process.json').read_text())
            self.assertEqual(receipt['returncode'],7)
            self.assertFalse(receipt['timed_out'])

    def test_timeout_kills_child_process_before_delayed_side_effect(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);marker=p/'leaked'
            child=f'import time,pathlib; time.sleep(1); pathlib.Path({str(marker)!r}).write_text("bad")'
            parent=f'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",{child!r}]); time.sleep(20)'
            with self.assertRaises(TimeoutError):self.m.run_command([sys.executable,'-c',parent],p/'timeout.log',p,time.monotonic()+.2)
            time.sleep(1.1);self.assertFalse(marker.exists())

    def test_expired_deadline_has_no_side_effect(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);marker=p/'bad'
            with self.assertRaises(TimeoutError):self.m.run_command([sys.executable,'-c',f'open({str(marker)!r},"w")'],p/'run.log',p,time.monotonic()-1)
            self.assertFalse(marker.exists())

    def test_cold_receipt_requires_matching_model_and_finite_logits(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);r=p/'receipt.json'
            good={'status':'COLD_LOAD_PASS','model_path':str(p.resolve()),'finite_logits':True,'generated_tokens':2,'dtype':'torch.bfloat16','device':'cuda:0'}
            for key,value in [('model_path','/old-smoke'),('finite_logits',False),('generated_tokens',0),('device','cpu')]:
                bad=dict(good);bad[key]=value;r.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):self.m.verify_cold_receipt(r,p)
            r.write_text(json.dumps(good));self.assertEqual(self.m.verify_cold_receipt(r,p)['status'],'COLD_LOAD_PASS')

    def test_real_candidate_dry_run_blocks_reuse_tamper_and_execution(self):
        import subprocess
        from common import sha256
        with tempfile.TemporaryDirectory() as folder:
            d=Path(folder);release=self.m.build_release(ROOT)
            f=d/'release.json';f.write_text(json.dumps(release))
            cmd=[sys.executable,str(ROOT/'scripts/week8_full_training.py'),'--release',str(f),'--release-sha256',sha256(f),'--run-dir',str(d/'dry'),'--dry-run']
            r=subprocess.run(cmd,capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr)
            self.assertFalse(json.loads((d/'dry/status.json').read_text())['training_started'])
            self.assertFalse((d/'dry/sft').exists())
            self.assertNotEqual(subprocess.run(cmd,capture_output=True).returncode,0)
            cmd[cmd.index('--run-dir')+1]=str(d/'actual');cmd.remove('--dry-run')
            self.assertNotEqual(subprocess.run(cmd,capture_output=True).returncode,0)
            self.assertFalse((d/'actual').exists())
            cmd.append('--dry-run');f.write_text(f.read_text()+' ')
            self.assertNotEqual(subprocess.run(cmd,capture_output=True).returncode,0)
            self.assertFalse((d/'actual').exists())

    def test_deadline_covers_cpu_validation_not_only_subprocess(self):
        with self.assertRaises(TimeoutError):
            with self.m.session_timeout(time.monotonic()+.05):
                time.sleep(.2)

    def test_wrong_trainer_step_header_stops_before_delayed_work(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);marker=p/'bad'
            code=f'import time,pathlib; print("Total optimization steps = 3",flush=True); time.sleep(1); pathlib.Path({str(marker)!r}).write_text("bad")'
            with self.assertRaisesRegex(RuntimeError,'step target'):
                self.m.run_command([sys.executable,'-c',code],p/'log',p,time.monotonic()+5,training_steps=1775)
            self.assertFalse(marker.exists())

    def test_evaluation_sample_count_does_not_override_training_header(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            code='print("***** Running training *****\\nNum examples = 1,422\\nTotal optimization steps = 1,775\\n***** Running Evaluation *****\\nNum examples = 158")'
            result=self.m.run_command([sys.executable,'-c',code],p/'log',p,time.monotonic()+5,training_steps=1775,training_samples=1422)
            self.assertTrue(result['header_verified'])

    def test_wrong_training_sample_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            code='print("***** Running training *****\\nNum examples = 100\\nTotal optimization steps = 1,775")'
            with self.assertRaisesRegex(RuntimeError,'sample count'):
                self.m.run_command([sys.executable,'-c',code],p/'log',p,time.monotonic()+5,training_steps=1775,training_samples=1422)

    def test_sft_failure_records_phase_and_blocks_dpo(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);run=p/'run';run.mkdir()
            cli=p/'fake-cli';cli.write_text('#!/bin/sh\nexit 7\n');cli.chmod(0o755)
            with self.assertRaises(RuntimeError):
                self.m.execute_stages(p,run,{}, {'expected_steps':{'sft':1775,'dpo':40}},time.monotonic()+10,str(cli))
            status=json.loads((run/'status.json').read_text())
            self.assertEqual(status['phase'],'sft')
            self.assertEqual(status['status'],'FORMAL_FAILED')
            self.assertFalse((run/'dpo').exists())
            self.assertFalse((run/'models').exists())

    def test_every_stage_failure_blocks_later_work(self):
        from unittest.mock import patch
        phases=['sft','sft_verify','sft_merge','sft_merge_verify','sft_cold_load',
                'dpo','dpo_verify','dpo_merge','dpo_merge_verify','dpo_cold_load']
        for failed in range(len(phases)):
            with self.subTest(phase=phases[failed]),tempfile.TemporaryDirectory() as folder:
                p=Path(folder);run=p/'run';run.mkdir();calls=[]
                def command(args,log,cwd,deadline,**kwargs):
                    calls.append(args)
                    if len(calls)==failed+1:raise RuntimeError('injected phase failure')
                    if '--output' in args:Path(args[args.index('--output')+1]).write_text('{}')
                with patch.object(self.m,'run_command',side_effect=command),patch.object(self.m,'verify_cold_receipt',return_value={}):
                    with self.assertRaisesRegex(RuntimeError,'injected'):
                        self.m.execute_stages(p,run,{'sft':{'model_name_or_path':'/base'},'dpo':{'model_name_or_path':'/sft'}},{'expected_steps':{'sft':1775,'dpo':40}},time.monotonic()+10)
                self.assertEqual(len(calls),failed+1)
                status=json.loads((run/'status.json').read_text())
                self.assertEqual(status['phase'],phases[failed])
                self.assertEqual(status['status'],'FORMAL_FAILED')
                self.assertFalse(status['formal_evaluation_allowed'])

    def test_nan_dpo_reward_is_not_accepted_as_finite_training(self):
        x=state(40);x['log_history'][0]['rewards/chosen']=float('nan')
        with self.assertRaises(ValueError):self.m.validate_formal_state(x,40)

    def test_expired_execution_session_rejected_before_gpu_check(self):
        from common import sha256
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'release.json';r=self.m.build_release(ROOT)
            r['execution_enabled']=True
            r['session']={'shutdown_confirmed':True,'shutdown_deadline_utc':'2020-01-01T00:00:00+00:00'}
            p.write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError,'expired'):
                self.m.admit_release(ROOT,p,sha256(p),False)

    def test_release_cannot_omit_pinned_candidate_or_accept_different_config(self):
        from common import sha256
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'release.json';r=self.m.build_release(ROOT)
            r['files']=[f for f in r['files'] if f['path']!='configs/week8_full_training_candidate/dpo.yaml']
            p.write_text(json.dumps(r))
            with self.assertRaises(ValueError):self.m.admit_release(ROOT,p,sha256(p),True)
        cfg={'sft':{'max_steps':3},'dpo':{'max_steps':40}}
        with self.assertRaises(ValueError):self.m.validate_configs(cfg)

if __name__=='__main__':unittest.main()
