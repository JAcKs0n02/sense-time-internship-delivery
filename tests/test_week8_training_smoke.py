"""Regressions for bounded training and fail-closed acceptance, without a GPU."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))


class SmokeTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('week8_training_smoke'),
                             'bounded smoke entry is not implemented')
        import week8_training_smoke
        self.m = week8_training_smoke

    def test_configs_cannot_inherit_full_training_or_historical_reference(self):
        sft, dpo = self.m.build_configs(ROOT, Path('/data'), Path('/new-run'), '/base')
        for c in (sft, dpo):
            self.assertEqual(c['max_steps'], 3)
            self.assertEqual(c['save_steps'], 3)
            self.assertEqual(c['eval_strategy'], 'no')
            self.assertIsNone(c['resume_from_checkpoint'])
            self.assertFalse(c['overwrite_output_dir'])
            self.assertNotIn('max_samples', c)
        self.assertEqual(sft['gradient_accumulation_steps'], 4)
        self.assertEqual(dpo['gradient_accumulation_steps'], 8)
        self.assertEqual(dpo['model_name_or_path'], '/new-run/models/smoke_sft')
        self.assertNotIn('ref_model', dpo)

    def test_state_rejects_missing_steps_zero_updates_and_nonfinite_metrics(self):
        good = {'global_step': 3, 'log_history': [
            {'step': i, 'loss': 1.0, 'grad_norm': 0.1, 'learning_rate': 1e-5}
            for i in (1, 2, 3)]}
        self.m.validate_state(good)
        bads = []
        x=copy.deepcopy(good); x['global_step']=40; bads.append(x)
        x=copy.deepcopy(good); x['log_history'].pop(); bads.append(x)
        for key, value in [('loss', float('nan')), ('grad_norm', float('inf')), ('grad_norm', 0), ('learning_rate', 0)]:
            x=copy.deepcopy(good)
            for row in x['log_history']: row[key]=value
            bads.append(x)
        for bad in bads:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.m.validate_state(bad)

    def test_adapter_rejects_zero_lora_update_and_nonfinite_tensors(self):
        import torch
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            (p/'adapter_config.json').write_text(json.dumps({'r':8,'lora_alpha':16,'peft_type':'LORA'}))
            (p/'trainer_state.json').write_text(json.dumps({'global_step':3,'log_history':[
                {'step':i,'loss':1.,'grad_norm':0.1,'learning_rate':1e-5} for i in (1,2,3)]}))
            def put(value):
                save_file({'layer.lora_A.weight':torch.ones(8,2),
                           'layer.lora_B.weight':torch.full((2,8),value)},str(p/'adapter_model.safetensors'))
            put(0.)
            with self.assertRaises(ValueError): self.m.verify_adapter(p)
            put(float('nan'))
            with self.assertRaises(ValueError): self.m.verify_adapter(p)
            put(0.01)
            result=self.m.verify_adapter(p)
            self.assertEqual(result['global_step'],3)
            self.assertGreater(result['nonzero_lora_b_tensors'],0)

    def test_manifest_rejects_modified_and_escaping_files(self):
        from common import sha256
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder); (p/'input').write_text('frozen')
            refs=[{'path':'input','sha256':sha256(p/'input')}]
            self.m.verify_files(p,refs)
            (p/'input').write_text('changed')
            with self.assertRaises(ValueError): self.m.verify_files(p,refs)
            with self.assertRaises(ValueError): self.m.verify_files(p,[{'path':'../input','sha256':'0'*64}])
            with self.assertRaises(ValueError): self.m.verify_files(p,[])

    def test_optimizer_rejects_wrong_steps_and_zero_moments(self):
        import torch
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder); (p/'checkpoint-3').mkdir()
            def put(step, value):
                torch.save({'state':{0:{'step':torch.tensor(step),'exp_avg':torch.tensor([value])}}},p/'checkpoint-3/optimizer.pt')
            for step,value in [(2,1.),(3,0.),(3,float('inf'))]:
                put(step,value)
                with self.assertRaises(ValueError): self.m.verify_optimizer(p)
            put(3,0.1)
            self.assertEqual(self.m.verify_optimizer(p)['step'],3)

    def test_merge_rejects_incomplete_index_and_nonfinite_shards(self):
        import torch
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder); (p/'config.json').write_text('{}')
            (p/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'weight':'part.safetensors'}}))
            for tensors in [{'other':torch.ones(2)}, {'weight':torch.tensor([float('nan')])}]:
                save_file(tensors,str(p/'part.safetensors'))
                with self.assertRaises(ValueError): self.m.verify_merge(p)
            save_file({'weight':torch.ones(2)},str(p/'part.safetensors'))
            self.assertEqual(self.m.verify_merge(p)['tensors'],1)

    def test_admission_refuses_different_base_even_with_rehashed_manifest(self):
        from common import sha256
        lock={'scope':'SFT3_MERGE_DPO3_ONLY','full_training_allowed':False,
              'base_model':'/different-base',
              'files':[{'path':p,'sha256':sha256(ROOT/p)} for p in sorted(self.m.REQUIRED)]}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'lock.json'; path.write_text(json.dumps(lock))
            with self.assertRaises(ValueError): self.m.admit(ROOT,path,sha256(path))

    def test_cli_dry_run_is_bounded_and_refuses_reuse_or_changed_digest(self):
        import subprocess
        from common import sha256
        base=json.loads((ROOT/'reports/week8/phase2_final_release/runtime_lock.json').read_text())['model_path']
        lock={'scope':'SFT3_MERGE_DPO3_ONLY','full_training_allowed':False,'base_model':base,
              'files':[{'path':p,'sha256':sha256(ROOT/p)} for p in sorted(self.m.REQUIRED)]}
        with tempfile.TemporaryDirectory() as folder:
            d=Path(folder); path=d/'lock.json'; path.write_text(json.dumps(lock))
            cmd=[sys.executable,str(ROOT/'scripts/week8_training_smoke.py'),'--lock',str(path),
                 '--lock-sha256',sha256(path),'--run-dir',str(d/'run'),'--dry-run']
            result=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            status=json.loads((d/'run/status.json').read_text())
            self.assertFalse(status['training_started'])
            cfg=json.loads((d/'run/planned_configs.json').read_text())
            self.assertEqual(cfg['sft']['max_steps'],3)
            self.assertEqual(cfg['dpo']['max_steps'],3)
            self.assertEqual(cfg['dpo']['model_name_or_path'],str((d/'run/models/smoke_sft').resolve()))
            self.assertNotEqual(subprocess.run(cmd,capture_output=True).returncode,0)
            path.write_text(path.read_text()+' ')
            cmd[cmd.index('--run-dir')+1]=str(d/'new-run')
            self.assertNotEqual(subprocess.run(cmd,capture_output=True).returncode,0)
            self.assertFalse((d/'new-run').exists())

    def test_cli_cannot_train_without_pinned_lock(self):
        import subprocess
        with tempfile.TemporaryDirectory() as folder:
            result=subprocess.run([sys.executable,str(ROOT/'scripts/week8_training_smoke.py'),
                '--run-dir',str(Path(folder)/'run')],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertFalse((Path(folder)/'run').exists())

if __name__ == '__main__': unittest.main()
