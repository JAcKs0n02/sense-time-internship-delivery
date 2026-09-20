import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))

class WeightTests(unittest.TestCase):
    def verifier(self):
        self.assertIsNotNone(importlib.util.find_spec('week8_distillation_verify'),'export verifier missing')
        from week8_distillation_verify import verify_export
        return verify_export

    def fixture(self, root):
        import torch
        from safetensors.torch import save_file
        base=root/'base';adapter=root/'adapter';merged=root/'merged'
        for p in (base,adapter,merged):p.mkdir()
        key='model.layers.0.self_attn.q_proj.weight'
        for p in (base,merged):
            (p/'config.json').write_text(json.dumps({'model_type':'qwen2','hidden_size':896,'num_hidden_layers':24}))
        (adapter/'adapter_config.json').write_text(json.dumps({'peft_type':'LORA','r':8,'lora_alpha':16,'base_model_name_or_path':str(base),'bias':'none'}))
        a=torch.ones(8,2);b=torch.zeros(2,8);b[0,0]=1;b[1,0]=2
        save_file({'base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight':a,
                   'base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight':b},str(adapter/'adapter_model.safetensors'))
        save_file({key:torch.zeros(2,2), 'model.norm.weight':torch.ones(2)},str(base/'model.safetensors'))
        save_file({key:torch.tensor([[2.,2.],[4.,4.]]),'model.norm.weight':torch.ones(2)},str(merged/'model.safetensors'))
        return base,adapter,merged,key

    def test_real_tensors_verify_actual_lora_merge(self):
        verify=self.verifier()
        with tempfile.TemporaryDirectory() as tmp:
            base,adapter,merged,key=self.fixture(Path(tmp))
            result=verify(base,adapter,merged)
            self.assertEqual(result['updated_modules'],1)
            self.assertEqual(result['merged_tensors'],2)

    def test_corrupt_or_unmerged_export_is_rejected(self):
        verify=self.verifier()
        import torch
        from safetensors.torch import save_file
        for kind in ('unchanged','wrong_merge','nan','missing','unrelated_change'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                base,adapter,merged,key=self.fixture(Path(tmp))
                tensors={key:torch.tensor([[2.,2.],[4.,4.]]),'model.norm.weight':torch.ones(2)}
                if kind=='unchanged':tensors[key]=torch.zeros(2,2)
                if kind=='wrong_merge':tensors[key]=torch.ones(2,2)*20
                if kind=='nan':tensors[key][0,0]=float('nan')
                if kind=='missing':tensors.pop(key)
                if kind=='unrelated_change':tensors['model.norm.weight']=torch.zeros(2)
                save_file(tensors,str(merged/'model.safetensors'))
                with self.assertRaises(ValueError):verify(base,adapter,merged)

    def test_sharded_export_matches_index_and_rejects_missing_shard(self):
        verify=self.verifier()
        from safetensors.torch import load_file,save_file
        with tempfile.TemporaryDirectory() as tmp:
            base,adapter,merged,key=self.fixture(Path(tmp))
            tensors=load_file(str(merged/'model.safetensors'))
            (merged/'model.safetensors').unlink()
            save_file({key:tensors[key]},str(merged/'model-1.safetensors'))
            save_file({'model.norm.weight':tensors['model.norm.weight']},str(merged/'model-2.safetensors'))
            (merged/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{key:'model-1.safetensors','model.norm.weight':'model-2.safetensors'}}))
            self.assertEqual(verify(base,adapter,merged)['merged_tensors'],2)
            (merged/'model-2.safetensors').unlink()
            with self.assertRaises(ValueError):verify(base,adapter,merged)

    def test_bfloat16_merge_rounding_is_accepted(self):
        verify=self.verifier()
        import torch
        from safetensors.torch import load_file,save_file
        with tempfile.TemporaryDirectory() as tmp:
            base,adapter,merged,key=self.fixture(Path(tmp))
            for folder in (base,adapter,merged):
                path=folder/('adapter_model.safetensors' if folder==adapter else 'model.safetensors')
                tensors=load_file(str(path))
                if folder==base:tensors[key]=torch.full((2,2),0.003)
                save_file({k:v.to(torch.bfloat16) for k,v in tensors.items()},str(path))
            self.assertEqual(verify(base,adapter,merged)['updated_modules'],1)

    def test_cold_load_failure_prevents_artifact_success(self):
        import distill
        self.assertTrue(hasattr(distill,'verify_student_artifacts'),'cold-load verification is not wired')
        from unittest.mock import patch
        from compare_distillation import run_bounded
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base,adapter,merged,key=self.fixture(root)
            for name in ('tokenizer.json','tokenizer_config.json','generation_config.json'):
                (merged/name).write_text('{}')
            def external(command, log, timeout):
                if 'week8_full_cold_load.py' in str(command[1]):
                    raise subprocess.CalledProcessError(1,command)
                return run_bounded(command,log,timeout)
            with patch('compare_distillation.run_bounded',side_effect=external):
                with self.assertRaises(subprocess.CalledProcessError):
                    distill.verify_student_artifacts(base,adapter,merged,root,60)
            self.assertTrue((root/'student_tensor_verification.json').exists())
            self.assertFalse((root/'student_artifact_verification.json').exists())

    def test_artifact_receipt_requires_unchanged_model_after_cold_load(self):
        import distill
        from unittest.mock import patch
        from compare_distillation import run_bounded
        from common import write_json
        for corrupt in (False,True):
            with self.subTest(corrupt=corrupt),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);base,adapter,merged,key=self.fixture(root)
                for name in ('tokenizer.json','tokenizer_config.json','generation_config.json'):
                    (merged/name).write_text('{}')
                def external(command,log,timeout):
                    if 'week8_full_cold_load.py' not in str(command[1]):
                        return run_bounded(command,log,timeout)
                    # This replaces only the unavailable CUDA boundary; no GPU evidence is produced.
                    write_json(root/'student_cold_load.json',{'status':'COLD_LOAD_PASS',
                               'model_path':str(merged.resolve()),'finite_logits':True,
                               'generated_tokens':2,'dtype':'torch.bfloat16','device':'cuda:0'})
                    if corrupt:(merged/'model.safetensors').write_bytes(b'changed')
                with patch('compare_distillation.run_bounded',side_effect=external):
                    if corrupt:
                        with self.assertRaises(ValueError):distill.verify_student_artifacts(base,adapter,merged,root,60)
                        self.assertFalse((root/'student_artifact_verification.json').exists())
                    else:
                        result=distill.verify_student_artifacts(base,adapter,merged,root,60)
                        self.assertEqual(result['quality_comparison'],'pending')
                        self.assertTrue((root/'student_artifact_verification.json').exists())

if __name__=='__main__':unittest.main()
