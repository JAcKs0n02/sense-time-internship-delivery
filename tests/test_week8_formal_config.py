import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import step3_eval


class FormalConfigTests(unittest.TestCase):
    def test_model_mismatch_and_dependency_drift_block_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'configs').mkdir()
            for name in ('config.py', 'records.json', 'template.py'):
                (root / name).write_text('original')
            artifact = lambda name: dict(path=name, sha256=step3_eval.sha256(root / name))
            lock = dict(verified=True, config=artifact('config.py'), records=artifact('records.json'),
                        dependencies=[artifact('template.py')], model_path='/approved/model',
                        judge=dict(model='judge-v1', url='https://example.com/v1'))
            path = root / 'configs/week8_evaluation_preflight.json'
            path.write_text(json.dumps(dict(runtime_lock=lock)))
            args = SimpleNamespace(model='/wrong/model', judge_model='judge-v1', judge_url='https://example.com/v1')
            with patch.object(step3_eval, 'ROOT', root), patch.object(step3_eval, 'load_judge_profile', return_value={'request_template':{'model':'judge-v1'},'base_url':'https://example.com/v1'}):
                with self.assertRaisesRegex(ValueError, 'model'):
                    step3_eval.require_evaluation_lock(args)
                args.model = '/approved/model'
                self.assertEqual(step3_eval.require_evaluation_lock(args), lock)
                (root / 'template.py').write_text('changed')
                with self.assertRaisesRegex(ValueError, 'artifact changed'):
                    step3_eval.require_evaluation_lock(args)

    def test_command_uses_only_bound_configuration(self):
        cmd = step3_eval.opencompass_command(Path('/fixed/config.py'), Path('/out'))
        self.assertEqual(cmd[1], '/fixed/config.py')
        self.assertNotIn('--hf-path', cmd)
        self.assertNotIn('--hf-type', cmd)
        self.assertIn('--dump-eval-details', cmd)

    def test_generated_formal_configuration_survives_real_cli_parsing(self):
        from build_week8_formal_eval_config import build_config
        from opencompass.utils.run import get_config_from_arg
        from mmengine.config import Config
        with tempfile.TemporaryDirectory() as directory:
            config, binding = build_config(ROOT, Path('/approved/model'))
            path = Path(directory) / 'formal.py'
            Config(config).dump(str(path))
            parsed = get_config_from_arg(SimpleNamespace(config=str(path), accelerator=None))
            self.assertEqual(len(parsed.datasets), 119)
            model = parsed.models[0]
            self.assertEqual(model.path, '/approved/model')
            self.assertFalse(model.tokenizer_only)
            self.assertEqual(model.max_out_len, 32)
            self.assertEqual(model.max_seq_len, 2048)
            self.assertEqual(model.generation_kwargs.do_sample, False)
            self.assertEqual(model.run_cfg.num_gpus, 1)
            self.assertTrue(all(d.infer_cfg.ice_template.type == 'scripts.week8_prompt_template.NonRecursivePromptTemplate' for d in parsed.datasets))
            self.assertFalse(binding['verified'])
            self.assertEqual(binding['model_path'], model.path)
            self.assertGreater(len(binding['dependencies']), 290)


if __name__ == '__main__':
    unittest.main()
