import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


class PipelineTests(unittest.TestCase):
    def test_data_entry_and_split(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            rows = [{'instruction': f'独立问题{i}', 'input': '', 'output': f'<p>答案{i}</p>'} for i in range(20)]
            rows.append(dict(rows[0]))
            rows.append({'instruction': '独立问题0', 'input': '', 'output': '另一个正确答案'})
            rows.append({'instruction': '', 'output': ''})
            (d/'input.json').write_text(json.dumps(rows))
            result = subprocess.run([sys.executable, str(ROOT/'scripts/step1_data_prep.py'), '--input', str(d/'input.json'), '--output-dir', str(d/'out'), '--quick'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            train = json.loads((d/'out/train_alpaca.json').read_text())
            val = json.loads((d/'out/validation_alpaca.json').read_text())
            self.assertFalse({x['instruction'] for x in train} & {x['instruction'] for x in val})
            self.assertEqual(len(train)+len(val), 21)
            self.assertTrue(all('<p>' not in x['output'] for x in train+val))
            stats = json.loads((d/'out/statistics.json').read_text())
            self.assertEqual(stats['mode'], 'quick_character_smoke')
            self.assertEqual(stats['prompt_overlap'], 0)

    def test_oom_reduction_preserves_effective_batch(self):
        import train_pipeline
        reduced = train_pipeline.reduce_batch({'per_device_train_batch_size': 4, 'gradient_accumulation_steps': 2})
        self.assertEqual(reduced['per_device_train_batch_size'], 2)
        self.assertEqual(reduced['gradient_accumulation_steps'], 4)
        with self.assertRaises(ValueError):
            train_pipeline.reduce_batch({'per_device_train_batch_size': 1})

    def test_judge_rejects_bad_and_nonfinite_scores(self):
        import step3_eval
        good = {k: {'score': 4, 'reason': '逐条依据'} for k in step3_eval.WEIGHTS}
        self.assertEqual(step3_eval.validate_judgment(good)['weighted_score'], 4)
        for bad in [float('nan'), 6, -1, True]:
            broken = dict(good, accuracy={'score': bad, 'reason': '依据'})
            with self.assertRaises(ValueError):
                step3_eval.validate_judgment(broken)

    def test_unknown_pipeline_option_fails(self):
        result = subprocess.run(['bash', str(ROOT/'run_pipeline.sh'), '--typo'], capture_output=True)
        self.assertNotEqual(result.returncode, 0)

    def test_training_non_oom_failure_is_not_retried(self):
        import train_pipeline
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            cli = d/'fake-cli'
            cli.write_text('#!/bin/sh\necho dataset schema invalid\nexit 7\n')
            cli.chmod(0o755)
            with self.assertRaisesRegex(RuntimeError, 'without OOM'):
                train_pipeline.execute_train({}, d/'run', str(cli))
            self.assertFalse((d/'run/attempt-02').exists())
            self.assertEqual(json.loads((d/'run/attempt-01/status.json').read_text())['returncode'], 7)

    def test_training_zero_exit_without_artifact_fails(self):
        import train_pipeline
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            cli = d/'fake-cli'; cli.write_text('#!/bin/sh\nexit 0\n'); cli.chmod(0o755)
            with self.assertRaisesRegex(RuntimeError, 'artifacts are missing'):
                train_pipeline.execute_train({}, d/'run', str(cli))

    def test_health_check_rejects_dead_child(self):
        import deploy
        process = subprocess.Popen([sys.executable, '-c', 'raise SystemExit(9)'])
        process.wait()
        with self.assertRaisesRegex(RuntimeError, 'service exited'):
            deploy.wait_http('http://127.0.0.1:1/', process, 2)

    def test_training_refuses_smoke_data(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            (d/'statistics.json').write_text('{"mode":"quick_character_smoke"}')
            result = subprocess.run([sys.executable, str(ROOT/'scripts/train_pipeline.py'), '--data-dir', str(d), '--run-dir', str(d/'run'), '--base-model', '/unused', '--dry-run'],capture_output=True,text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('refuses quick', result.stderr)

    def test_quick_has_real_artifact_and_honest_status(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(['bash', str(ROOT/'run_pipeline.sh'), '--quick', '--run-dir', directory], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            status = json.loads((Path(directory)/'evaluation/status.json').read_text())
            self.assertEqual(status['mode'], 'archived_evidence_replay')
            self.assertFalse(status['fresh_model_inference'])


if __name__ == '__main__':
    unittest.main()
