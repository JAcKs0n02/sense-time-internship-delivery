import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import compare_distillation as comparison

ARCHIVE = ROOT / 'reports/week8/phase2_trained_generation/retrieved/logs/trained-generation-20260917'


class ComparisonTests(unittest.TestCase):
    def summary(self, directory):
        self.assertTrue(hasattr(comparison, 'summarize_ceval'), 'strict CEval summary missing')
        return comparison.summarize_ceval(directory, ROOT)

    def test_real_results_count_questions_not_metadata(self):
        for model, correct in [('final_sft', 1072), ('final_dpo', 1077)]:
            with self.subTest(model=model):
                result = self.summary(ARCHIVE / model / 'opencompass')
                self.assertEqual(result['subjects'], 52)
                self.assertEqual(result['questions'], 1346)
                self.assertEqual(result['correct'], correct)
                self.assertAlmostEqual(result['accuracy'], correct / 1346 * 100)

    def test_missing_duplicate_and_tampered_results_fail(self):
        self.assertTrue(hasattr(comparison, 'summarize_ceval'), 'strict CEval summary missing')
        source = ARCHIVE / 'final_sft/opencompass'
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'oc'
            for original in source.glob('*/results/*/ceval-*.json'):
                target = out / original.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original, target)
            target = next(out.glob('*/results/*/ceval-*.json'))
            original = target.read_text()
            for change in ['missing', 'duplicate', 'gold', 'accuracy']:
                with self.subTest(change=change):
                    if change == 'missing':
                        target.unlink()
                    elif change == 'duplicate':
                        duplicate = out / 'other/results/model' / target.name
                        duplicate.parent.mkdir(parents=True)
                        duplicate.write_text(original)
                    else:
                        payload = json.loads(original)
                        if change == 'gold':
                            payload['details']['0']['references'] = 'INVALID'
                        else:
                            payload['accuracy'] = 0
                        target.write_text(json.dumps(payload))
                    with self.assertRaises(ValueError):
                        self.summary(out)
                    target.write_text(original)
                    if change == 'duplicate':
                        shutil.rmtree(out / 'other')

    def test_speed_rejects_short_output_or_invalid_timing(self):
        self.assertTrue(hasattr(comparison, 'validate_speed'), 'speed validation missing')
        valid = {'warmups': 2, 'dtype': 'bfloat16', 'includes_prefill': True,
                 'runs': [{'id': str(i), 'tokens': 128, 'seconds': 2.0} for i in range(10)]}
        self.assertEqual(comparison.validate_speed(valid)['tokens_per_second'], 64.0)
        for kind in ['short', 'zero', 'nan', 'duplicate', 'missing']:
            value = copy.deepcopy(valid)
            if kind == 'short': value['runs'][0]['tokens'] = 127
            if kind == 'zero': value['runs'][0]['seconds'] = 0
            if kind == 'nan': value['runs'][0]['seconds'] = float('nan')
            if kind == 'duplicate': value['runs'][0]['id'] = '1'
            if kind == 'missing': value['runs'].pop()
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                comparison.validate_speed(value)

    def test_config_uses_student_tokenizer_and_only_frozen_ceval(self):
        self.assertTrue(hasattr(comparison, 'build_ceval_config'), 'explicit CEval config missing')
        cfg, binding = comparison.build_ceval_config(ROOT, '/tmp/student-before')
        self.assertEqual(len(cfg['datasets']), 52)
        self.assertTrue(all(d['abbr'].startswith('ceval-') for d in cfg['datasets']))
        self.assertEqual(cfg['models'][0]['path'], '/tmp/student-before')
        self.assertEqual(cfg['models'][0]['tokenizer_path'], '/tmp/student-before')
        self.assertFalse(cfg['models'][0]['tokenizer_only'])
        self.assertFalse(binding['verified'])
        for dataset in cfg['datasets']:
            self.assertEqual(dataset['reader_cfg']['test_split'], 'val')
            self.assertEqual(dataset['infer_cfg']['retriever']['fix_id_list'], [0, 1, 2, 3, 4])

    def test_model_manifest_detects_changed_or_extra_files(self):
        self.assertTrue(hasattr(comparison, 'verify_model_manifest'), 'model identity check missing')
        from common import sha256
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / 'model'
            model.mkdir()
            for name, text in {'config.json': '{}', 'tokenizer.json': '{}',
                               'tokenizer_config.json': '{}', 'generation_config.json': '{}',
                               'model.safetensors': 'test fixture, not valid tensors'}.items():
                (model / name).write_text(text)
            files = [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': sha256(p)}
                     for p in sorted(model.iterdir())]
            manifest = Path(tmp) / 'manifest.json'
            manifest.write_text(json.dumps({'model_dir': str(model), 'files': files}))
            comparison.verify_model_manifest(model, manifest)
            (model / 'model.safetensors').write_text('changed')
            with self.assertRaises(ValueError):
                comparison.verify_model_manifest(model, manifest)
            (model / 'model.safetensors').write_text('test fixture, not valid tensors')
            (model / 'adapter_model.safetensors').write_text('unbound adapter')
            with self.assertRaises(ValueError):
                comparison.verify_model_manifest(model, manifest)

    def test_cli_requires_identities_before_any_execution(self):
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'untouched'
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/compare_distillation.py'),
                                     '--before', '/tmp/student', '--after', '/tmp/student-after',
                                     '--output-dir', str(output)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('--before-manifest', result.stderr)
            self.assertFalse(output.exists())

    def test_timed_command_stops_instead_of_waiting_forever(self):
        self.assertTrue(hasattr(comparison, 'run_bounded'), 'bounded subprocess runner missing')
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(subprocess.TimeoutExpired):
                comparison.run_bounded([sys.executable, '-c', 'import time; time.sleep(10)'],
                                       Path(tmp) / 'log', 0.1)

    def test_speed_pair_rejects_changed_device_or_tokenized_prompts(self):
        self.assertTrue(hasattr(comparison, 'validate_speed_pair'), 'paired speed protocol check missing')
        value = {'environment': {'gpu_uuid': 'test-gpu', 'torch': 'test-version'},
                 'prompt_source_sha256': 'fixture-source',
                 'prompt_token_sha256': ['fixture-' + str(i) for i in range(12)]}
        comparison.validate_speed_pair(value, copy.deepcopy(value))
        for key in ['environment', 'prompt_token_sha256', 'prompt_source_sha256']:
            changed = copy.deepcopy(value)
            changed[key] = None
            with self.subTest(key=key), self.assertRaises(ValueError):
                comparison.validate_speed_pair(value, changed)


if __name__ == '__main__':
    unittest.main()
