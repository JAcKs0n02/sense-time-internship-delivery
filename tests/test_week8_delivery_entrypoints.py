"""User-facing commands must accept reviewed data and plan fresh Gemini evaluation.

These CPU checks do not claim that model training or inference has run.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/week8/prepared'


class DeliveryEntrypointTests(unittest.TestCase):
    def run_command(self, *args):
        return subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                              env={**os.environ, 'PIPELINE_PYTHON': sys.executable})

    def test_reviewed_formal_data_produces_portable_training_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'training'
            result = self.run_command('bash', str(ROOT/'scripts/step2_train.sh'),
                '--data-dir', str(DATA), '--run-dir', str(out),
                '--base-model', '/models/Qwen2.5-7B-Instruct', '--dry-run')
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads((out/'planned_configs.json').read_text())
            self.assertEqual(plan['sft']['num_train_epochs'], 5)
            self.assertEqual(plan['sft']['dataset_dir'], str(DATA))
            self.assertEqual(plan['dpo']['model_name_or_path'], str(out.resolve()/'models/final_sft'))
            self.assertEqual(plan['dpo']['max_steps'], 40)
            self.assertFalse(json.loads((out/'status.json').read_text())['trained'])

    def test_master_dry_run_prepares_data_and_current_eval_without_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'pipeline'
            env = {**os.environ, 'PIPELINE_PYTHON': sys.executable,
                   'EVAL_MODEL_PATH': '/models/final_dpo'}
            for name in ('JUDGE_MODEL', 'JUDGE_BASE_URL', 'DATA_INPUT', 'TOKENIZER_PATH'):
                env.pop(name, None)
            result = subprocess.run(['bash', str(ROOT/'run_pipeline.sh'), '--skip-train',
                '--dry-run', '--eval-limit-per-subject', '1', '--run-dir', str(out)],
                capture_output=True, text=True, env=env)
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads((out/'evaluation/plan.json').read_text())
            self.assertEqual(plan['benchmark_questions'], 119)
            self.assertEqual(plan['custom_questions'], 20)
            self.assertEqual(plan['judge_model'], 'gemini-3.1-pro-preview')
            self.assertEqual(plan['coverage'], 'sample')
            self.assertFalse((out/'training').exists())
            self.assertEqual(json.loads((out/'data/statistics.json').read_text())['mode'],
                             'protocol_formal_tokenizer')
            status = json.loads((out/'evaluation/status.json').read_text())
            self.assertFalse(status['fresh_model_inference'])
            self.assertEqual(status['new_api_calls'], 0)

    def test_training_rejects_modified_formal_data_before_creating_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)/'data'
            shutil.copytree(DATA, data)
            (data/'train_sharegpt.json').write_text('[]')
            out = Path(tmp)/'training'
            result = self.run_command('bash', str(ROOT/'scripts/step2_train.sh'),
                '--data-dir', str(data), '--run-dir', str(out),
                '--base-model', '/models/Qwen2.5-7B-Instruct', '--dry-run')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('formal data reproduction differs', result.stderr)
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()


@pytest.fixture
def sample_generation(tmp_path):
    """Real configuration/validators; synthetic model output, no GPU or network."""
    sys.path.insert(0, str(ROOT/'scripts'))
    from common import load_module, write_json, sha256
    m = load_module('delivery_eval_entry', ROOT/'scripts/pipeline/step3_eval.py')
    out = tmp_path/'evaluation'
    plan = m.build_plan(Path('/models/final_dpo'), out, 1)
    generation = out/'generation'
    results = generation/'opencompass/run/results/model'
    results.mkdir(parents=True)
    for r in json.loads((out/'benchmark_records.json').read_text()):
        if r['split'] != ('val' if r['benchmark'] == 'ceval' else 'test'):
            continue
        correct = r['benchmark'] == 'ceval'
        pred = r['answer'] if correct else next(a for a in 'ABCD' if a != r['answer'])
        write_json(results/(r['benchmark']+'-'+r['subject']+'.json'),
            {'accuracy': 100 if correct else 0,
             'details': {'0': {'predictions': pred, 'references': r['answer'], 'correct': correct}}})
    write_json(generation/'benchmarks.json', m.benchmark_rows(out, plan))
    spec = json.loads((ROOT/m.shared.QUESTIONS).read_text())
    answers = [{'id': q['id'], 'answer': 'synthetic test '+q['id'],
                'finish_reason': 'eos', 'generated_tokens': 8} for q in spec['questions']]
    (generation/'answers.jsonl').write_text(''.join(json.dumps(a)+'\n' for a in answers))
    files = {str(p.relative_to(generation)): sha256(p) for p in generation.rglob('*') if p.is_file()}
    write_json(generation/'generation_receipt.json', {'files': files,
        'plan_sha256': sha256(out/'plan.json'), 'judge_calls': 0})
    return m, out, plan


def test_current_scoring_writes_correct_csv_and_reuses_completed_responses(sample_generation, tmp_path):
    import csv
    m, out, plan = sample_generation
    real_run = m.scoring.run
    calls = []

    def fixture_api(body):
        payload = json.loads(body['contents'][0]['parts'][0]['text'])
        qid = payload['candidate_answer'].removeprefix('synthetic test ')
        calls.append(qid)
        return json.loads((ROOT/'reports/week8/phase2_gemini_score_original_base/judge'/qid/'response.json').read_text())['response']

    def offline_run(bound, output, claims):
        return real_run(bound, output, tmp_path/'claims', call=fixture_api)

    with patch.object(m.scoring, 'run', side_effect=offline_run):
        m.score(out, plan)
        with (out/'summary.csv').open() as f:
            rows = list(csv.DictReader(f))
        assert [r['dataset'] for r in rows] == ['ceval', 'cmmlu', 'custom20']
        assert [int(r['questions']) for r in rows] == [52, 67, 20]
        assert [float(r['score']) for r in rows] == [100.0, 0.0, 3.9775]
        assert rows[0]['evidence_kind'] == 'fresh_opencompass_sample'
        assert len(calls) == 20
        m.score(out, plan)
        assert len(calls) == 20
        assert json.loads((out/'status.json').read_text())['new_api_calls'] == 0


def test_changed_answers_rejected_before_any_scoring(sample_generation):
    m, out, plan = sample_generation
    with (out/'generation/answers.jsonl').open('a') as f:
        f.write('{}\n')
    with patch.object(m.scoring, 'run', side_effect=AssertionError('must not score')):
        with pytest.raises(ValueError, match='generation artifact changed'):
            m.score(out, plan)
    assert not (out/'scoring').exists()


def test_sample_subject_coverage_cannot_be_incomplete(sample_generation):
    m, out, plan = sample_generation
    next((out/'generation/opencompass').glob('*/results/*/ceval-*.json')).unlink()
    with pytest.raises(ValueError, match='incomplete or duplicate subject coverage'):
        m.benchmark_rows(out, plan)
