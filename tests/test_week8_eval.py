import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import step3_eval as evaluation

class EvaluationTests(unittest.TestCase):
    def test_metadata_is_not_a_question(self):
        value = {'accuracy': 50, 'details': {'type': 'GEN', '0': {'predictions': 'A', 'references': 'A'}, '1': {'predictions': '', 'references': 'B'}}}
        self.assertEqual(evaluation.validate_subject_result(value, ['A', 'B']), (2, 1))

    def test_legacy_scalar_details(self):
        value = {'accuracy': 100, 'details': {'0': {'pred': 'B', 'refr': 'B', 'is_correct': True}}}
        self.assertEqual(evaluation.validate_subject_result(value, ['B']), (1, 1))
        value['details']['0']['is_correct'] = False
        with self.assertRaises(ValueError):
            evaluation.validate_subject_result(value, ['B'])

    def native_list(self):
        return {'accuracy': 50, 'details': [
            {'example_abbr': 'cmmlu-physics_test_0', 'pred': ['A'], 'refr': ['A'], 'is_correct': [True]},
            {'example_abbr': 'cmmlu-physics_test_1', 'pred': ['C'], 'refr': ['B'], 'is_correct': [False]},
        ]}

    def test_actual_cmmlu_list_details(self):
        value = self.native_list()
        original = copy.deepcopy(value)
        self.assertEqual(evaluation.validate_subject_result(value, ['A', 'B'], subject_id='cmmlu-physics'), (2, 1))
        self.assertEqual(value, original)

    def test_native_list_requires_subject_binding(self):
        with self.assertRaisesRegex(ValueError, 'subject'):
            evaluation.validate_subject_result(self.native_list(), ['A', 'B'])

    def test_rejects_corrupt_native_lists(self):
        for mutation in ('missing', 'extra', 'reordered', 'duplicate_id', 'wrong_subject',
                         'empty_pred', 'multiple_pred', 'scalar_ref', 'wrong_gold',
                         'false_flag', 'integer_flag', 'missing_flag', 'false_accuracy'):
            value = self.native_list()
            rows = value['details']
            if mutation == 'missing': rows.pop()
            if mutation == 'extra': rows.append(copy.deepcopy(rows[-1]))
            if mutation == 'reordered': rows.reverse()
            if mutation == 'duplicate_id': rows[1]['example_abbr'] = rows[0]['example_abbr']
            if mutation == 'wrong_subject': rows[0]['example_abbr'] = 'cmmlu-biology_test_0'
            if mutation == 'empty_pred': rows[0]['pred'] = []
            if mutation == 'multiple_pred': rows[0]['pred'] = ['A', 'B']
            if mutation == 'scalar_ref': rows[0]['refr'] = 'A'
            if mutation == 'wrong_gold': rows[0]['refr'] = ['B']
            if mutation == 'false_flag': rows[0]['is_correct'] = [False]
            if mutation == 'integer_flag': rows[0]['is_correct'] = [1]
            if mutation == 'missing_flag': rows[0].pop('is_correct')
            if mutation == 'false_accuracy': value['accuracy'] = 100
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                evaluation.validate_subject_result(value, ['A', 'B'], subject_id='cmmlu-physics')

    def test_rejects_missing_extra_shifted_answers_and_false_accuracy(self):
        original = {'accuracy': 100, 'details': {'0': {'predictions': 'A', 'references': 'A'}}}
        for mutation in ('missing', 'extra', 'shifted', 'answer', 'accuracy', 'metadata'):
            value = copy.deepcopy(original)
            if mutation == 'missing': value['details'].clear()
            if mutation == 'extra': value['details']['1'] = value['details']['0']
            if mutation == 'shifted': value['details']['1'] = value['details'].pop('0')
            if mutation == 'answer': value['details']['0']['references'] = 'B'
            if mutation == 'accuracy': value['accuracy'] = 0
            if mutation == 'metadata': value['details']['unexpected'] = 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                evaluation.validate_subject_result(value, ['A'])

    def test_fresh_blocked_before_subprocess(self):
        args = SimpleNamespace(model='unused', judge_model='unused', judge_url='https://example.com', output_dir=Path('/unused'))
        with patch.object(evaluation.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'runtime lock'):
                evaluation.fresh(args)
            run.assert_not_called()

if __name__ == '__main__': unittest.main()
