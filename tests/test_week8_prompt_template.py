"""Regression tests require the actual OpenCompass 0.5.3 runtime."""
import copy
import csv
import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if os.environ.get('WEEK8_TEST_ORIGINAL_TEMPLATE') == '1':
    from opencompass.openicl.icl_prompt_template import PromptTemplate as Template
else:
    from scripts.week8_prompt_template import NonRecursivePromptTemplate as Template

QUESTION = '{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}'
META = dict(round=[dict(role='HUMAN', prompt=QUESTION),
                       dict(role='BOT', prompt='{answer}')])


class PromptTemplateTests(unittest.TestCase):
    def test_all_eleven_observed_corruptions(self):
        hits = json.loads((ROOT / 'reports/week8/phase2_gpu_monitor/recursive_placeholder_hits.json').read_text())
        self.assertEqual(len(hits), 11)
        for hit in hits:
            dataset, subject, split, index = hit['id'].split(':')
            path = ROOT / 'reports/week8/phase2_preflight/local_data' / dataset / split / (subject + '.csv')
            # C-Eval filenames include the split suffix.
            if dataset == 'ceval':
                path = path.with_name(subject + '_' + split + '.csv')
            with path.open() as f:
                row = list(csv.DictReader(f))[int(index)]
            if dataset == 'cmmlu':
                row['question'], row['answer'] = row['Question'], row['Answer']
            row = {key: row[key] for key in ('question', 'A', 'B', 'C', 'D', 'answer')}
            before = copy.deepcopy(row)
            actual = Template(copy.deepcopy(META)).generate_item(row, output_field='answer')
            with self.subTest(record=hit['id']):
                self.assertEqual([p['prompt'] for p in actual if isinstance(p, dict) and 'prompt' in p], [hit['intended'], ''])
                self.assertEqual(row, before)

    def test_values_are_literal_and_answer_is_removed(self):
        meta = dict(begin='<ICE>', round=[dict(role='HUMAN', prompt='{question}|{question}|{missing}'),
                           dict(role='BOT', prompt='{answer}')])
        row = dict(question=r'\\oint_{C} {answer} {question}', C='WRONG', answer='SECRET')
        original = copy.deepcopy(meta)
        template = Template(meta, ice_token='<ICE>')
        result = template.generate_item(row, output_field='answer')
        prompts = [p['prompt'] for p in result if isinstance(p, dict) and 'prompt' in p]
        self.assertEqual(prompts, [row['question'] + '|' + row['question'] + '|{missing}', ''])
        ice = template.generate_ice_item(row, 'SECRET')
        self.assertEqual([p['prompt'] for p in ice if isinstance(p, dict) and 'prompt' in p], [prompts[0], 'SECRET'])
        self.assertEqual(meta, original)
        self.assertEqual(row['answer'], 'SECRET')

    def test_inserted_ice_is_never_formatted_again(self):
        from opencompass.utils.prompt import PromptList
        template = Template(dict(begin='<ICE>', **copy.deepcopy(META)), ice_token='<ICE>')
        shot = dict(question='{A} {answer} <ICE>', A='a', B='b', C='c', D='d', answer='A')
        ice = template.generate_ice_item(shot, 'A')
        before = copy.deepcopy(ice)
        row = dict(question='test <ICE>', A='WRONG', B='b', C='c', D='d', answer='B')
        actual = template.generate_item(row, output_field='answer', ice_field_replace_token=ice)
        prompts = [p['prompt'] for p in actual if isinstance(p, dict) and 'prompt' in p]
        self.assertEqual(prompts, [QUESTION.format(**shot), 'A', QUESTION.format(**row), ''])
        self.assertEqual(ice, before)
        self.assertIsInstance(actual, PromptList)

    def test_scope_rejects_string_templates(self):
        with self.assertRaisesRegex(ValueError, 'meta'):
            Template('{question}')


if __name__ == '__main__':
    unittest.main()
