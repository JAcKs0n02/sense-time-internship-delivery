import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
SOURCE = ROOT / 'reports/week8/phase2_eval_resume/window2_failure'
RUN = Path('logs/formal-resume-20260917-01/original_base')


class RecoveryAuditTests(unittest.TestCase):
    def audit(self, source=SOURCE):
        import week8_score_recovery_audit as recovery
        digest = hashlib.sha256((source / 'manifest.json').read_bytes()).hexdigest()
        return recovery.audit(source, digest, ROOT)

    def test_real_failure_separates_valid_truncated_and_unattempted(self):
        result = self.audit()
        self.assertEqual(result['accepted_ids'], ['MATH-01', 'MATH-02', 'MATH-03', 'MATH-04'])
        self.assertEqual(result['truncated_ids'], ['MATH-05'])
        self.assertEqual(len(result['unattempted_ids']), 15)
        self.assertEqual(result['usage']['total_tokens'], 27922)
        self.assertFalse(result['formal_scoring_ready'])
        self.assertEqual(len(result['candidate_requests']), 20)
        self.assertTrue(all(x['body']['max_tokens'] == 8192 for x in result['candidate_requests']))

    def test_manifest_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / 'source'; shutil.copytree(SOURCE, dst)
            with (dst / RUN / 'answers.jsonl').open('a') as f: f.write('\n')
            with self.assertRaisesRegex(ValueError, 'hash|size'): self.audit(dst)

    def semantic_change(self, relative, change):
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / 'source'; shutil.copytree(SOURCE, dst)
            path = dst / RUN / relative
            value = json.loads(path.read_text()); change(value)
            path.write_text(json.dumps(value))
            m = dst / 'manifest.json'; manifest = json.loads(m.read_text())
            row = next(x for x in manifest['files'] if x['path'] == str(RUN / relative))
            row.update(size=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            m.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError): self.audit(dst)

    def test_changed_candidate_in_paid_request_rejected(self):
        def change(x):
            body = json.loads(x['request']['messages'][1]['content'])
            body['candidate_answer'] = 'altered answer'
            x['request']['messages'][1]['content'] = json.dumps(body)
        self.semantic_change('judge/MATH-01/attempt-1.json', change)

    def test_forged_accepted_score_rejected(self):
        self.semantic_change('judge/MATH-01/result.json', lambda x: x['scores'].update(weighted_score=0))

    def test_incorrect_usage_rejected(self):
        self.semantic_change('judge_usage.json', lambda x: x['usage'].update(total_tokens=1))

    def test_unmanifested_file_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / 'source'; shutil.copytree(SOURCE, dst)
            (dst / 'unexpected').write_text('x')
            with self.assertRaises(ValueError): self.audit(dst)


if __name__ == '__main__': unittest.main()
