import copy
from decimal import Decimal
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
RAW = json.loads((ROOT / 'reports/week8/phase2_judge_calibration_v4/PILOT-MATH-05-raw.json').read_text())


class ScoreOnlyTests(unittest.TestCase):
    def plan(self):
        return {'identity': 'same-answer-protocol', 'requests': [
            {'question_id': f'Q{i}', 'body': copy.deepcopy(RAW['request'])} for i in range(20)],
            'max_calls': 20, 'reserve_cny': '1', 'max_spend_cny': '1'}

    def test_success_and_completed_rerun_do_not_rebill(self):
        from week8_score_only import run
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'out'; calls = []
            def call(body): calls.append(body); return copy.deepcopy(RAW['response'])
            run(self.plan(), output, Path(tmp)/'claims', call, lambda: Decimal('9'))
            self.assertEqual(len(calls), 20)
            run(self.plan(), output, Path(tmp)/'claims', call, lambda: Decimal('9'))
            self.assertEqual(len(calls), 20)
            self.assertEqual(json.loads((output/'summary.json').read_text())['questions'], 20)
            with self.assertRaises(ValueError): run(self.plan(), Path(tmp)/'other', Path(tmp)/'claims', call, lambda: Decimal('9'))
            self.assertEqual(len(calls), 20)

    def test_truncation_preserves_usage_stops_and_blocks_replay(self):
        from week8_score_only import run
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'out'; calls=[]
            def call(body):
                calls.append(body); raw=copy.deepcopy(RAW['response']);raw['choices'][0]['finish_reason']='length';return raw
            with self.assertRaises(ValueError): run(self.plan(), out, Path(tmp)/'claims', call, lambda: Decimal('9'))
            self.assertEqual(len(calls), 1)
            self.assertEqual(json.loads((out/'usage.json').read_text())['attempts'], 1)
            self.assertTrue((out/'judge/Q0/response.json').exists())
            with self.assertRaises(ValueError): run(self.plan(), out, Path(tmp)/'claims', call, lambda: Decimal('9'))
            self.assertEqual(len(calls), 1)

    def test_unknown_transport_never_replayed(self):
        from week8_score_only import run
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'out';calls=[]
            def call(body): calls.append(body);raise TimeoutError('private text must not persist')
            with self.assertRaises(TimeoutError): run(self.plan(),out,Path(tmp)/'claims',call,lambda:Decimal('9'))
            self.assertFalse(json.loads((out/'usage.json').read_text())['usage_complete'])
            self.assertNotIn('private text', (out/'status.json').read_text())
            with self.assertRaises(ValueError):run(self.plan(),out,Path(tmp)/'claims',call,lambda:Decimal('9'))
            self.assertEqual(len(calls),1)

    def test_budget_refuses_before_paid_marker(self):
        from week8_score_only import run
        with tempfile.TemporaryDirectory() as tmp:
            calls=[];out=Path(tmp)/'out'
            with self.assertRaises(ValueError):run(self.plan(),out,Path(tmp)/'claims',lambda b:calls.append(b),lambda:Decimal('0.5'))
            self.assertEqual(calls,[])
            self.assertEqual(list(out.glob('judge/*/started.json')),[])

    def test_valid_durable_response_recovers_without_rebill(self):
        from week8_score_only import run
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'out';claims=Path(tmp)/'claims';calls=[]
            def call(body):calls.append(body);return copy.deepcopy(RAW['response'])
            run(self.plan(),out,claims,call,lambda:Decimal('9'))
            # Simulate an interruption after raw response but before derived output.
            (out/'judge/Q19/result.json').unlink()
            (out/'status.json').write_text(json.dumps({'status':'RUNNING'}))
            run(self.plan(),out,claims,call,lambda:Decimal('9'))
            self.assertEqual(len(calls),20)
            self.assertTrue((out/'judge/Q19/result.json').exists())

    def test_changed_request_or_saved_score_blocks_reuse(self):
        from week8_score_only import run
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'out';claims=Path(tmp)/'claims';calls=[]
            def call(body):calls.append(body);return copy.deepcopy(RAW['response'])
            run(self.plan(),out,claims,call,lambda:Decimal('9'))
            result=out/'judge/Q0/result.json';value=json.loads(result.read_text());value['weighted_score']=0;result.write_text(json.dumps(value))
            with self.assertRaises(ValueError):run(self.plan(),out,claims,call,lambda:Decimal('9'))
            self.assertEqual(len(calls),20)

    def test_deleted_paid_directory_cannot_be_reissued(self):
        import shutil
        from week8_score_only import run
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'out';claims=Path(tmp)/'claims';calls=[]
            def call(body):calls.append(body);return copy.deepcopy(RAW['response'])
            run(self.plan(),out,claims,call,lambda:Decimal('9'))
            shutil.rmtree(out/'judge/Q0')
            with self.assertRaises(ValueError):run(self.plan(),out,claims,call,lambda:Decimal('9'))
            self.assertEqual(len(calls),20)

    def test_real_release_plan_is_bound_to_calibrated_requests(self):
        from week8_score_only import prepare
        p=prepare(ROOT/'reports/week8/phase2_eval_resume/window2_failure',
            '22665b1f4040f239aeb258367a9a1552520f5399ce2fb8b827b3747aaa878a22',
            ROOT/'reports/week8/phase2_judge_calibration_v4')
        self.assertEqual(len(p['requests']),20)
        self.assertFalse(p['historical_scores_reused'])
        self.assertTrue(p['generation_reused'])


if __name__=='__main__':unittest.main()
