import copy,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import step3_eval as ev
import week8_deepseek_judge as adapter

class JudgeIntegrationTests(unittest.TestCase):
    def profile(self):
        return json.loads((ROOT/'reports/week8/phase2_judge_calibration_v3/judge_profile_candidate.json').read_text())

    def test_explicit_prompt_matches_calibrated_wire_request(self):
        expected=json.loads((ROOT/'reports/week8/phase2_judge_calibration_v3/V3-01/attempt-1.json').read_text())['request']
        payload=json.loads(expected['messages'][1]['content'])
        self.assertIn('system_prompt',__import__('inspect').signature(adapter.make_request_body).parameters)
        actual=adapter.make_request_body(payload['question'],payload['candidate_answer'],payload['rubric'],system_prompt=expected['messages'][0]['content'])
        self.assertEqual(actual,expected)

    def test_missing_judge_profile_blocks_lock(self):
        self.assertTrue(callable(getattr(ev,'load_judge_profile',None)))
        with self.assertRaises(ValueError):ev.load_judge_profile({})

    def test_profile_drift_rejected(self):
        # Build a test-only current-code binding; never rewrite released profiles.
        profile=json.loads((ROOT/'configs/week8_judge_profile.json').read_text())
        profile['artifacts']={name:ev.sha256(ROOT/name) for name in profile['artifacts']}
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'current.json';p.write_text(json.dumps(profile))
            loaded=ev.load_judge_profile({'judge_profile':{'path':str(p),'sha256':ev.sha256(p)}})
            self.assertEqual(loaded['request_template'],self.profile()['request_template'])
            with self.assertRaises(ValueError):ev.load_judge_profile({'judge_profile':{'path':str(p),'sha256':'0'*64}})
            changed=copy.deepcopy(profile);changed['request_template']['max_tokens']=8192
            q=Path(tmp)/'profile.json';q.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):ev.load_judge_profile({'judge_profile':{'path':str(q),'sha256':ev.sha256(q)}})
            changed=copy.deepcopy(profile)
            changed['artifacts']['scripts/step3_eval.py']='0'*64
            q.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'code/data artifact changed'):
                ev.load_judge_profile({'judge_profile':{'path':str(q),'sha256':ev.sha256(q)}})

    def run_set(self, directory, raw_responses):
        def transport(question,answer,rubric,model,system_prompt=None):
            raw=raw_responses.pop(0)
            audit={'request':adapter.make_request_body(question,answer,rubric,system_prompt=system_prompt),'response':raw}
            try:return adapter.validate_response(raw,model),audit
            except ValueError as e:return None,{**audit,'validation_error':str(e),'retryable':isinstance(e,adapter.RetryableJudgmentError)}
        spec={'questions':[{'id':f'Q-{i}','instruction':'test'} for i in range(20)]}
        answers={q['id']:'test' for q in spec['questions']}
        with patch.object(adapter,'call_once',side_effect=transport),patch.object(ev,'judge_balance',return_value=10):
            return ev.score_custom20(spec,answers,{},self.profile(),directory,10)

    def test_all_20_results_and_usage_include_failed_attempt(self):
        self.assertTrue(callable(getattr(ev,'score_custom20',None)))
        good=json.loads((ROOT/'reports/week8/phase2_judge_calibration_v3/V3-01/attempt-1.json').read_text())['response']
        bad=json.loads((ROOT/'reports/week8/phase2_judge_calibration_v2/V2-06.json').read_text())['response']
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);rows=self.run_set(p,[bad]+[good]*20)
            self.assertEqual(len(rows),20)
            usage=json.loads((p/'judge_usage.json').read_text())
            self.assertEqual(usage['attempts'],21)
            self.assertEqual(usage['usage']['total_tokens'],bad['usage']['total_tokens']+20*good['usage']['total_tokens'])
            self.assertTrue(usage['usage_complete'])

    def test_double_failure_stops_without_summary(self):
        self.assertTrue(callable(getattr(ev,'score_custom20',None)))
        bad=json.loads((ROOT/'reports/week8/phase2_judge_calibration_v2/V2-06.json').read_text())['response']
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            with self.assertRaises(ValueError):self.run_set(p,[bad,bad])
            self.assertFalse((p/'summary.csv').exists())
            self.assertEqual(json.loads((p/'judge_usage.json').read_text())['attempts'],2)
            self.assertTrue((p/'judge/Q-0/attempt-2.json').exists())

if __name__=='__main__':unittest.main()
