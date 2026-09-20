import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import week8_deepseek_judge as judge


def fixture(name):
    return json.loads((ROOT/'reports/week8/phase2_judge_calibration_v2'/f'{name}.json').read_text())['response']


class JudgeHardeningTests(unittest.TestCase):
    def test_real_bad_responses_rejected(self):
        for name in ('V2-06','V2-08','V2-12'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                judge.validate_response(fixture(name),judge.MODEL)

    def test_strict_nested_schema_and_duplicates(self):
        good=fixture('V2-09')
        content=json.loads(good['choices'][0]['message']['content'])
        variants=[]
        for value in (True,'5',None):
            bad=copy.deepcopy(content);bad['accuracy']['score']=value;variants.append(json.dumps(bad))
        bad=copy.deepcopy(content);bad['accuracy']['extra']='x';variants.append(json.dumps(bad))
        bad=copy.deepcopy(content);bad['accuracy']['reason']=123;variants.append(json.dumps(bad))
        bad=copy.deepcopy(content);bad.pop('format');variants.append(json.dumps(bad))
        variants.append(json.dumps(content).replace('"score": 5','"score": 0, "score": 5',1))
        for text in variants:
            raw=copy.deepcopy(good);raw['choices'][0]['message']['content']=text
            with self.subTest(text=text[:90]),self.assertRaises(ValueError):judge.validate_response(raw,judge.MODEL)

    def run_bounded(self, responses, path):
        def send(*args):
            raw=responses.pop(0)
            if isinstance(raw,Exception):raise raw
            audit={'request':{'model':judge.MODEL},'response':raw}
            try:return judge.validate_response(raw,judge.MODEL),audit
            except ValueError as exc:return None,{**audit,'validation_error':str(exc),'retryable':isinstance(exc,judge.RetryableJudgmentError)}
        with patch.object(judge,'call_once',side_effect=send):
            return judge.call_bounded({},'answer',{},judge.MODEL,path)

    def test_retry_saves_invalid_attempt_and_accounts_all_usage(self):
        self.assertTrue(callable(getattr(judge,'call_bounded',None)),'bounded audited runner missing')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'case';raws=[fixture('V2-06'),fixture('V2-09')]
            total=sum(x['usage']['total_tokens'] for x in raws)
            result=self.run_bounded(raws,path)
            self.assertEqual(result['status'],'accepted')
            self.assertEqual(result['attempts'],2)
            self.assertEqual(result['usage']['total_tokens'],total)
            self.assertEqual(json.loads((path/'attempt-1.json').read_text())['response'],fixture('V2-06'))
            self.assertEqual(result['scores']['accuracy'],5)
            with self.assertRaises(FileExistsError):self.run_bounded([],path)

    def test_stops_after_two_bad_responses(self):
        self.assertTrue(callable(getattr(judge,'call_bounded',None)))
        with tempfile.TemporaryDirectory() as tmp:
            raws=[fixture('V2-06'),fixture('V2-08'),fixture('V2-09')]
            result=self.run_bounded(raws,Path(tmp)/'case')
            self.assertEqual(result['status'],'rejected');self.assertEqual(len(raws),1)
            self.assertIsNone(result['scores'])

    def test_identity_or_transport_failure_never_retried(self):
        self.assertTrue(callable(getattr(judge,'call_bounded',None)))
        bad=fixture('V2-09');bad['model']='unknown'
        with tempfile.TemporaryDirectory() as tmp:
            result=self.run_bounded([bad],Path(tmp)/'identity')
            self.assertEqual(result['attempts'],1);self.assertEqual(result['status'],'rejected')
            path=Path(tmp)/'transport'
            with self.assertRaises(TimeoutError):self.run_bounded([TimeoutError('secret-like text')],path)
            saved=(path/'attempt-1-error.json').read_text()
            self.assertNotIn('secret-like text',saved)
            self.assertTrue((path/'attempt-1.started').exists())

if __name__=='__main__':unittest.main()
