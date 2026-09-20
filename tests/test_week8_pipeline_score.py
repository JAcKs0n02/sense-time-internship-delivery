import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))

class PipelineScoreTests(unittest.TestCase):
    def test_entry_exists(self):
        self.assertIsNotNone(importlib.util.find_spec('week8_pipeline_score'))

    def test_main_entry_verifies_all_without_preparing_data_or_api(self):
        with tempfile.TemporaryDirectory() as t:
            out=Path(t)/'run'
            proc=subprocess.run(['bash',str(ROOT/'run_pipeline.sh'),'--score-only','all','--run-dir',str(out)],capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,proc.stderr)
            result=json.loads((out/'score_recovery.json').read_text())
            self.assertEqual(result['status'],'VERIFIED_EXISTING_SCORES')
            self.assertEqual(result['new_api_calls'],0)
            self.assertEqual([r['weighted_mean'] for r in result['models']],[3.9775,3.8275,3.87])
            self.assertFalse((out/'data').exists())
            self.assertFalse((out/'training').exists())

    def test_incompatible_flags_fail_before_work(self):
        for extra in [['--quick'],['--deploy'],['--skip-eval'],['--skip-train'],['--execute-score']]:
            with self.subTest(extra=extra),tempfile.TemporaryDirectory() as t:
                out=Path(t)/'run'
                proc=subprocess.run(['bash',str(ROOT/'run_pipeline.sh'),'--score-only','all',*extra,'--run-dir',str(out)],capture_output=True,text=True)
                self.assertNotEqual(proc.returncode,0)
                self.assertFalse(out.exists())

    def test_empty_score_model_fails_before_data_preparation(self):
        with tempfile.TemporaryDirectory() as t:
            out=Path(t)/'run'
            proc=subprocess.run(['bash',str(ROOT/'run_pipeline.sh'),'--score-only','','--run-dir',str(out)],capture_output=True,text=True)
            self.assertEqual(proc.returncode,2)
            self.assertFalse(out.exists())

    def test_completed_execute_never_calls_paid_runner(self):
        import week8_pipeline_score as w
        with tempfile.TemporaryDirectory() as t,patch.object(w.scoring,'run',side_effect=AssertionError('must not call paid runner')):
            result=w.integrate('final_sft',Path(t)/'run',execute=True)
            self.assertEqual(result['new_api_calls'],0)

    def test_tampered_score_rejected(self):
        import week8_pipeline_score as w
        plan=w.load_release('original_base')
        with tempfile.TemporaryDirectory() as t:
            out=Path(t)/'copy';shutil.copytree(w.score_directory('original_base'),out)
            p=out/'judge/MATH-01/result.json';d=json.loads(p.read_text());d['weighted_score']=0;p.write_text(json.dumps(d))
            with self.assertRaises(ValueError):w.verify_completed(plan,out)

    def test_unknown_paid_attempt_is_not_called_complete(self):
        import week8_pipeline_score as w
        plan=w.load_release('original_base')
        with tempfile.TemporaryDirectory() as t:
            out=Path(t)/'copy';shutil.copytree(w.score_directory('original_base'),out)
            (out/'judge/MATH-01/response.json').unlink()
            with self.assertRaises(ValueError):w.verify_completed(plan,out)

    def test_wrong_plan_hash_rejected(self):
        import week8_pipeline_score as w
        with patch.dict(w.RELEASES,{'original_base':(w.RELEASES['original_base'][0],'0'*64)}):
            with self.assertRaisesRegex(ValueError,'release hash'):w.load_release('original_base')

    def test_report_cannot_modify_frozen_calibration(self):
        import week8_pipeline_score as w
        plan=w.load_release('original_base')
        source=w.score_directory('original_base')
        with tempfile.TemporaryDirectory() as t,patch.object(w,'ROOT',Path(t)),patch.object(w,'load_release',return_value=plan),patch.object(w,'score_directory',return_value=source):
            target=Path(t)/plan['calibration']/'forbidden-pipeline-report'
            with self.assertRaisesRegex(ValueError,'preserved'):
                w.integrate('original_base',target)
            self.assertFalse(target.exists())

if __name__=='__main__':unittest.main()
