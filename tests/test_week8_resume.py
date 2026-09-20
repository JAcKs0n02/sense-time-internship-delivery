import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import step3_eval as ev
import week8_resume_eval as resume

SOURCE = ROOT/'reports/week8/phase2_formal_inference/recovered_failure'
OLD_LOCK = ROOT/'reports/week8/phase2_three_model_eval/original_base/runtime_lock.json'


def bound_lock():
    lock = json.loads(OLD_LOCK.read_text())
    lock['resume_source'] = {
        'manifest_sha256': ev.sha256(SOURCE/'manifest.json'),
        'run_name': 'formal-eval-20260916-01', 'model_name': 'original_base',
        'previous_lock': {'path': str(OLD_LOCK), 'sha256': ev.sha256(OLD_LOCK)},
    }
    for dep in lock['dependencies']:
        if dep['path'] == 'scripts/step3_eval.py': dep['sha256'] = ev.sha256(ROOT/dep['path'])
    lock['dependencies'].append({'path': 'scripts/week8_resume_eval.py', 'sha256': ev.sha256(ROOT/'scripts/week8_resume_eval.py')})
    return lock


class ResumeTests(unittest.TestCase):
    def test_real_evidence_validates_and_records_provenance(self):
        rows, provenance = resume.validate_source(SOURCE, bound_lock())
        self.assertEqual([(r['questions'], r['correct']) for r in rows], [(1346,1049),(11582,9246)])
        self.assertEqual(provenance['source_manifest_sha256'], ev.sha256(SOURCE/'manifest.json'))
        self.assertTrue(all(r['evidence_kind']=='reused_verified_opencompass' for r in rows))

    def test_changed_identity_and_missing_code_binding_rejected(self):
        for kind in ('model','config','records','weights','source_lock','manifest','code'):
            lock=bound_lock()
            if kind=='model':lock['model_path']='/wrong/model'
            if kind in ('config','records'):lock[kind]['sha256']='0'*64
            if kind=='weights':
                next(d for d in lock['dependencies'] if d['path'].endswith('.safetensors'))['sha256']='0'*64
            if kind=='source_lock':lock['resume_source']['previous_lock']['sha256']='0'*64
            if kind=='manifest':lock['resume_source']['manifest_sha256']='0'*64
            if kind=='code':lock['dependencies']=[d for d in lock['dependencies'] if d['path']!='scripts/week8_resume_eval.py']
            with self.subTest(kind=kind),self.assertRaises(ValueError):resume.validate_source(SOURCE,lock)

    def test_tampering_extra_answers_missing_files_and_symlinks_rejected(self):
        for kind in ('tamper','missing','extra_answer','symlink'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                src=Path(tmp)/'source';shutil.copytree(SOURCE,src)
                path=next(src.glob('*/original_base/opencompass/*/results/*/*.json'))
                if kind=='tamper':path.write_text('{}')
                if kind=='missing':path.unlink()
                if kind=='extra_answer':(src/'formal-eval-20260916-01/original_base/answers.jsonl').write_text('{}\n')
                if kind=='symlink':path.unlink();path.symlink_to('/etc/hosts')
                with self.assertRaises(ValueError):resume.validate_source(src,bound_lock())

    def test_invalid_source_stops_before_custom_and_paid_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=SimpleNamespace(source=SOURCE,output_dir=Path(tmp)/'out',model=bound_lock()['model_path'])
            lock=bound_lock();lock['resume_source']['manifest_sha256']='0'*64
            with patch.object(ev,'require_evaluation_lock',return_value=lock),patch.object(ev,'complete_custom20') as custom,patch.object(ev,'judge_balance') as balance,patch.object(ev.subprocess,'run') as oc:
                with self.assertRaises(ValueError):resume.resume(args)
                custom.assert_not_called();balance.assert_not_called();oc.assert_not_called()

    def test_failed_custom_attempt_cannot_be_restarted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=SimpleNamespace(source=SOURCE,output_dir=root/'out',model=bound_lock()['model_path'])
            args.output_dir.mkdir()
            with patch.object(resume,'ROOT',root),patch.object(ev,'require_evaluation_lock',return_value=bound_lock()),patch.object(ev,'load_judge_profile',return_value={}),patch.object(ev,'complete_custom20',side_effect=RuntimeError('simulated failure')) as custom:
                with self.assertRaisesRegex(RuntimeError,'simulated failure'):resume.resume(args)
                with self.assertRaises(FileExistsError):resume.resume(args)
                custom.assert_called_once()

    def test_source_output_overlap_rejected_before_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=SimpleNamespace(source=SOURCE,output_dir=SOURCE/'new',model=bound_lock()['model_path'])
            with patch.object(resume,'ROOT',root),patch.object(ev,'require_evaluation_lock',return_value=bound_lock()),patch.object(ev,'load_judge_profile',return_value={}),patch.object(ev,'complete_custom20') as custom:
                with self.assertRaisesRegex(ValueError,'outside'):resume.resume(args)
                self.assertFalse((root/'logs').exists());custom.assert_not_called()

    def test_success_skips_opencompass_and_duplicate_claim_blocks_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=SimpleNamespace(source=SOURCE,output_dir=root/'out',model=bound_lock()['model_path'])
            args.output_dir.mkdir()
            with patch.object(resume,'ROOT',root),patch.object(ev,'require_evaluation_lock',return_value=bound_lock()),patch.object(ev,'load_judge_profile',return_value={}),patch.object(ev,'complete_custom20') as custom,patch.object(ev.subprocess,'run') as oc:
                resume.resume(args)
                custom.assert_called_once();oc.assert_not_called()
                recorded=json.loads((args.output_dir/'resume_runtime_lock.json').read_text())
                self.assertEqual(recorded, bound_lock())
                self.assertEqual(custom.call_args.args[2][0]['evidence_kind'],'reused_verified_opencompass')
                with self.assertRaises(FileExistsError):resume.resume(args)
                custom.assert_called_once()

if __name__ == '__main__': unittest.main()
