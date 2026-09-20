import copy
import inspect
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import distill
from common import sha256,write_json

class CurationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.r=Path(self.tmp.name)/'curation'
        shutil.copytree(ROOT/'reports/week8/phase3_distillation_curation',self.r)
        self.manifest=self.r/'curation_manifest.json';self.targets=self.r/'curated_targets.json'
        self.receipt=self.r/'retrieved/generation/generation_receipt.json'
        self.config=json.loads(self.receipt.read_text())['config']
        self.review=self.r/'approval.json';self.refresh_review()
    def refresh_review(self):
        write_json(self.review,{'status':'APPROVED_FOR_STUDENT_TRAINING','generation_receipt_sha256':sha256(self.receipt),
            'targets_sha256':sha256(self.targets),'curation_manifest_sha256':sha256(self.manifest),
            'reviewed_sample_ids':[r['sample_id'] for r in json.loads(self.targets.read_text())],
            'notes':'Fixture: reviewed retained subset.', 'unresolved_issues':[]})
    def verify(self):
        self.assertIn('curation_path',inspect.signature(distill.verify_target_review).parameters,
                      'training entry has no auditable subset support')
        return distill.verify_target_review(self.targets,self.receipt,self.review,self.config,curation_path=self.manifest)
    def test_accepts_exact_reviewed_subset_preserving_generation_count(self):
        result=self.verify();self.assertEqual(result['accepted'],191);self.assertEqual(result['curated_count'],152)
    def test_rejects_missing_duplicate_or_unreasoned_decisions(self):
        original=json.loads(self.manifest.read_text())
        for kind in ('missing','duplicate','blank_reason','unknown_decision','wrong_raw_sha','wrong_count','raw_escape','wrong_receipt','wrong_original','wrong_curated_path','wrong_schema'):
            m=copy.deepcopy(original)
            if kind=='missing':m['decisions'].pop()
            elif kind=='duplicate':m['decisions'][-1]=m['decisions'][0]
            elif kind=='blank_reason':m['decisions'][0]['notes']=' '
            elif kind=='unknown_decision':m['decisions'][0]['decision']='MAYBE'
            elif kind=='wrong_raw_sha':m['decisions'][0]['raw_sha256']='0'*64
            elif kind=='wrong_count':m['counts']['retained']=153
            elif kind=='raw_escape':m['decisions'][0]['raw_path']='../outside.json'
            elif kind=='wrong_receipt':m['generation_receipt_sha256']='0'*64
            elif kind=='wrong_original':m['original_targets_sha256']='0'*64
            elif kind=='wrong_curated_path':m['curated_targets_path']='elsewhere.json'
            elif kind=='wrong_schema':m['schema_version']=999
            write_json(self.manifest,m);self.refresh_review()
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.verify()
    def test_rejects_rehashed_answer_edit_and_reordered_subset(self):
        original=json.loads(self.targets.read_text());m=json.loads(self.manifest.read_text())
        for kind in ('edit','reorder','omission','include_excluded'):
            rows=copy.deepcopy(original)
            if kind=='edit':rows[0]['output']='rewritten answer'
            elif kind=='reorder':rows.reverse()
            elif kind=='omission':rows.pop()
            else:rows.append(json.loads((self.r/'retrieved/generation/teacher_targets.json').read_text())[0])
            write_json(self.targets,rows);m['curated_targets_sha256']=sha256(self.targets);write_json(self.manifest,m);self.refresh_review()
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.verify()
    def test_rejects_stale_or_partial_review(self):
        for kind in ('stale','partial','pending','issue'):
            self.refresh_review();v=json.loads(self.review.read_text())
            if kind=='stale':v['curation_manifest_sha256']='0'*64
            elif kind=='partial':v['reviewed_sample_ids']=v['reviewed_sample_ids'][:10]
            elif kind=='pending':v['status']='PENDING'
            else:v['unresolved_issues']=['unresolved']
            write_json(self.review,v)
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.verify()
    def test_rejects_changed_excluded_raw_answer(self):
        raw=self.r/'retrieved/generation/teacher_raw/0000.json';v=json.loads(raw.read_text());v['answer']='changed';write_json(raw,v)
        with self.assertRaises(ValueError):self.verify()

if __name__=='__main__':unittest.main()
