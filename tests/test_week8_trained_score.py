import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))

class TrainedScoreTests(unittest.TestCase):
    def test_adapter_available(self):
        self.assertIsNotNone(importlib.util.find_spec('week8_trained_score'))

    def test_real_models_same_protocol_distinct_identity(self):
        from week8_trained_score import prepare
        base=json.loads((ROOT/'reports/week8/phase2_gemini_score_release/plan.json').read_text())
        s,d=prepare('final_sft'),prepare('final_dpo')
        self.assertNotEqual(s['identity'],d['identity'])
        for p in (s,d):
            self.assertEqual(len(p['requests']),20)
            self.assertEqual(p['requests'][0]['body']['systemInstruction'],base['requests'][0]['body']['systemInstruction'])
            self.assertEqual(p['requests'][0]['body']['generationConfig'],base['requests'][0]['body']['generationConfig'])
            self.assertIn('最终答案：202',json.loads(p['requests'][0]['body']['contents'][0]['parts'][0]['text'])['candidate_answer'])

    def test_tampered_answers_block_plan(self):
        from week8_trained_score import prepare
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/'source'
            shutil.copytree(ROOT/'reports/week8/phase2_trained_generation',src,ignore=shutil.ignore_patterns('*.tar.gz'))
            answer=src/'retrieved/logs/trained-generation-20260917/final_sft/answers.jsonl'
            answer.write_text(answer.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'artifact changed'):
                prepare('final_sft',source=src)

    def test_wrong_model_launch_rejected_even_with_updated_manifest(self):
        from week8_trained_score import prepare
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/'source'
            shutil.copytree(ROOT/'reports/week8/phase2_trained_generation',src,ignore=shutil.ignore_patterns('*.tar.gz'))
            rel='logs/trained-generation-20260917/final_sft/launch.json'
            f=src/'retrieved'/rel
            value=json.loads(f.read_text());value['model_name']='final_dpo';f.write_text(json.dumps(value))
            manifest=src/'retrieved/retrieval_manifest.json';value=json.loads(manifest.read_text())
            for row in value['files']:
                if row['path']==rel:
                    row['bytes']=f.stat().st_size;row['sha256']=hashlib.sha256(f.read_bytes()).hexdigest()
            manifest.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'model launch mismatch'):
                prepare('final_sft',source=src)

    def test_unknown_model_rejected(self):
        from week8_trained_score import prepare
        with self.assertRaises(ValueError):prepare('original_base')

if __name__=='__main__':unittest.main()
