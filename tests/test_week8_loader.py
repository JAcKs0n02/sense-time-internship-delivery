import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import check_week8_loader as audit

class SupervisionTests(unittest.TestCase):
    def setUp(self):
        self.expected={'input_ids':[10,11,20,99,12,21,99], 'labels':[-100,-100,20,99,-100,21,99], 'attention_mask':[1]*7,'assistant_spans':[[2,4],[5,7]],'eos_token_id':99}
        self.actual={k:copy.deepcopy(self.expected[k]) for k in ['input_ids','labels','attention_mask']}
    def test_preserved_all_assistant_turns_pass(self):
        result=audit.assert_supervision(self.expected,self.actual)
        self.assertEqual(result['supervised_tokens'],4)
        self.assertEqual(result['assistant_turns'],2)
    def test_prompt_supervision_is_rejected(self):
        self.actual['labels'][0]=10
        with self.assertRaisesRegex(ValueError,'labels'):audit.assert_supervision(self.expected,self.actual)
    def test_masked_history_is_rejected(self):
        self.actual['labels'][2:4]=[-100,-100]
        with self.assertRaisesRegex(ValueError,'labels'):audit.assert_supervision(self.expected,self.actual)
    def test_truncated_eos_is_rejected(self):
        for k in self.actual:self.actual[k].pop()
        with self.assertRaisesRegex(ValueError,'input_ids'):audit.assert_supervision(self.expected,self.actual)
    def test_invisible_attention_and_shifted_labels_rejected(self):
        self.actual['attention_mask'][2]=0
        with self.assertRaisesRegex(ValueError,'attention_mask'):audit.assert_supervision(self.expected,self.actual)
        self.actual['attention_mask'][2]=1;self.actual['labels'][2]=99
        with self.assertRaisesRegex(ValueError,'labels'):audit.assert_supervision(self.expected,self.actual)


class AssetIdentityTests(unittest.TestCase):
    def test_staging_binds_real_qwen_config_and_excludes_adapter_redirect(self):
        import tempfile,json,hashlib
        root=Path(__file__).resolve().parents[1]
        protocol=json.loads((root/'configs/week8_data_protocol.json').read_text())
        with tempfile.TemporaryDirectory(prefix='attempt-mpt-') as td:
            target=Path(td)/'tokenizer';target.mkdir()
            receipt=audit.stage_tokenizer(protocol,target)
            config=json.loads((target/'config.json').read_text())
            self.assertEqual(config['model_type'],'qwen2')
            self.assertEqual(receipt['config_sha256'],'7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c')
            self.assertFalse((target/'adapter_config.json').exists())
            self.assertEqual(hashlib.sha256((target/'tokenizer.json').read_bytes()).hexdigest(),protocol['tokenizer_files'][0]['sha256'])

if __name__=='__main__':unittest.main()
