"""Regression boundaries: leakage, lost history, subset partition, and identity."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import week8_data as data

class BoundaryTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return list(range(sum(len(m['content']) for m in messages)+2))

def sample(question, answer):
    return {'conversations':[{'from':'human','value':question},{'from':'gpt','value':answer}]}

def lineage(ids):
    return [{'sample_id':s,'source':'fixture','content_sha256':'fixture'} for s in ids]

class ProtocolDataTests(unittest.TestCase):
    def test_multiturn_system_and_literal_content_survive_both_formats(self):
        row={'system':'  system\n','conversations':[{'from':'human','value':'q1\\n'}, {'from':'gpt','value':'<p>a1</p>'},{'from':'human','value':'q2'},{'from':'gpt','value':'  a2\n'}]}
        messages=data.messages_from_sharegpt(row)
        a=data.to_alpaca(messages)
        self.assertEqual(a,{'system':'  system\n','instruction':'q2','input':'','output':'  a2\n','history':[['q1\\n','<p>a1</p>']]})
        self.assertEqual(data.from_alpaca(a),messages)
        self.assertEqual(data.messages_from_sharegpt(data.to_sharegpt(messages)),messages)

    def test_invalid_role_order_and_blank_answers_rejected(self):
        for r in [sample('q',' '),{'conversations':[{'from':'gpt','value':'a'}]}, {'conversations':[{'from':'human','value':'q'},{'from':'system','value':'x'},{'from':'gpt','value':'a'}]}]:
            with self.assertRaises(ValueError): data.messages_from_sharegpt(r)

    def test_protected_schema_items_and_prompts_are_not_silently_empty(self):
        self.assertEqual(data.extract_questions({'items':[{'id':'a','messages':[{'role':'user','content':'题目'}]}]}),[{'id':'a','turn':0,'text':'题目'}])
        self.assertEqual(data.extract_questions({'prompts':[{'id':'b','text':'题目2'}]}),[{'id':'b','turn':0,'text':'题目2'}])
        with self.assertRaises(ValueError): data.extract_questions({'unexpected':[]})

    def test_nfkc_and_near_duplicate_rules_keep_raw_text_out_of_normalization(self):
        self.assertEqual(data.match_text(' Ａ B\n','ab')['kind'],'exact')
        long='这是为了验证长文本包含规则而设计的一段足够长的中文句子，必须被正确隔离。'
        self.assertEqual(data.match_text(long,'前缀'+long)['kind'],'containment')
        self.assertIsNone(data.match_text('短题','短题额外'))
        base=''.join(chr(0x4e00+i) for i in range(100))
        self.assertEqual(data.match_text(base,base[:50]+'X'+base[51:])['kind'],'char5_jaccard')

    def test_group_quarantine_propagates_from_answer_hit_to_related_question(self):
        raw=[sample('共同问题','受保护题目'),sample('共同问题','独立答复'),sample('第二题','答案乙'),sample('第三题','答案丙')]
        result=data.prepare_records(raw,lineage(['a','b','c','d']),BoundaryTokenizer(),[{'id':'p','text':'受保护题目','path':'fixture'}],[],max_tokens=2048)
        self.assertEqual({x['sample_id'] for x in result['excluded']},{'a','b'})
        self.assertEqual(set(result['records']),{'c','d'})
        self.assertEqual(result['hits'][0]['sample_id'],'a')

    def test_length_boundary_rejects_whole_record_without_truncation(self):
        r=data.prepare_records([sample('Q','abc'),sample('R','abcd')],lineage(['a','b']),BoundaryTokenizer(),[],[],max_tokens=6)
        self.assertEqual(set(r['records']),{'a'})
        self.assertEqual(r['records']['a']['messages'][-1]['content'],'abc')
        self.assertEqual(r['excluded'][0]['reason'],'overlength')

    def test_duplicate_survivor_by_id_and_historical_links_survive(self):
        raw=[sample('Q','A'),sample('Q','A'),sample('R','B')]
        r=data.prepare_records(raw,lineage(['z','a','b']),BoundaryTokenizer(),[],[{'left':'z','right':'b'}],max_tokens=2048)
        self.assertEqual(set(r['records']),{'a','b'})
        self.assertEqual(r['groups'],[['a','b']])
        self.assertEqual(r['excluded'][0]['duplicate_of'],'a')

    def test_subset_split_uses_record_counts_and_never_breaks_groups(self):
        groups=[['a','b'],['c','d','e'],['f','g','h','i','j']]
        train,val=data.split_groups(groups)
        self.assertEqual(len(val),0)  # target1, feasible0/2 equidistant: smaller wins
        self.assertEqual(len(train),10)
        groups=[['a','b']]+[[str(i)] for i in range(18)]
        train,val=data.split_groups(groups)
        self.assertEqual(len(val),2)
        self.assertFalse(set(train)&set(val))
        self.assertEqual(data.split_groups(list(reversed(groups))),(train,val))
        self.assertEqual('a' in val,'b' in val)

    def test_hashed_input_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'data.json';p.write_text('[]')
            (Path(td)/'data').mkdir()
            (Path(td)/'data/input_paths.json').write_text('{}')
            with patch.object(data, 'ROOT', Path(td)):
                with self.assertRaisesRegex(ValueError,'hash mismatch'):
                    data.verify_entry({'path':'data.json','sha256':'0'*64})

if __name__=='__main__': unittest.main()

class ProtocolEntryTests(unittest.TestCase):
    def test_entry_checks_protocol_before_loading_tokenizer(self):
        import subprocess
        with tempfile.TemporaryDirectory() as td:
            d=Path(td)
            p=json.loads((ROOT/'configs/week8_data_protocol.json').read_text())
            p['source']['sha256']='0'*64
            (d/'protocol.json').write_text(json.dumps(p))
            result=subprocess.run([sys.executable,str(ROOT/'scripts/step1_data_prep.py'),'--protocol',str(d/'protocol.json'),'--output-dir',str(d/'out')],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('hash mismatch',result.stderr)
            self.assertFalse((d/'out').exists())

    def test_protocol_cannot_be_combined_with_legacy_overrides(self):
        import subprocess
        with tempfile.TemporaryDirectory() as td:
            result=subprocess.run([sys.executable,str(ROOT/'scripts/step1_data_prep.py'),'--protocol',str(ROOT/'configs/week8_data_protocol.json'),'--quick','--output-dir',str(Path(td)/'out')],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('cannot combine protocol',result.stderr)

class TrainingBoundaryTests(unittest.TestCase):
    def test_task3_data_does_not_silently_start_training(self):
        import subprocess
        with tempfile.TemporaryDirectory() as td:
            d=Path(td)
            (d/'statistics.json').write_text(json.dumps({'mode':'protocol_formal_tokenizer','status':'TASK3_MACHINE_CHECKS_PASS_NOT_DATA_READY'}))
            result=subprocess.run([sys.executable,str(ROOT/'scripts/train_pipeline.py'),'--data-dir',str(d),'--run-dir',str(d/'run'),'--base-model','/unused','--dry-run'],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('task3 data still requires',result.stderr)
            self.assertFalse((d/'run').exists())
