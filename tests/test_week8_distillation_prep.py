import json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import distill
from step1_data_prep import CharacterSmokeTokenizer

class DistillationPrepTests(unittest.TestCase):
    def test_explicit_student_data_avoids_frozen_protocol_cli(self):
        self.assertTrue(hasattr(distill,'prepare_student_data'),'missing explicit student-data adapter')
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);source=p/'targets.json'
            source.write_text(json.dumps([{'instruction':'Explain photosynthesis in plants.','input':'','output':'Plants convert light into chemical energy using chlorophyll.'},{'instruction':'Compute the sum of three and five.','input':'','output':'The result is eight.'},{'instruction':'Describe the role of a database transaction.','input':'','output':'A transaction groups changes into an atomic unit.'}]))
            factory=types.SimpleNamespace(from_pretrained=lambda path,**kw: CharacterSmokeTokenizer())
            with patch.dict(sys.modules,{'transformers':types.SimpleNamespace(AutoTokenizer=factory)}):
                result=distill.prepare_student_data(source,'test-student',p/'data',{'seed':42,'cutoff_len':1024})
            self.assertEqual(result['clean_count'],3)
            self.assertEqual(result['train_count'],2)
            self.assertEqual(result['validation_count'],1)
            self.assertEqual(result['prompt_overlap'],0)
            self.assertEqual(result['source_sha256'],distill.sha256(source))
            self.assertEqual(json.loads((p/'data/dataset_info.json').read_text())['week8_train']['file_name'],'train_alpaca.json')

    def test_teacher_preserves_system_and_history(self):
        self.assertTrue(hasattr(distill,'teacher_messages'))
        row={'system':'Be precise','history':[['Earlier question','Earlier answer']], 'instruction':'Current question','input':'Extra context'}
        self.assertEqual(distill.teacher_messages(row),[{'role':'system','content':'Be precise'},{'role':'user','content':'Earlier question'},{'role':'assistant','content':'Earlier answer'},{'role':'user','content':'Current question\nExtra context'}])

if __name__=='__main__':unittest.main()
