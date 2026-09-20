import copy,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from week8_generation_only import validate
class GenerationTests(unittest.TestCase):
 def setUp(self):self.p=json.loads((ROOT/'reports/week8/phase2_trained_generation/plan.json').read_text())
 def test_local_evidence(self):self.assertTrue(validate(self.p,ROOT,check_models=False))
 def test_wrong_model_or_changed_lock(self):
  for mode in ['base','hash','judge']:
   p=copy.deepcopy(self.p)
   if mode=='base':p['models']['original_base']=p['models'].pop('final_sft')
   elif mode=='hash':p['models']['final_sft']['lock_sha256']='0'*64
   else:p['judge_calls']=1
   with self.assertRaises(ValueError):validate(p,ROOT,check_models=False)
 def test_changed_dependency(self):
  p=copy.deepcopy(self.p);p['dependencies']['scripts/week8_generation_only.py']='0'*64
  with self.assertRaises(ValueError):validate(p,ROOT,check_models=False)
if __name__=='__main__':unittest.main()
