import copy,json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from week8_responses_judge import validate_response
class ResponsesJudgeTests(unittest.TestCase):
 def setUp(self):
  scores={d:{'score':5,'reason':'依据明确'} for d in ['accuracy','completeness','logic','safety','format']}
  self.raw={'object':'response','model':'deepseek-flash','status':'completed','error':None,'incomplete_details':None,'usage':{'input_tokens':10,'output_tokens':20,'total_tokens':30},'output':[{'type':'reasoning','status':'completed'},{'type':'message','role':'assistant','status':'completed','content':[{'type':'output_text','text':json.dumps(scores)}]}]}
 def test_valid(self):self.assertEqual(validate_response(self.raw)['weighted_score'],5)
 def test_reject_envelopes(self):
  for key,value in [('model','wrong'),('status','incomplete'),('error',{'code':'error'}),('incomplete_details',{'reason':'max_output_tokens'})]:
   with self.subTest(key=key):
    r=copy.deepcopy(self.raw);r[key]=value
    with self.assertRaises(ValueError):validate_response(r)
 def test_usage(self):
  for value in [None,True,-1,31]:
   r=copy.deepcopy(self.raw);r['usage']['total_tokens']=value
   with self.assertRaises(ValueError):validate_response(r)
 def test_output_shape(self):
  for value in [[],[{'type':'function_call'}],self.raw['output']*2]:
   r=copy.deepcopy(self.raw);r['output']=value
   with self.assertRaises(ValueError):validate_response(r)
 def test_invalid_json_no_repair(self):
  for text in ['{}','{"accuracy":1,"accuracy":2}',self.raw['output'][1]['content'][0]['text'][:-1]]:
   r=copy.deepcopy(self.raw);r['output'][1]['content'][0]['text']=text
   with self.assertRaises(ValueError):validate_response(r)
 def test_refusal(self):
  r=copy.deepcopy(self.raw);r['output'][1]['content']=[{'type':'refusal','refusal':'no'}]
  with self.assertRaises(ValueError):validate_response(r)
if __name__=='__main__':unittest.main()
