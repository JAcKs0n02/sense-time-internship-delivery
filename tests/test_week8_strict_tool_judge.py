import copy,json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from week8_strict_tool_judge import validate_response
class StrictToolTests(unittest.TestCase):
 def setUp(self):
  s={d:{'score':5,'reason':'具体依据'} for d in ['accuracy','completeness','logic','safety','format']}
  self.r={'model':'deepseek-flash','usage':{'prompt_tokens':1,'completion_tokens':2,'total_tokens':3},'choices':[{'finish_reason':'tool_calls','message':{'role':'assistant','content':None,'tool_calls':[{'id':'call_1','type':'function','function':{'name':'submit_scores','arguments':json.dumps(s)}}]}}]}
 def test_valid(self):self.assertEqual(validate_response(self.r)['weighted_score'],5)
 def test_reject_transport_shape(self):
  for mutate in [lambda r:r.update(model='wrong'),lambda r:r['choices'][0].update(finish_reason='length'),lambda r:r['choices'][0]['message'].update(content='extra'),lambda r:r['choices'][0]['message'].update(tool_calls=[]),lambda r:r['choices'][0]['message']['tool_calls'][0]['function'].update(name='exec'),lambda r:r['usage'].update(total_tokens=4)]:
   r=copy.deepcopy(self.r);mutate(r)
   with self.assertRaises(ValueError):validate_response(r)
 def test_no_json_repair(self):
  for args in ['{}','{"accuracy":1,"accuracy":2}',self.r['choices'][0]['message']['tool_calls'][0]['function']['arguments'][:-1]]:
   r=copy.deepcopy(self.r);r['choices'][0]['message']['tool_calls'][0]['function']['arguments']=args
   with self.assertRaises(ValueError):validate_response(r)
if __name__=='__main__':unittest.main()
