import copy,json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from week8_gemini_judge import validate_response,MODEL
class GeminiTests(unittest.TestCase):
 def setUp(self):
  s={d:{'score':5,'reason':'依据'} for d in ['accuracy','completeness','logic','safety','format']}
  self.r={'modelVersion':MODEL,'usageMetadata':{'promptTokenCount':10,'candidatesTokenCount':20,'thoughtsTokenCount':30,'totalTokenCount':60},'candidates':[{'finishReason':'STOP','content':{'role':'model','parts':[{'text':json.dumps(s)}]}}]}
 def test_valid(self):self.assertEqual(validate_response(self.r)['weighted_score'],5)
 def test_reject(self):
  for change in [lambda r:r.update(modelVersion='wrong'),lambda r:r['usageMetadata'].update(totalTokenCount=30),lambda r:r['candidates'][0].update(finishReason='MAX_TOKENS'),lambda r:r.update(promptFeedback={'blockReason':'SAFETY'}),lambda r:r['candidates'][0]['content'].update(parts=[{'functionCall':{}}]),lambda r:r['candidates'][0]['content'].update(parts=[{'text':'{}'}])]:
   r=copy.deepcopy(self.r);change(r)
   with self.assertRaises(ValueError):validate_response(r)
if __name__=='__main__':unittest.main()
