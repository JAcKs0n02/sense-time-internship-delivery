import copy
import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))

class DeepSeekJudgeTests(unittest.TestCase):
    def test_explicit_bounded_request(self):
        from week8_deepseek_judge import make_request_body
        body=make_request_body({'question':'2+2?','reference':'4'},'4',{'weights':{}})
        self.assertEqual(body['model'],'deepseek-flash')
        self.assertEqual(body['thinking'],{'type':'enabled'})
        self.assertEqual(body['max_tokens'],4096)
        self.assertNotIn('temperature',body)
        self.assertEqual(body['response_format'],{'type':'json_object'})
        self.assertIn('准确性',body['messages'][0]['content'])
        self.assertEqual(json.loads(body['messages'][1]['content'])['candidate_answer'],'4')

    def test_valid_response_and_reject_silent_failure(self):
        from week8_deepseek_judge import validate_response
        scores={key:{'score':4,'reason':'依据：结论与参考一致'} for key in ('accuracy','completeness','logic','safety','format')}
        raw={'id':'test','model':'deepseek-flash','choices':[{'finish_reason':'stop','message':{'content':json.dumps(scores)}}],
             'usage':{'prompt_tokens':100,'completion_tokens':200,'total_tokens':300}}
        self.assertEqual(validate_response(raw,'deepseek-flash')['weighted_score'],4)
        for change in ('model','truncated','empty','missing_usage','missing_reason','extra_choice'):
            bad=copy.deepcopy(raw)
            if change=='model':bad['model']='unexpected-model'
            if change=='truncated':bad['choices'][0]['finish_reason']='length'
            if change=='empty':bad['choices'][0]['message']['content']=''
            if change=='missing_usage':bad.pop('usage')
            if change=='missing_reason':
                scores['logic']['reason']='';bad['choices'][0]['message']['content']=json.dumps(scores)
            if change=='extra_choice':bad['choices']*=2
            with self.subTest(change=change),self.assertRaises(ValueError):validate_response(bad,'deepseek-flash')

if __name__=='__main__':unittest.main()
