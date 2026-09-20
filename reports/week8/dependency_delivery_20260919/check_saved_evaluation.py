"""Read-only validation of relocated, already completed evaluation evidence."""
from pathlib import Path
import json
import sys

root = Path.cwd()
sys.path.insert(0, str(root/'scripts'))
from common import load_module, sha256

m = load_module('delivery_eval', root/'scripts/pipeline/step3_eval.py')
output = root/'reports/week8/delivery_eval_20260919/retrieved/eval-check-002/evaluation'
plan = json.loads((output/'plan.json').read_text())
receipt = json.loads((output/'generation/generation_receipt.json').read_text())
assert receipt['plan_sha256'] == sha256(output/'plan.json')
for path, expected in plan['dependencies'].items():
    assert sha256(root/path) == expected, path
for name, key in [('opencompass_config.py', 'config_sha256'), ('benchmark_records.json', 'records_sha256')]:
    assert sha256(output/name) == plan[key], name
m.shared.verify_source_files(output/'generation', receipt)
rows = m.benchmark_rows(output, plan)
assert rows == json.loads((output/'generation/benchmarks.json').read_text())
answers = [json.loads(line) for line in (output/'generation/answers.jsonl').read_text().splitlines()]
score_plan = m.shared.bind_score_plan(answers, sha256(output/'generation/answers.jsonl'),
                                     sha256(output/'generation/generation_receipt.json'))
score_plan.update(scope='pipeline_custom20', model_name=plan['model_path'])
result = m.verify_completed(score_plan, output/'scoring')
assert result['questions'] == 20 and result['weighted_mean'] == 3.775
print(json.dumps({'status':'PASS_READ_ONLY_EXISTING_EVALUATION',
    'generation_files':len(receipt['files']), 'dependencies':len(plan['dependencies']),
    'benchmark_questions':sum(r['questions'] for r in rows),
    'custom_questions':result['questions'], 'custom_score':result['weighted_mean'],
    'new_api_calls':0, 'new_inference':False}, ensure_ascii=False))
