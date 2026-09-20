"""Package explicit per-record model review notes and machine alignment evidence."""
from pathlib import Path
import json,hashlib,collections,html
O=Path(__file__).resolve().parent;R=O.parents[2];D=R/'logs/week8-protocol-data-20260914/run-a'
load=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
samples=load(O/'selected_samples.json');notes=load(O/'review_notes.json')
assert {x['review_id'] for x in samples}==set(notes) and len(notes)==50
stats=load(D/'statistics.json')
for name,value in stats['outputs'].items():assert sha(D/name)==value,name
for entry in load(D/'input_receipt.json'):assert sha(R/entry['path'])==entry['sha256'],entry['path']
sg={s:load(D/f'{s}_sharegpt.json') for s in ['train','validation']};al={s:load(D/f'{s}_alpaca.json') for s in ['train','validation']}
flagged={'R08':'name_localization','R10':'shared_default_cache','R14':'inherited_fragment','R15':'inherited_fragment','R16':'inherited_fragment','R17':'inherited_fragment','R18':'inherited_fragment','R33':'underspecified_prompt','R42':'region_zone_translation'}
sources={
'R18':['https://www.bbc.com/ukchina/trad/vert-fut-40289371'],
'R36':['https://learn.microsoft.com/en-us/azure/virtual-machines/spot-vms'],
'R42':['https://docs.cloud.google.com/compute/docs/gcloud-compute'],
'R46':['https://www.belgium.be/en/about_belgium/government/federale_staat'],
'R48':['https://obamalibrary.archives.gov/obamas/president-barack-obama']}
examples=load(O/'example_checks.json');checked={x['review_id'] for x in examples['checks']}
reviews=[];pages=[]
for x in samples:
 sid=x['sample_id'];rid=x['review_id'];s=sg[x['split']][x['row_index']];a=al[x['split']][x['row_index']]
 assert x['record']==s
 pairs=[(s['conversations'][n]['value'],s['conversations'][n+1]['value']) for n in range(0,len(s['conversations']),2)]
 assert a['history']==[list(p) for p in pairs[:-1]] and a['instruction']==pairs[-1][0] and a['output']==pairs[-1][1] and a['system']==s['system'] and a['input']==''
 review={'review_id':rid,'sample_id':sid,'split':x['split'],'row_index':x['row_index'],'source':x['source'],'candidate_row_number':x['lineage']['candidate_row_number'],'assistant_turns_reviewed':x['turns'],'reviewer_type':'AI_ASSISTANT','human_reviewed_this_round':False,'review_date':'2026-09-15','all_turns_read':True,'alignment_check':'PASS_EXACT','decision':'KEEP_WITH_NOTE' if rid in flagged else 'KEEP','issue_tag':flagged.get(rid),'blocking_issue':False,'evidence':notes[rid],'example_execution':rid in checked,'external_sources_checked':sources.get(rid,[]),'action':'retain_immutable_original','raw_source_human_signature_added':False}
 reviews.append(review)
 content=''.join('<h4>'+html.escape(c['from'])+'</h4><pre>'+html.escape(c['value'])+'</pre>' for c in s['conversations'])
 pages.append(f'<details id="{rid}"><summary>{rid} · {x["source"]} · {x["split"]} · {x["tokens"]} tokens · {review["decision"]}</summary><p>{html.escape(sid)}</p><p>{html.escape(notes[rid])}</p>'+''.join('<p><a href="'+html.escape(u,quote=True)+'">外部核验来源</a></p>' for u in sources.get(rid,[]))+content+'</details>')
(O/'review_records.json').write_text(json.dumps(reviews,ensure_ascii=False,indent=2)+'\n')
(O/'issues.json').write_text(json.dumps([r for r in reviews if r['issue_tag']],ensure_ascii=False,indent=2)+'\n')
(O/'review_pack.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>Week8任务4逐条抽查</title><style>body{max-width:1000px;margin:32px auto;padding:0 20px;font:16px/1.7 system-ui}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f7f9;padding:12px}details{border:1px solid #ddd;padding:12px;margin:12px 0}summary{cursor:pointer;font-weight:600}</style><h1>任务4：50条内容抽查</h1><p>AI辅助逐条复核，不是新增人工审核。已阅读全部轮次；41条保留、9条带提示保留，无新增阻断项。不外推全量事实正确。</p>'+''.join(pages)+'</html>')
receipt={'status':'TASK4_SAMPLE_REVIEW_PASS_NOT_DATA_READY','date':'2026-09-15','reviewed_records':50,'reviewed_assistant_turns':sum(r['assistant_turns_reviewed'] for r in reviews),'reviewer_type':'AI_ASSISTANT','human_reviewed_this_round':False,'decisions':dict(collections.Counter(r['decision'] for r in reviews)),'blocking_issues':0,'new_exclusions':0,'text_changes':0,'sample_all_multiturn_covered':sum(x['turns']>1 for x in samples)==13,'source_split_counts':dict(collections.Counter(x['source']+'/'+x['split'] for x in samples)),'example_checks_passed':examples['count'],'snapshot_hashes':{p.name:sha(p) for p in sorted(D.glob('*.json'))},'protocol_sha256':sha(R/'configs/week8_data_protocol.json'),'retained_machine_check_receipt':sha(R/'reports/week8/phase1_task3/verification.json'),'data_rerun_required':False,'rerun_reason':'No source, processing rule, split, or protocol changes. Rechecked hashes and sampled dual-format alignment. Prior full machine checks remain bound to exact data bytes.','pending':['actual_LLaMA_Factory_loader','benchmark_and_judge_runtime_lock','final_data_release'],'not_applicable':['current_run_truncated_samples_0','current_run_rejected_samples_0','real_system_examples_0'],'quality_scope':'risk_stratified_sample_not_population_estimate'}
(O/'verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print({k:v for k,v in receipt.items() if k not in ['snapshot_hashes','source_split_counts']})
