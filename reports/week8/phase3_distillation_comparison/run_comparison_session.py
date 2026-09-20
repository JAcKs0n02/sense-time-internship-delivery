"""Bounded existing-protocol comparison; preserve source exports and use canonical views."""
import base64, datetime, fcntl, gzip, hashlib, json, os, shutil, socket, subprocess, sys, time
from pathlib import Path
from importlib.metadata import version
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT)]
from common import sha256,write_json
from compare_distillation import verify_model_manifest,run_bounded,build_ceval_config


def export_evidence(out):
    files=[]
    for p in sorted(out.rglob('*')):
        if p.is_file() and 'views' not in p.relative_to(out).parts and p.suffix in ('.json','.csv','.log','.py','.txt') and p.stat().st_size<4_000_000:
            b=p.read_bytes();files.append(dict(path=str(p.relative_to(out)),sha256=hashlib.sha256(b).hexdigest(),base64=base64.b64encode(b).decode()))
    b=json.dumps({'files':files},ensure_ascii=False).encode();h=hashlib.sha256(b).hexdigest()
    (ROOT/'comparison_evidence.txt').write_text(json.dumps({'bundle_sha256':h,'gzip_base64':base64.b64encode(gzip.compress(b)).decode()}))
    write_json(ROOT/'evidence_summary.json',dict(files=len(files),bundle_sha256=h))


def audit_tokenizers(before,after,out):
    from transformers import AutoTokenizer
    from mmengine.config import Config
    from scripts.week8_prompt_template import NonRecursivePromptTemplate
    from opencompass.models.huggingface_above_v4_33 import HuggingFacewithChatTemplate,_convert_chat_messages
    from opencompass.openicl.icl_retriever import FixKRetriever
    from opencompass.openicl.icl_inferencer import GenInferencer
    from opencompass.registry import LOAD_DATASET,ICL_PROMPT_TEMPLATES
    a=AutoTokenizer.from_pretrained(before,local_files_only=True)
    b=AutoTokenizer.from_pretrained(after,local_files_only=True)
    assert a.get_vocab()==b.get_vocab() and a.chat_template==b.chat_template
    assert a.special_tokens_map==b.special_tokens_map
    assert json.loads(a.backend_tokenizer.to_str())==json.loads(b.backend_tokenizer.to_str())
    expected={(r['dataset'],r['subject'],r['row_index']):r for r in json.loads((ROOT/'reports/week8/phase2_eval_hardening_v2/prompt_lengths.json').read_text())}
    model=HuggingFacewithChatTemplate(path=str(before),tokenizer_only=True,max_seq_len=2048,generation_kwargs={'do_sample':False},tokenizer_kwargs={'local_files_only':True,'trust_remote_code':False})
    infer=GenInferencer(model=model,max_out_len=32,max_seq_len=2048,batch_size=1,output_json_filepath=str(out/'unused_predictions'))
    cfg,_=build_ceval_config(ROOT,str(before));rows=[]
    for d in cfg['datasets']:
        ds=LOAD_DATASET.build({k:v for k,v in d.items() if k not in ('infer_cfg','eval_cfg')})
        retriever=FixKRetriever(ds,fix_id_list=[0,1,2,3,4]);template=ICL_PROMPT_TEMPLATES.build(d['infer_cfg']['ice_template'])
        assert isinstance(template,NonRecursivePromptTemplate)
        prompts=infer.get_generation_prompt_list_from_retriever_indices(retriever.retrieve(),retriever,'',max_seq_len=2048,ice_template=template)
        for i,prompt in enumerate(prompts):
            messages=_convert_chat_messages([model.parse_template(prompt,mode='gen')])[0]
            assert [m['role'] for m in messages]==['user','assistant']*5+['user']
            ids=a.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
            assert ids==b.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
            rendered=model.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            assert ids==model.tokenizer.batch_encode_plus([rendered],padding=True,truncation=True,add_special_tokens=False,max_length=2048)['input_ids'][0]
            digest=hashlib.sha256(json.dumps(ids).encode()).hexdigest();exp=expected[('ceval',d['name'],i)]
            assert digest==exp['token_ids_sha256'] and len(ids)==exp['input_tokens'] and len(ids)+32<=2048
            rows.append(dict(subject=d['name'],row_index=i,input_tokens=len(ids),token_ids_sha256=digest))
    assert len(rows)==1346 and len({r['subject'] for r in rows})==52
    questions=json.loads((ROOT/'deliverables/week3/day14/source/data/evaluation_questions.json').read_text())['questions'][:12]
    assert len(questions)==12
    for q in questions: assert a.apply_chat_template(q['messages'],tokenize=True,add_generation_prompt=True)==b.apply_chat_template(q['messages'],tokenize=True,add_generation_prompt=True)
    write_json(out/'prompt_rows.json',rows)
    write_json(out/'tokenizer_audit.json',dict(status='TOKENIZER_AND_ACTUAL_OC_PROMPTS_PASS',questions=1346,subjects=52,speed_prompts=12,max_input_tokens=max(r['input_tokens'] for r in rows),backend_equal=True,vocab_equal=True,chat_template_equal=True,no_truncation=True,padding_policy='original canonical tokenizer used in both isolated views; batch_size=1',prompt_rows_sha256=sha256(out/'prompt_rows.json')))


def main():
    deadline=datetime.datetime.fromisoformat(sys.argv[1].replace('Z','+00:00')).timestamp()
    assert socket.gethostname()=='autodl-container-be044ebe99-be706b14'
    assert deadline-time.time()>3600
    guard=(ROOT/'session.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=ROOT/'run-01';out.mkdir(exist_ok=False);completed=[]
    try:
        for e in json.loads((ROOT/'upload_manifest.json').read_text()): assert sha256(ROOT/e['path'])==e['sha256'],e['path']
        expected={'transformers':'4.50.0','torch':'2.5.1+cu121','opencompass':'0.5.3'}
        versions={k:version(k) for k in expected};assert versions==expected
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.total','--format=csv,noheader'],text=True).strip();assert len(gpu.splitlines())==1 and 'RTX 3090' in gpu
        assert shutil.disk_usage(ROOT).free>2*1024**3
        write_json(out/'launch.json',dict(host=socket.gethostname(),gpu=gpu,versions=versions,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),deadline_utc=sys.argv[1]))
        bm=json.loads((ROOT/'before_manifest.json').read_text());am=json.loads((ROOT/'after_manifest.json').read_text())
        before=Path(bm['model_dir']);after=Path(am['model_dir'])
        verify_model_manifest(before,ROOT/'before_manifest.json');verify_model_manifest(after,ROOT/'after_manifest.json')
        audit_tokenizers(before,after,out);completed.append('model_identities_and_1346_prompt_pairs')
        # Compare only learned weights under identical tokenizer/generation settings.
        # Hard links preserve weight bytes without duplicating ~1 GB per view.
        views={}
        for label,source in [('before',before),('after',after)]:
            view=out/'views'/label;view.mkdir(parents=True)
            for p in before.iterdir():
                origin=source/p.name if p.name in ('model.safetensors','config.json') else p
                if p.is_file(): os.link(origin,view/p.name)
            assert sha256(view/'model.safetensors')==sha256(source/'model.safetensors')
            manifest=dict(model_dir=str(view),source_model_dir=str(source),canonical_tokenizer_dir=str(before),canonical_generation_config_dir=str(before),files=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha256(p)) for p in sorted(view.iterdir()) if p.is_file()])
            write_json(out/(label+'_view_manifest.json'),manifest);verify_model_manifest(view,out/(label+'_view_manifest.json'));views[label]=view
        write_json(out/'status.json',dict(status='RUNNING',stage='paired_ceval_and_speed',completed=completed))
        cmd=[sys.executable,str(ROOT/'scripts/compare_distillation.py'),'--before',str(views['before']),'--after',str(views['after']),'--before-manifest',str(out/'before_view_manifest.json'),'--after-manifest',str(out/'after_view_manifest.json'),'--timeout-seconds','3600','--output-dir',str(out/'comparison')]
        write_json(out/'comparison_command.json',cmd)
        run_bounded(cmd,out/'comparison.log',min(3600,deadline-time.time()-60))
        result=json.loads((out/'comparison/status.json').read_text());assert result['status']=='completed' and result['same_protocol'] is True
        verify_model_manifest(before,ROOT/'before_manifest.json');verify_model_manifest(after,ROOT/'after_manifest.json')
        completed.append('both_ceval_1346_and_paired_speed')
        write_json(out/'status.json',dict(status='COMPARISON_COMPLETED_PENDING_LOCAL_RECOUNT',completed=completed))
    except BaseException as e:
        write_json(out/'status.json',dict(status='FAILED_NO_RETRY',completed=completed,error_type=type(e).__name__,error=str(e)));raise
    finally:
        write_json(out/'session_exit.json',dict(ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed=completed))
        export_evidence(out)

if __name__=='__main__': main()
