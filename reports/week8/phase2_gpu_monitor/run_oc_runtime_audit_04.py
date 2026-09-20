"""Exercise actual OC configuration, loaders, retriever and chat wrapper without weights."""
import os
os.environ.update(OMP_NUM_THREADS='4', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
import json,hashlib,copy,importlib.metadata,socket
from pathlib import Path
import opencompass
from mmengine.config import Config
from opencompass.models.huggingface_above_v4_33 import HuggingFacewithChatTemplate, _convert_chat_messages
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.registry import LOAD_DATASET

root=Path('/root/autodl-tmp/week8-recheck-20260915')
os.chdir(root)
out=root/'oc-runtime-04';out.mkdir(exist_ok=False)
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert socket.gethostname()=='autodl-container-be044ebe99-be706b14'
assert importlib.metadata.version('opencompass')=='0.5.3'
expected=json.loads((Path('/root/autodl-tmp/prompt_lengths.json')).read_text())
expected={(r['dataset'],r['subject'],r['row_index']):r for r in expected}
model=HuggingFacewithChatTemplate(path=str(root/'loader-attempt-01/tokenizer'),tokenizer_only=True,max_seq_len=2048,generation_kwargs={'do_sample':False},tokenizer_kwargs={'local_files_only':True,'trust_remote_code':False})
assert not hasattr(model,'model')
infer=GenInferencer(model=model,max_out_len=32,max_seq_len=2048,batch_size=1,output_json_filepath=str(out/'unused_predictions'))
rows=[];configs=[];source_hashes={}
for dataset,filename in [('ceval','ceval_gen_5f30c7.py'),('cmmlu','cmmlu_gen_c13365.py')]:
 p=Path(opencompass.__file__).parent/'configs/datasets'/dataset/filename
 source_hashes[str(p)]=h(p)
 cfg=Config.fromfile(str(p))
 for d in cfg[dataset+'_datasets']:
  d=copy.deepcopy(d);d['path']=str(root/'reports/week8/phase2_preflight/local_data'/dataset)
  configs.append(d)
  load={k:v for k,v in d.items() if k not in ('infer_cfg','eval_cfg')}
  ds=LOAD_DATASET.build(load)
  retriever=FixKRetriever(ds,fix_id_list=[0,1,2,3,4])
  indices=retriever.retrieve()
  template=PromptTemplate(**{k:v for k,v in d['infer_cfg']['ice_template'].items() if k!='type'})
  prompts=infer.get_generation_prompt_list_from_retriever_indices(indices,retriever,'',max_seq_len=2048,ice_template=template)
  for i,prompt in enumerate(prompts):
   parsed=model.parse_template(prompt,mode='gen')
   messages=_convert_chat_messages([parsed])[0]
   assert len(messages)==11 and [m['role'] for m in messages]==['user','assistant']*5+['user'],(dataset,d['name'],i,'shots')
   rendered=model.tokenizer.apply_chat_template(messages,add_generation_prompt=True,tokenize=False)
   ids=model.tokenizer(rendered,add_special_tokens=False)['input_ids']
   actual=model.tokenizer.batch_encode_plus([rendered],padding=True,truncation=True,add_special_tokens=False,max_length=2048)['input_ids'][0]
   assert actual==ids and len(ids)+32<=2048
   digest=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
   exp=expected[(dataset,d['name'],i)]
   assert digest==exp['token_ids_sha256'] and len(ids)==exp['input_tokens'],(dataset,d['name'],i,'token mismatch')
   rows.append({'dataset':dataset,'subject':d['name'],'row_index':i,'input_tokens':len(ids),'token_ids_sha256':digest,'fewshot_count':5})
  print('SUBJECT_PASS',dataset,d['name'],len(prompts),flush=True)
Config(dict(datasets=configs,models=[dict(type=HuggingFacewithChatTemplate,path=str(root/'loader-attempt-01/tokenizer'),tokenizer_only=True,max_seq_len=2048,generation_kwargs=dict(do_sample=False),batch_size=1,max_out_len=32)],seed=42)).dump(str(out/'expanded_audit_config.py'))
assert len(rows)==len(expected)==12928 and len(configs)==119
(out/'prompt_rows.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
receipt={'status':'ACTUAL_OC_PROMPT_RUNTIME_PASS_NO_INFERENCE','host':socket.gethostname(),'opencompass_version':importlib.metadata.version('opencompass'),'subjects':len(configs),'questions':len(rows),'max_input_tokens':max(r['input_tokens'] for r in rows),'all_token_hashes_match_independent':True,'all_five_shot':True,'truncation_changes':0,'source_config_hashes':source_hashes,'model_weights_loaded':False,'model_inference_run':False,'training_started':False,'full_evaluation_lock_ready':False,'expanded_config_sha256':h(out/'expanded_audit_config.py'),'prompt_rows_sha256':h(out/'prompt_rows.json')}
(out/'verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt,indent=2),flush=True)
