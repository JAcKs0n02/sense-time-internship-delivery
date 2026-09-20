#!/usr/bin/env python3
"""Fresh-process BF16 merged-model loading check; no benchmark or judge API."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();model_path=a.model.resolve()
    if a.output.exists():raise ValueError('receipt already exists')
    if (model_path/'adapter_config.json').exists():raise ValueError('expected standalone merged model')
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise ValueError('CUDA BF16 required')
    torch.manual_seed(42);torch.cuda.reset_peak_memory_stats()
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,trust_remote_code=False,
                                            torch_dtype=torch.bfloat16,device_map={'':'cuda:0'}).eval()
    text=tokenizer.apply_chat_template([{'role':'user','content':'你好'}],tokenize=False,add_generation_prompt=True)
    inputs=tokenizer(text,return_tensors='pt').to('cuda:0')
    with torch.inference_mode():
        logits=model(**inputs).logits
        if not torch.isfinite(logits).all().item():raise ValueError('nonfinite cold-load logits')
        generated=model.generate(**inputs,max_new_tokens=2,do_sample=False)
    count=generated.shape[-1]-inputs['input_ids'].shape[-1]
    if not 1<=count<=2:raise ValueError('no generated token')
    result={'status':'COLD_LOAD_PASS','model_path':str(model_path),'finite_logits':True,
            'generated_tokens':count,'generated_token_ids':generated[0,-count:].tolist(),
            'text':tokenizer.decode(generated[0,-count:]),'dtype':str(model.dtype),'device':str(model.device),
            'peak_memory_allocated_bytes':torch.cuda.max_memory_allocated(),'benchmark_scored':False}
    with a.output.open('x') as f:f.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
