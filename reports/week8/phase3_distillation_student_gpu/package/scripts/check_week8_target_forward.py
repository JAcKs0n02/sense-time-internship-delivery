"""Bounded target-only model identity and forward check; no training/scoring."""
import os
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4')
import argparse
import hashlib
import json
import socket
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--prompt-receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('output already exists')
    assert socket.gethostname() == 'autodl-container-be044ebe99-be706b14'
    receipt = json.loads(args.prompt_receipt.read_text())
    assert receipt['status'] == 'ACTUAL_OC_PROMPT_RUNTIME_PASS_NO_INFERENCE'
    assert receipt['platform'] == 'linux' and receipt['questions'] == 12928
    assert receipt['custom_template_sha256'] == hashlib.sha256((ROOT/'scripts/week8_prompt_template.py').read_bytes()).hexdigest()
    from mmengine.config import Config
    import torch
    from opencompass.models.huggingface_above_v4_33 import HuggingFacewithChatTemplate
    cfg = dict(Config.fromfile(str(args.config)).models[0])
    assert cfg['tokenizer_only'] is False and cfg['max_out_len'] == 32
    base = Path(cfg['path'])
    prior = json.loads((ROOT/'reports/week8/phase2_gpu_320/remote_preflight.json').read_text())
    assert str(base) == prior['base_path']
    for item in prior['files']:
        p = base/item['path']
        digest = hashlib.file_digest(p.open('rb'), 'sha256').hexdigest() if sys.version_info >= (3,11) else file_hash(p)
        assert digest == item['sha256'], str(p)
    print('BASE_IDENTITY_PASS', len(prior['files']), flush=True)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert torch.cuda.is_bf16_supported()
    torch.manual_seed(42)
    model_args = {k:v for k,v in cfg.items() if k not in ('type','abbr','batch_size','max_out_len','run_cfg')}
    wrapper = HuggingFacewithChatTemplate(**model_args)
    model = wrapper.model.eval()
    assert all(p.device.type == 'cuda' and p.dtype == torch.bfloat16 for p in model.parameters())
    assert not getattr(model, 'is_quantized', False)
    ids = wrapper.tokenizer.apply_chat_template([{'role':'user','content':'请用一句话介绍你自己。'}], tokenize=True, add_generation_prompt=True, return_tensors='pt').to('cuda')
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    assert output.logits.shape[:2] == ids.shape
    assert torch.isfinite(output.logits).all().item()
    shape = list(output.logits.shape)
    del output
    # Also exercise the actual OC wrapper's generation plumbing, bounded to 2 tokens.
    response = wrapper.generate([[{'role':'HUMAN','prompt':'你好'}]], max_out_len=2)
    assert len(response) == 1 and isinstance(response[0], str)
    result = dict(status='TARGET_BASE_FORWARD_PASS_NOT_TRAINING',host=socket.gethostname(),base_files=len(prior['files']),
                  input_tokens=ids.shape[1],logits_shape=shape,all_logits_finite=True,all_parameters_cuda_bfloat16=True,
                  oc_generate_smoke_max_new_tokens=2,oc_generate_response=response,
                  peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(),training_started=False,benchmark_scored=False,
                  config_sha256=file_hash(args.config),prompt_receipt_sha256=file_hash(args.prompt_receipt))
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)


def file_hash(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


if __name__=='__main__':
    main()
