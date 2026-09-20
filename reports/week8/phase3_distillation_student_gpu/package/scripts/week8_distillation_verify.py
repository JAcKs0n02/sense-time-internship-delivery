#!/usr/bin/env python3
"""Audit actual student tensors and LoRA merge; run on CPU in a bounded worker."""
import argparse
import json
from pathlib import Path
from common import sha256, write_json


def tensor_index(folder):
    from safetensors import safe_open
    folder = Path(folder).resolve()
    paths = sorted(folder.glob('*.safetensors'))
    if not paths or any(p.is_symlink() for p in paths):
        raise ValueError('missing weights or symlinked weight files')
    if (folder/'adapter_config.json').exists():
        raise ValueError('expected standalone model, found adapter')
    index = folder/'model.safetensors.index.json'
    expected = json.loads(index.read_text()).get('weight_map', {}) if index.exists() else None
    if expected is None and [p.name for p in paths] != ['model.safetensors']:
        raise ValueError('sharded weights need an index')
    actual = {}
    for path in paths:
        with safe_open(str(path), framework='pt', device='cpu') as stream:
            for key in stream.keys():
                if key in actual:
                    raise ValueError('duplicate tensor across shards')
                actual[key] = path.name
    if not actual or (expected is not None and actual != expected):
        raise ValueError('missing or extra tensors/shards relative to index')
    return actual


def verify_export(base, adapter, merged):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    base, adapter, merged = (Path(p).resolve() for p in (base, adapter, merged))
    cfg = json.loads((adapter/'adapter_config.json').read_text())
    if (cfg.get('peft_type') != 'LORA' or cfg.get('r') != 8 or cfg.get('lora_alpha') != 16
            or Path(cfg.get('base_model_name_or_path', '')).resolve() != base
            or cfg.get('bias', 'none') != 'none' or cfg.get('use_dora') or cfg.get('use_rslora')
            or cfg.get('modules_to_save') or cfg.get('rank_pattern') or cfg.get('alpha_pattern')
            or cfg.get('fan_in_fan_out')):
        raise ValueError('adapter differs from planned plain rank-8 alpha-16 student LoRA')
    for field in ('model_type', 'hidden_size', 'num_hidden_layers', 'num_attention_heads',
                  'num_key_value_heads', 'intermediate_size', 'vocab_size', 'tie_word_embeddings'):
        if json.loads((base/'config.json').read_text()).get(field) != json.loads((merged/'config.json').read_text()).get(field):
            raise ValueError(f'export architecture changed: {field}')
    before, after = tensor_index(base), tensor_index(merged)
    if set(before) != set(after):
        raise ValueError('base/export tensor key coverage differs')
    tensors = load_file(str(adapter/'adapter_model.safetensors'), device='cpu')
    pairs = {}
    for key, tensor in tensors.items():
        if not tensor.is_floating_point() or not torch.isfinite(tensor).all().item():
            raise ValueError('nonfinite or nonfloating LoRA tensor')
        prefix = 'base_model.model.'
        if not key.startswith(prefix):
            raise ValueError('unknown adapter tensor prefix')
        tail = key[len(prefix):]
        for side in ('A', 'B'):
            suffix = f'.lora_{side}.weight'
            if tail.endswith(suffix):
                module = tail[:-len(suffix)] + '.weight'
                pairs.setdefault(module, {})[side] = tensor
                break
        else:
            raise ValueError('unsupported adapter tensor layout')
    if not pairs or any(set(v) != {'A','B'} for v in pairs.values()) or not set(pairs) <= set(before):
        raise ValueError('unpaired or unmatched LoRA tensors')
    updated = 0
    errors = {}
    for key in before:
        with safe_open(str(base/before[key]), framework='pt', device='cpu') as stream:
            old = stream.get_tensor(key)
        with safe_open(str(merged/after[key]), framework='pt', device='cpu') as stream:
            new = stream.get_tensor(key)
        if old.shape != new.shape or not old.is_floating_point() or not new.is_floating_point():
            raise ValueError(f'weight shape/dtype mismatch: {key}')
        if not torch.isfinite(old).all().item() or not torch.isfinite(new).all().item():
            raise ValueError(f'nonfinite model tensor: {key}')
        if key not in pairs:
            if not torch.equal(old, new):
                raise ValueError(f'non-LoRA tensor changed: {key}')
            continue
        a, b = pairs[key]['A'], pairs[key]['B']
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != 8 or b.shape[1] != 8 or (b.shape[0],a.shape[1]) != tuple(old.shape):
            raise ValueError(f'LoRA dimensions incompatible: {key}')
        delta = (b.float() @ a.float()) * 2
        expected = old.float() + delta
        error = (new.float()-expected).abs()
        # Allow the two rounding steps of BF16/FP16 merge; record the discrepancy.
        eps = max(torch.finfo(old.dtype).eps, torch.finfo(new.dtype).eps)
        tolerance = torch.maximum(old.float().abs(), expected.abs()) * (2 * eps) + 1e-7
        if not (error <= tolerance).all().item():
            raise ValueError(f'export does not match base + scaled LoRA update: {key}')
        if torch.count_nonzero(delta).item() and not torch.equal(old,new):
            updated += 1
        errors[key] = float(error.max().item())
    if not updated:
        raise ValueError('no actual student weight update')
    return {'status':'TENSOR_MERGE_VERIFIED','updated_modules':updated,'merged_tensors':len(after),
            'adapter_tensor_count':len(tensors),'merge_max_abs_error_by_module':errors,
            'adapter_sha256':sha256(adapter/'adapter_model.safetensors')}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--adapter',type=Path,required=True)
    p.add_argument('--merged',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise ValueError('verification output already exists')
    result=verify_export(args.base,args.adapter,args.merged)
    result['model_dir']=str(args.merged.resolve())
    result['files']=[{'path':str(f.relative_to(args.merged)),'bytes':f.stat().st_size,'sha256':sha256(f)}
                     for f in sorted(args.merged.rglob('*')) if f.is_file()]
    write_json(args.output,result)

if __name__=='__main__':main()
