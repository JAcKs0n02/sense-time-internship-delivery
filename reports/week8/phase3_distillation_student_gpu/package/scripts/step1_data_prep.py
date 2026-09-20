#!/usr/bin/env python3
"""Reuse the audited Day7 cleaning rules and add deterministic grouped splits."""
import argparse
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
from common import ROOT, load_module, read_records, sha256, write_json

clean = load_module('week8_day7_cleaner', ROOT/'deliverables/week2/day7/clean_pipeline.py')


class CharacterSmokeTokenizer:
    """Offline plumbing check only; never a substitute for Qwen token lengths."""
    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, ids, **kwargs):
        return ''.join(map(chr, ids))

    def apply_chat_template(self, messages, **kwargs):
        return self.encode(''.join('<'+m['role']+'>'+m['content']+'</end>' for m in messages))


def prompt_key(record):
    return json.dumps([m for m in record.messages if m['role'] != 'assistant'], ensure_ascii=False, sort_keys=True)


def prepare(args):
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('output directory must be empty; use a new run directory')
    if args.quick:
        tokenizer = CharacterSmokeTokenizer()
    else:
        if not args.tokenizer:
            raise ValueError('formal preparation requires --tokenizer')
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    raw = read_records(args.input)
    records, audits = [], []
    for i, row in enumerate(raw):
        sid = f'source-{i:06d}'
        try:
            parser = clean.parse_sharegpt_record if 'conversations' in row else clean.parse_alpaca_record
            parsed = parser(row, sid)
            normalized, changes = clean.clean_record_text(parsed)
            messages, truncation = clean.truncate_messages(normalized.messages, tokenizer, args.max_tokens)
            record = replace(normalized, messages=messages, was_truncated=truncation['truncated'])
            records.append(record)
            audits.append({'id': sid, 'status': 'cleaned', **changes, **truncation})
        except clean.RecordRejected as exc:
            audits.append({'id': sid, 'status': 'rejected', 'reason': exc.reason})
    records, exact = clean.exact_deduplicate(records)
    records, fuzzy, candidates = clean.fuzzy_deduplicate(records, threshold=3)
    groups = defaultdict(list)
    for record in records:
        groups[prompt_key(record)].append(record)
    keys = sorted(groups)
    if len(keys) < 2:
        raise ValueError('need at least two independent prompts for train/validation split')
    random.Random(args.seed).shuffle(keys)
    # Split whole prompt groups: no answer variants of a question across splits.
    val_count = max(1, round(len(keys)*0.1))
    val_keys = set(keys[:val_count])
    splits = {'train': [], 'validation': []}
    for key in keys:
        splits['validation' if key in val_keys else 'train'].extend(groups[key])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        for fmt, render in [('alpaca', clean._render_alpaca), ('sharegpt', clean._render_sharegpt)]:
            write_json(args.output_dir/f'{name}_{fmt}.json', [dict(render(r), sample_id=r.sample_id) for r in items])
    write_json(args.output_dir/'dataset_info.json', {
        f'week8_{split}': {'file_name': f'{split}_alpaca.json', 'columns': {'prompt': 'instruction', 'query': 'input', 'response': 'output'}} for split in splits
    })
    train_keys = {prompt_key(r) for r in splits['train']}
    validation_keys = {prompt_key(r) for r in splits['validation']}
    stats = {
        'mode': 'quick_character_smoke' if args.quick else 'formal_tokenizer',
        'tokenizer': 'CharacterSmokeTokenizer' if args.quick else args.tokenizer,
        'source': str(args.input), 'source_sha256': sha256(args.input), 'seed': args.seed,
        'raw_count': len(raw), 'clean_count': len(records),
        'rejected': sum(a['status']=='rejected' for a in audits),
        'exact_duplicates': len(exact), 'fuzzy_duplicates': len(fuzzy),
        'train_count': len(splits['train']), 'validation_count': len(splits['validation']),
        'split_policy': '90:10 shuffled prompt groups; integer rounding; same prompt stays together',
        'prompt_overlap': len(train_keys & validation_keys),
        'max_length': max(clean.chat_token_length(r.messages, tokenizer) for r in records),
        'length_unit': 'unicode_character_smoke_template' if args.quick else 'model_chat_template_tokens',
        'outputs': {p.name: sha256(p) for p in sorted(args.output_dir.glob('*.json'))},
    }
    write_json(args.output_dir/'audit.json', {'records': audits, 'exact_duplicates': exact, 'fuzzy_duplicates': fuzzy})
    write_json(args.output_dir/'statistics.json', stats)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, help='Frozen protocol; default for formal preparation')
    parser.add_argument('--legacy', action='store_true', help='Explicit historical data preparation only')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--tokenizer')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--max-tokens', type=int)
    args = parser.parse_args()
    if args.protocol and (args.quick or args.legacy):
        parser.error('cannot combine protocol with quick/legacy mode')
    if args.protocol or not (args.quick or args.legacy):
        if any(x is not None for x in [args.input, args.tokenizer, args.seed, args.max_tokens]):
            parser.error('cannot combine protocol with input/tokenizer/seed/length overrides')
        from week8_data import run
        stats = run(args.protocol or ROOT/'configs/week8_data_protocol.json', args.output_dir)
    else:
        args.input = args.input or ROOT/'deliverables/week2/day6/source/data/formatted/week2_5k_sharegpt.jsonl'
        args.seed = 42 if args.seed is None else args.seed
        args.max_tokens = 2048 if args.max_tokens is None else args.max_tokens
        stats = prepare(args)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
