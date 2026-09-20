#!/usr/bin/env python3
"""Independent candidate prompt audit; does NOT certify OpenCompass runtime."""
import ast
import csv
import hashlib
import json
from pathlib import Path
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'reports/week8/phase2_preflight'
OUT = ROOT / 'reports/week8/phase2_eval_hardening'

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def expand_candidate(dataset, filename):
    path = BASE / 'package_source/opencompass/configs/datasets' / dataset / filename
    # Resolve the published Python configuration with import names represented as
    # strings. This is a static expansion, explicitly not mmengine registration.
    tree = ast.parse(path.read_text())
    namespace = {}
    body = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                namespace[alias.asname or alias.name] = node.module + '.' + alias.name
        else:
            body.append(node)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[dataset + '_datasets']

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer_path = ROOT / 'logs/week8-real-loader-20260915/attempt-02/tokenizer'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    all_configs, audit = [], []
    newline_changes = 0
    for dataset, filename in [('ceval', 'ceval_gen_5f30c7.py'), ('cmmlu', 'cmmlu_gen_c13365.py')]:
        for config in expand_candidate(dataset, filename):
            subject = config['name']
            config['path'] = str((BASE / 'local_data' / dataset).relative_to(ROOT))
            all_configs.append(config)
            split_rows = {}
            for split in ('dev', 'val' if dataset == 'ceval' else 'test'):
                path = BASE / 'local_data' / dataset / split / (f'{subject}_{split}.csv' if dataset == 'ceval' else f'{subject}.csv')
                with path.open(encoding='utf-8') as stream:
                    rows = list(csv.DictReader(stream))
                with path.open(encoding='utf-8', newline='') as stream:
                    raw_rows = list(csv.DictReader(stream))
                newline_changes += sum(a != b for a, b in zip(rows, raw_rows))
                if dataset == 'cmmlu':
                    rows = [{**r, 'question': r['Question'], 'answer': r['Answer']} for r in rows]
                split_rows[split] = rows
            rounds = config['infer_cfg']['ice_template']['template']['round']
            indices = config['infer_cfg']['retriever']['fix_id_list']
            assert indices == [0, 1, 2, 3, 4] and len(split_rows['dev']) == 5
            shots = []
            for i in indices:
                for item in rounds:
                    shots.append({'role': 'user' if item['role'] == 'HUMAN' else 'assistant', 'content': item['prompt'].format(**split_rows['dev'][i])})
            test_split = config['reader_cfg']['test_split']
            for i, row in enumerate(split_rows[test_split]):
                # Last BOT is the generation role, excluded from input. Actual
                # API parser equivalence remains a separate runtime gate.
                messages = shots + [{'role': 'user', 'content': rounds[0]['prompt'].format(**row)}]
                assert len(messages) == 11
                ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                assert ids == tokenizer(rendered, add_special_tokens=False)['input_ids']
                audit.append({'dataset': dataset, 'subject': subject, 'row_index': i, 'fewshot_count': 5, 'input_tokens': len(ids), 'token_ids_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest()})
    assert len(all_configs) == 119 and len(audit) == 12928
    (OUT / 'candidate_datasets.json').write_text(json.dumps(all_configs, ensure_ascii=False, indent=2)+'\n')
    (OUT / 'prompt_lengths.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
    summary = {'status': 'INDEPENDENT_PROMPT_RECONSTRUCTION_CHECKED_RUNTIME_PENDING', 'actual_opencompass_runtime_verified': False, 'subjects': len(all_configs), 'questions': len(audit), 'csv_rows_changed_by_universal_newline': newline_changes, 'input_token_max': max(r['input_tokens'] for r in audit), 'over_input_2048': sum(r['input_tokens'] > 2048 for r in audit), 'over_total_2048_with_32_output': sum(r['input_tokens'] + 32 > 2048 for r in audit), 'proposed_context_limit': 2048, 'evaluation_output_limit': 32, 'tokenizer_files': {p.name: digest(p) for p in tokenizer_path.iterdir() if p.is_file()}, 'artifacts': {n: digest(OUT/n) for n in ('candidate_datasets.json', 'prompt_lengths.json')}, 'limitations': ['Independent reconstruction; actual registered dataset/retriever/parser and model wrapper must reproduce token hashes before runtime lock.', 'Retain evaluation context 2048 and output 32; target runtime must verify no example removal or truncation. SFT cutoff stays 2048.']}
    (OUT / 'prompt_audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('tokenizer_files','artifacts')},ensure_ascii=False,indent=2))

if __name__ == '__main__': main()
