"""Frozen Week8 protocol preparation; never mutates source conversations."""
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import unicodedata
from common import ROOT, read_records, sha256, write_json


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def input_path(relative):
    """Resolve a frozen logical input name to its packaged storage location."""
    mapping = json.loads((ROOT/'data/input_paths.json').read_text())
    target = mapping.get(relative, {}).get('path', relative)
    path = ROOT/target
    if Path(target).is_absolute() or '..' in Path(target).parts or not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('input path escapes the repository')
    return path


def verify_entry(entry):
    path = input_path(entry['path'])
    if not path.is_file() or sha256(path) != entry['sha256']:
        raise ValueError(f'hash mismatch or missing input: {path}')
    return path


def messages_from_sharegpt(row):
    turns = row.get('conversations')
    if not isinstance(turns, list) or not turns:
        raise ValueError('missing_conversations')
    messages = []
    if row.get('system'):
        messages.append({'role': 'system', 'content': row['system']})
    roles = {'human': 'user', 'gpt': 'assistant', 'system': 'system'}
    for turn in turns:
        if not isinstance(turn, dict) or turn.get('from') not in roles:
            raise ValueError('unsupported_role')
        messages.append({'role': roles[turn['from']], 'content': turn.get('value')})
    if any(not isinstance(m['content'], str) or not m['content'].strip() for m in messages):
        raise ValueError('empty_or_invalid_content')
    body = messages[1:] if messages[0]['role'] == 'system' else messages
    if not body or len(body) % 2 or any(m['role'] != ('user' if n % 2 == 0 else 'assistant') for n, m in enumerate(body)):
        raise ValueError('invalid_role_order')
    return messages


def to_sharegpt(messages):
    system = messages[0]['content'] if messages[0]['role'] == 'system' else ''
    body = messages[1:] if system else messages
    return {'system': system, 'conversations': [{'from': 'human' if m['role'] == 'user' else 'gpt', 'value': m['content']} for m in body]}


def to_alpaca(messages):
    system = messages[0]['content'] if messages[0]['role'] == 'system' else ''
    body = messages[1:] if system else messages
    return {'system': system, 'instruction': body[-2]['content'], 'input': '', 'output': body[-1]['content'], 'history': [[body[n]['content'], body[n+1]['content']] for n in range(0, len(body)-2, 2)]}


def from_alpaca(row):
    messages = [{'role': 'system', 'content': row['system']}] if row['system'] else []
    for user, assistant in row['history']:
        messages.extend([{'role': 'user', 'content': user}, {'role': 'assistant', 'content': assistant}])
    messages.extend([{'role': 'user', 'content': row['instruction']}, {'role': 'assistant', 'content': row['output']}])
    if row['input']:
        raise ValueError('protocol requires empty Alpaca input')
    return messages


def evaluation_rows(payload):
    if isinstance(payload, list):
        result = payload
    elif isinstance(payload, dict):
        keys = [k for k in ('items', 'questions', 'prompts', 'records') if isinstance(payload.get(k), list)]
        if len(keys) != 1:
            raise ValueError('unknown or ambiguous evaluation schema')
        result = payload[keys[0]]
    else:
        raise ValueError('invalid evaluation payload')
    if not result:
        raise ValueError('empty evaluation input')
    return result


def extract_questions(payload):
    result = []
    for index, row in enumerate(evaluation_rows(payload)):
        rid = row.get('id', row.get('sample_id', str(index+1)))
        texts = []
        if 'messages' in row:
            texts = [(n, m.get('content')) for n, m in enumerate(row['messages']) if m.get('role') == 'user']
        elif 'conversations' in row:
            texts = [(n, m.get('value')) for n, m in enumerate(row['conversations']) if m.get('from') == 'human']
        elif 'text' in row:
            texts = [(0, row['text'])]
        if not texts or any(not isinstance(t, str) or not t.strip() for _, t in texts):
            raise ValueError(f'missing evaluation question: {rid}')
        result.extend({'id': rid, 'turn': n, 'text': t} for n, t in texts)
    return result


def normalize(text):
    return ''.join(unicodedata.normalize('NFKC', text).split()).casefold()


@lru_cache(maxsize=20000)
def grams(text):
    return frozenset(text[n:n+5] for n in range(len(text)-4))


def match_normalized(a, b):
    if not a or not b:
        return None
    if a == b:
        return {'kind': 'exact', 'score': 1.0}
    if min(len(a), len(b)) < 32:
        return None
    if a in b or b in a:
        return {'kind': 'containment', 'score': 1.0}
    ga, gb = grams(a), grams(b)
    if min(len(ga), len(gb)) < .8 * max(len(ga), len(gb)):
        return None
    shared = len(ga & gb)
    union = len(ga) + len(gb) - shared
    if shared >= .8 * union:
        return {'kind': 'char5_jaccard', 'score': shared / union}
    return None


def match_text(left, right):
    return match_normalized(normalize(left), normalize(right))


def prepare_records(raw, lineage, tokenizer, protected, links, max_tokens=2048):
    if len(raw) != len(lineage):
        raise ValueError('source/lineage count mismatch')
    ids = [x['sample_id'] for x in lineage]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate lineage IDs')
    records, excluded, lengths = {}, [], []
    for row, origin in zip(raw, lineage):
        sid = origin['sample_id']
        try:
            messages = messages_from_sharegpt(row)
            length = len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
            lengths.append({'sample_id': sid, 'tokens': length})
            if length > max_tokens:
                excluded.append({'sample_id': sid, 'reason': 'overlength', 'tokens': length})
                continue
            records[sid] = {'messages': messages, 'tokens': length, 'source': origin['source'], 'raw_sha256': digest(row), 'content_sha256': digest(messages), 'lineage': origin}
        except ValueError as exc:
            excluded.append({'sample_id': sid, 'reason': 'invalid_record', 'detail': str(exc)})
    clean_count = len(records)
    survivors, aliases = {}, {}
    for sid in sorted(records):
        key = records[sid]['content_sha256']
        if key in survivors:
            aliases[sid] = survivors[key]
            excluded.append({'sample_id': sid, 'reason': 'exact_duplicate', 'duplicate_of': survivors[key]})
        else:
            survivors[key] = sid
    for sid in aliases:
        del records[sid]
    parent = {sid: sid for sid in records}
    def find(s):
        while parent[s] != s:
            parent[s] = parent[parent[s]]; s = parent[s]
        return s
    edges = []
    def union(a, b, evidence):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
        edges.append({'left': a, 'right': b, **evidence})
    for link in links:
        a, b = aliases.get(link['left'], link['left']), aliases.get(link['right'], link['right'])
        if a in records and b in records and a != b:
            union(a, b, {'kind': 'historical_link'})
    # Every candidate turn is checked, including answers; evaluation side uses user questions.
    indexed = [(sid, n, normalize(m['content'])) for sid in sorted(records) for n, m in enumerate(records[sid]['messages'])]
    for n, (a, ta, text) in enumerate(indexed):
        for b, tb, other in indexed[:n]:
            if a == b:
                continue
            hit = match_normalized(text, other)
            if hit:
                union(a, b, {**hit, 'left_turn': ta, 'right_turn': tb})
    hits = []
    # Deduplicate question text while retaining every origin for audit.
    pmap = defaultdict(list)
    for p in protected:
        pmap[normalize(p['text'])].append({k: v for k, v in p.items() if k != 'text'})
    for sid, turn, text in indexed:
        for question, origins in pmap.items():
            hit = match_normalized(text, question)
            if hit:
                hits.append({'sample_id': sid, 'turn': turn, **hit, 'protected_origins': origins})
    hit_roots = {find(h['sample_id']) for h in hits}
    all_groups = defaultdict(list)
    for sid in sorted(records):
        all_groups[find(sid)].append(sid)
    for sid in list(records):
        if find(sid) in hit_roots:
            excluded.append({'sample_id': sid, 'reason': 'protected_question_group', 'group_members': all_groups[find(sid)]})
            del records[sid]
    groups = sorted(g for root, g in all_groups.items() if root not in hit_roots)
    return {'records': records, 'excluded': sorted(excluded, key=lambda x: x['sample_id']), 'groups': groups, 'edges': edges, 'hits': hits, 'lengths': lengths, 'clean_count': clean_count, 'deduplicated_count': len(survivors), 'protected_unique_questions': len(pmap)}


def group_id(group):
    return hashlib.sha256('\n'.join(sorted(group)).encode()).hexdigest()


def split_groups(groups):
    ordered = sorted([sorted(g) for g in groups], key=lambda g: hashlib.sha256(('week8-v1:42:' + group_id(g)).encode()).hexdigest())
    total = sum(map(len, ordered)); target = math.floor(total*.1+.5)
    reachable = {0: ()}
    for n, group in enumerate(ordered):
        for count, chosen in sorted(list(reachable.items())):
            next_count = count+len(group)
            if next_count not in reachable:
                reachable[next_count] = chosen+(n,)
    size = min(reachable, key=lambda x: (abs(x-target), x))
    selected = set(reachable[size])
    val = sorted(sid for n, g in enumerate(ordered) if n in selected for sid in g)
    train = sorted(sid for n, g in enumerate(ordered) if n not in selected for sid in g)
    return train, val


def load_protocol(path):
    protocol = json.loads(Path(path).read_text())
    supported = {'seed': 42, 'max_tokens': 2048, 'overlength_policy': 'reject_whole_record_no_truncation', 'training_format': 'sharegpt', 'delivery_format': 'alpaca_with_history_and_system', 'normalization_for_matching_only': 'NFKC_remove_unicode_whitespace_casefold', 'containment_min_chars': 32, 'near_duplicate_ngram': 5, 'near_duplicate_jaccard': .8, 'lf_template': 'qwen', 'train_on_prompt': False, 'mask_history': False, 'packing': False}
    for k, value in supported.items():
        if protocol[k] != value:
            raise ValueError(f'unsupported protocol setting: {k}')
    expected_split = {'validation_ratio': .1, 'rounding': 'floor(N*0.1+0.5)', 'unit': 'connected_question_group', 'algorithm': 'reachable_subset_sum_nearest_target_tie_smaller_then_first_reachable', 'hash_salt': 'week8-v1:42:', 'new_split_of_historical_train_only': True}
    if protocol['split'] != expected_split:
        raise ValueError('unsupported split protocol')
    entries = [protocol['source'], protocol['lineage'], protocol['historical_group_links']] + protocol['protected_files'] + protocol['tokenizer_files']
    for e in entries:
        verify_entry(e)
    verify_entry({'path': protocol['rules_document'], 'sha256': protocol['rules_sha256']})
    return protocol, entries


def run(protocol_path, output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError('use a new output directory')
    protocol, entries = load_protocol(protocol_path)
    import transformers
    if transformers.__version__ != protocol['transformers_version']:
        raise ValueError('transformers version does not match protocol')
    tokenizer_dir = input_path(protocol['tokenizer_files'][0]['path']).parent
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, use_fast=True)
    if not tokenizer.is_fast or hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() != protocol['chat_template_sha256']:
        raise ValueError('tokenizer or chat template mismatch')
    raw = read_records(input_path(protocol['source']['path']))
    lineage = read_records(input_path(protocol['lineage']['path']))
    if len(raw) != protocol['source']['records'] or len(lineage) != protocol['lineage']['records']:
        raise ValueError('source count mismatch')
    protected, coverage = [], []
    for e in protocol['protected_files']:
        if e['role'] == 'custom20_fixed_rubric':
            coverage.append({**e, 'question_count': 0, 'status': 'rubric_hashed_not_a_question_source'})
            continue
        path = input_path(e['path'])
        payload = read_records(path) if path.suffix == '.jsonl' else json.loads(path.read_text())
        questions = extract_questions(payload)
        count = len(evaluation_rows(payload))
        if count != e['records']:
            raise ValueError(f'evaluation count mismatch: {path}: {count} != {e["records"]}')
        coverage.append({**e, 'question_count': len(questions), 'status': 'parsed'})
        protected.extend({**q, 'path': e['path']} for q in questions)
    links = read_records(input_path(protocol['historical_group_links']['path']))
    result = prepare_records(raw, lineage, tokenizer, protected, links, protocol['max_tokens'])
    train, validation = split_groups(result['groups'])
    if not train or not validation:
        raise ValueError('not enough independent data to produce nonempty train and validation')
    records = result['records']
    membership = {sid: name for name, ids in [('train', train), ('validation', validation)] for sid in ids}
    if set(train) & set(validation) or len(membership) != len(records):
        raise ValueError('split membership mismatch')
    if any(len({membership[sid] for sid in g}) != 1 for g in result['groups']):
        raise ValueError('group crosses splits')
    payloads, aligned = {}, []
    for name, ids in [('train', train), ('validation', validation)]:
        sg = [to_sharegpt(records[s]['messages']) for s in ids]
        al = [to_alpaca(records[s]['messages']) for s in ids]
        for n, (sid, s, a) in enumerate(zip(ids, sg, al)):
            if messages_from_sharegpt(s) != records[sid]['messages'] or from_alpaca(a) != records[sid]['messages']:
                raise ValueError('dual-format content mismatch')
            aligned.append({'sample_id': sid, 'split': name, 'row_index': n, 'source': records[sid]['source'], 'raw_sha256': records[sid]['raw_sha256'], 'content_sha256': records[sid]['content_sha256'], 'tokens': records[sid]['tokens'], 'lineage': records[sid]['lineage']})
        payloads[name+'_sharegpt.json'] = sg
        payloads[name+'_alpaca.json'] = al
    payloads['dataset_info.json'] = {f'week8_{name}': {'file_name': f'{name}_sharegpt.json', 'formatting': 'sharegpt', 'columns': {'messages': 'conversations', 'system': 'system'}, 'tags': {'role_tag': 'from', 'content_tag': 'value', 'user_tag': 'human', 'assistant_tag': 'gpt'}} for name in ['train', 'validation']}
    payloads['lineage.json'] = aligned
    payloads['groups.json'] = [{'group_id': group_id(g), 'members': g, 'split': membership[g[0]]} for g in result['groups']]
    payloads['exclusions.json'] = result['excluded']
    payloads['group_edges.json'] = result['edges']
    payloads['protected_hits.json'] = result['hits']
    payloads['protected_coverage.json'] = coverage
    payloads['lengths.json'] = result['lengths']
    payloads['protocol.json'] = protocol
    payloads['input_receipt.json'] = entries
    source_by_id = {l['sample_id']: row for l, row in zip(lineage, raw)}
    if any(messages_from_sharegpt(source_by_id[s]) != records[s]['messages'] for s in records):
        raise ValueError('source conversation changed')
    remaining_hits = [h for h in result['hits'] if h['sample_id'] in records]
    if remaining_hits:
        raise ValueError('protected hit escaped quarantine')
    # Catch concurrent mutations before writing anything.
    for e in entries:
        verify_entry(e)
    reasons = Counter(x['reason'] for x in result['excluded'])
    def describe(ids):
        lengths = sorted(records[s]['tokens'] for s in ids)
        return {'count': len(ids), 'sources': dict(sorted(Counter(records[s]['source'] for s in ids).items())), 'multiturn': sum(sum(m['role']=='assistant' for m in records[s]['messages']) > 1 for s in ids), 'system_records': sum(records[s]['messages'][0]['role']=='system' for s in ids), 'tokens': {'min': min(lengths), 'median': lengths[len(lengths)//2], 'p95': lengths[math.ceil(.95*len(lengths))-1], 'max': max(lengths), 'mean': sum(lengths)/len(lengths)}}
    stats = {'status': 'TASK3_MACHINE_CHECKS_PASS_NOT_DATA_READY', 'mode': 'protocol_formal_tokenizer', 'protocol_id': protocol['protocol_id'], 'protocol_sha256': sha256(protocol_path), 'source_sha256': protocol['source']['sha256'], 'raw_count': len(raw), 'valid_within_length': result['clean_count'], 'after_exact_dedup': result['deduplicated_count'], 'clean_count': len(records), 'exclusion_reasons': dict(sorted(reasons.items())), 'rejected': len(result['excluded']), 'truncated': 0, 'train_count': len(train), 'validation_count': len(validation), 'validation_ratio': len(validation)/len(records), 'groups': len(result['groups']), 'split_details': {'train': describe(train), 'validation': describe(validation)}, 'protected_unique_questions': result['protected_unique_questions'], 'protected_hit_pairs_before_quarantine': len(result['hits']), 'protected_hits_after_quarantine': 0, 'cross_split_group_count': 0, 'unchanged_source_records': len(records), 'dual_format_aligned_records': len(aligned), 'tokenizer': str(Path(protocol['tokenizer_files'][0]['path']).parent), 'max_length': max(r['tokens'] for r in records.values()), 'pending_gates': ['content_sample_review', 'actual_lf_loader', 'benchmark_snapshot_and_judge_runtime_lock', 'final_release'], 'gpu_started': False}
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, payload in payloads.items():
        write_json(output_dir/name, payload)
    stats['outputs'] = {name: sha256(output_dir/name) for name in sorted(payloads)}
    write_json(output_dir/'statistics.json', stats)
    return stats
