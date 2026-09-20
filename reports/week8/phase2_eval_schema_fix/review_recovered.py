"""Read-only review of the preserved run using the corrected production validator."""
import collections
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
import step3_eval as evaluation


def review():
    recovered = ROOT / 'reports/week8/phase2_formal_inference/recovered_failure'
    manifest = json.loads((recovered / 'manifest.json').read_text())
    for item in manifest:
        name = pathlib.PurePosixPath(item['path'])
        assert not name.is_absolute() and '..' not in name.parts
        data = (recovered / name).read_bytes()
        assert len(data) == item['bytes']
        assert hashlib.sha256(data).hexdigest() == item['sha256']
    records_path = ROOT / 'reports/week8/phase2_preflight/benchmark_records.json'
    assert evaluation.sha256(records_path) == '2d2a88a2f6afd2870cb1efcfc19f1c5c2eb5e33c45297532d50cdcae7afcfa32'
    records = json.loads(records_path.read_text())
    oc = recovered / 'formal-eval-20260916-01/original_base/opencompass'
    summaries = {}
    for dataset, subjects, expected_rows in [('ceval', 52, 1346), ('cmmlu', 67, 11582)]:
        golds = collections.defaultdict(list)
        for row in records:
            if row['benchmark'] == dataset and row['split'] == ('val' if dataset == 'ceval' else 'test'):
                golds[row['subject']].append(row)
        paths = sorted(oc.glob(f'*/results/*/{dataset}-*.json'))
        assert len(paths) == subjects == len(golds)
        assert {p.stem for p in paths} == {f'{dataset}-{s}' for s in golds}
        total = correct = 0
        for path in paths:
            subject = path.stem.removeprefix(dataset + '-')
            ordered = sorted(golds[subject], key=lambda r: r['row_index'])
            assert [r['row_index'] for r in ordered] == list(range(len(ordered)))
            n, hits = evaluation.validate_subject_result(
                json.loads(path.read_text()), [r['answer'] for r in ordered], subject_id=path.stem)
            total += n
            correct += hits
        assert total == expected_rows
        summaries[dataset] = dict(subjects=subjects, questions=total, correct=correct, accuracy=100*correct/total)
    return dict(status='RECOVERED_OBJECTIVE_VALIDATION_PASS', manifest_files=len(manifest),
                validator_sha256=evaluation.sha256(ROOT/'scripts/step3_eval.py'),
                records_sha256=evaluation.sha256(records_path), results=summaries,
                gpu_or_api_used=False, formal_evaluation_complete=False)


if __name__ == '__main__':
    print(json.dumps(review(), ensure_ascii=False, indent=2))
