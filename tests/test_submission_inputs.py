import copy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import week8_data as data


def test_submission_has_no_recovery_or_temporary_root():
    assert not (ROOT/'.worktrees').exists()
    assert not (ROOT/'outputs').exists()
    paths = json.loads((ROOT/'data/input_paths.json').read_text())
    assert len(paths) == 23
    for old, entry in paths.items():
        path = data.input_path(old)
        assert path.is_relative_to(ROOT/'data')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']


@pytest.mark.parametrize('target', ['../../outside.json', '/tmp/outside.json'])
def test_mapped_input_cannot_escape_repository(tmp_path, target):
    (tmp_path/'data').mkdir()
    (tmp_path/'data/input_paths.json').write_text(json.dumps({'legacy': {'path':target}}))
    with patch.object(data, 'ROOT', tmp_path):
        with pytest.raises(ValueError, match='escapes'):
            data.input_path('legacy')


def test_mapped_input_tamper_still_rejected(tmp_path):
    (tmp_path/'data').mkdir()
    (tmp_path/'data/current.json').write_text('[]')
    (tmp_path/'data/input_paths.json').write_text(json.dumps({'legacy': {'path':'data/current.json'}}))
    with patch.object(data, 'ROOT', tmp_path):
        with pytest.raises(ValueError, match='hash mismatch'):
            data.verify_entry({'path':'legacy', 'sha256':'0'*64})


def test_final_answers_remain_bound_to_original_requests():
    from week8_pipeline_score import load_release
    from submission_evidence import validate_plan
    plan = copy.deepcopy(load_release('original_base'))
    body = plan['requests'][0]['body']
    payload = json.loads(body['contents'][0]['parts'][0]['text'])
    payload['candidate_answer'] = 'changed answer'
    body['contents'][0]['parts'][0]['text'] = json.dumps(payload)
    with pytest.raises(ValueError, match='binding changed'):
        validate_plan('original_base', plan)
