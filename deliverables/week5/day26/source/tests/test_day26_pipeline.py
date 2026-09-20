import importlib.util
import base64
import hashlib
import json
import subprocess
from pathlib import Path

import pytest


DAY26 = Path(__file__).resolve().parents[2]
SCRIPTS = DAY26 / "source" / "scripts"


def load_script(name: str):
    path = SCRIPTS / f"{name}.py"
    assert path.exists(), f"missing Day26 script: {path}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_facts():
    build = load_script("build_vlm_dataset")
    facts = []
    split_sizes = {"train": 10, "dev": 2, "final": 2}
    for category in build.CATEGORIES:
        for split, size in split_sizes.items():
            for index in range(size):
                image_id = f"{category}-{split}-{index:02d}"
                facts.append(
                    {
                        "image_id": image_id,
                        "split": split,
                        "category": category,
                        "image_relative_path": f"images/{split}/{image_id}.jpg",
                        "source_page_url": f"https://example.org/page/{image_id}",
                        "download_url": f"https://example.org/file/{image_id}.jpg",
                        "author": "Example Author",
                        "license": "CC BY 4.0",
                        "license_url": "https://creativecommons.org/licenses/by/4.0/",
                        "sha256": f"{len(facts) + 1:064x}",
                        "summary": f"这是 {image_id} 的可见内容摘要。",
                        "subject": f"主体 {image_id}",
                        "details": ["细节一", "细节二"],
                        "visible_text": "SAMPLE TEXT",
                        "explanation": f"依据图中可见证据解释 {image_id}。",
                        "false_claim": f"图中没有 {image_id}",
                        "correction": f"该说法不正确，图中可见 {image_id}。",
                    }
                )
    return facts


def test_source_facts_require_traceable_real_image_metadata():
    validate = load_script("validate_vlm_dataset")
    fact = make_facts()[0]
    validate.validate_source_fact(fact)

    for field in (
        "source_page_url",
        "download_url",
        "author",
        "license",
        "license_url",
        "sha256",
    ):
        bad = dict(fact)
        bad[field] = ""
        with pytest.raises(ValueError, match=field):
            validate.validate_source_fact(bad)


def test_acquisition_spec_has_70_pinned_real_sources():
    acquire = load_script("acquire_source_images")
    specs = acquire.SOURCE_SPECS
    assert len(specs) == 70
    assert len({row["image_id"] for row in specs}) == 70
    assert all(len(row["revision"]) == 40 for row in specs)
    assert all(row["source_page_url"].startswith("https://github.com/") for row in specs)
    assert all(row["download_url"].startswith("https://raw.githubusercontent.com/") for row in specs)
    for category in acquire.CATEGORIES:
        rows = [row for row in specs if row["category"] == category]
        assert [row["split"] for row in rows].count("train") == 10
        assert [row["split"] for row in rows].count("dev") == 2
        assert [row["split"] for row in rows].count("final") == 2


def test_downloader_uses_bounded_retrying_curl_without_a_shell(monkeypatch):
    acquire = load_script("acquire_source_images")
    captured = {}

    class Result:
        stdout = b"image-bytes"

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return Result()

    monkeypatch.setattr(acquire.subprocess, "run", fake_run)
    assert acquire._download("https://raw.githubusercontent.com/example/image.png") == b"image-bytes"
    assert captured["command"][0] == "curl"
    assert "-4" in captured["command"]
    assert "--retry-all-errors" in captured["command"]
    assert captured["kwargs"]["check"] is True
    assert "shell" not in captured["kwargs"]


def test_acquisition_rejects_bytes_that_do_not_match_bundled_manifest():
    acquire = load_script("acquire_source_images")
    raw = b"pinned-image"
    expected = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "file_bytes": len(raw),
    }

    acquire.verify_expected_bytes("image-01", raw, expected)
    with pytest.raises(ValueError, match="expected manifest"):
        acquire.verify_expected_bytes("image-01", b"tampered", expected)


def test_downloader_falls_back_to_pinned_github_blob_api(monkeypatch):
    acquire = load_script("acquire_source_images")
    spec = dict(acquire.SOURCE_SPECS[0])
    expected = b"pinned-image-bytes"
    calls = []

    def fake_download(url):
        calls.append(url)
        if url == spec["download_url"]:
            raise subprocess.CalledProcessError(28, ["curl"])
        if url == spec["api_url"]:
            return json.dumps({"git_url": "https://api.github.com/repos/x/y/git/blobs/abc"}).encode()
        return json.dumps(
            {"encoding": "base64", "content": base64.b64encode(expected).decode()}
        ).encode()

    monkeypatch.setattr(acquire, "_download", fake_download)
    assert acquire.download_source(spec) == expected
    assert calls == [
        spec["download_url"],
        spec["api_url"],
        "https://api.github.com/repos/x/y/git/blobs/abc",
    ]


def test_reviewed_annotations_cover_all_70_images_without_placeholders():
    acquire = load_script("acquire_source_images")
    facts = load_script("build_source_facts")
    assert set(facts.ANNOTATIONS) == {row["image_id"] for row in acquire.SOURCE_SPECS}
    assert len(facts.ANNOTATIONS) == 70
    for image_id, annotation in facts.ANNOTATIONS.items():
        assert set(annotation) == {"subject", "details", "visible_text"}, image_id
        assert annotation["subject"].strip()
        assert len(annotation["details"]) >= 2
        assert all(detail.strip() for detail in annotation["details"])
        assert annotation["visible_text"].strip()
        assert not any("todo" in str(value).casefold() for value in annotation.values())


def test_builder_produces_exact_200_20_20_balanced_contract():
    build = load_script("build_vlm_dataset")
    records = build.build_internal_records(make_facts())

    assert {split: len(rows) for split, rows in records.items()} == {
        "train": 200,
        "dev": 20,
        "final": 20,
    }
    for category in build.CATEGORIES:
        assert sum(row["category"] == category for row in records["train"]) == 40


def test_each_train_image_has_four_modes_and_heldout_has_two():
    build = load_script("build_vlm_dataset")
    records = build.build_internal_records(make_facts())
    expected_modes = {
        "train": {"direct", "structured", "explanation", "correction"},
        "dev": {"direct", "correction"},
        "final": {"direct", "correction"},
    }
    for split, rows in records.items():
        by_image = {}
        for row in rows:
            by_image.setdefault(row["source_image_id"], set()).add(row["mode"])
        assert all(modes == expected_modes[split] for modes in by_image.values())


def test_llamafactory_export_has_one_image_token_and_one_image_path():
    build = load_script("build_vlm_dataset")
    internal = build.build_internal_records(make_facts())["train"][:3]
    exported = build.to_llamafactory_records(internal)

    for row in exported:
        assert row["messages"][0]["role"] == "user"
        assert row["messages"][1]["role"] == "assistant"
        assert row["messages"][0]["content"].count("<image>") == 1
        assert len(row["images"]) == 1
        assert set(row) == {"messages", "images"}


def test_internal_records_keep_the_teacher_audit_contract():
    build = load_script("build_vlm_dataset")
    row = build.build_internal_records(make_facts())["train"][0]
    required = {
        "id",
        "image_relative_path",
        "image_sha256",
        "task_type",
        "user_instruction",
        "target_answer",
        "source",
        "license",
        "split",
        "construction_method",
    }
    assert required <= set(row)
    assert row["construction_method"] == "human-reviewed-source-fact+deterministic-template"


def test_grounded_builder_keeps_200_unique_pairs_and_weights_primary_modes():
    grounded = load_script("build_grounded_train")
    facts = [row for row in make_facts() if row["split"] == "train"]
    unique, weighted = grounded.build_grounded_records(facts)

    assert len(unique) == 200
    assert len({row["id"] for row in unique}) == 200
    assert len(weighted) == 300
    assert sum(row["mode"] == "direct" for row in weighted) == 100
    assert sum(row["mode"] == "correction" for row in weighted) == 100
    assert sum(row["mode"] == "structured" for row in weighted) == 50
    assert sum(row["mode"] == "explanation" for row in weighted) == 50
    assert all("不作推断" in row["target_answer"] for row in unique if row["mode"] == "direct")
    assert all(
        row["target_answer"].startswith("结论：不符合。可见依据：")
        for row in unique
        if row["mode"] == "correction"
    )

    exported = grounded.to_llamafactory_records(weighted)
    assert len(exported) == 300
    assert all(row["messages"][0]["content"].count("<image>") == 1 for row in exported)


def test_grounded_builder_writes_a_hashed_v6_training_input_manifest(tmp_path):
    grounded = load_script("build_grounded_train")
    facts = [row for row in make_facts() if row["split"] == "train"]

    manifest = grounded.write_grounded_outputs(facts, tmp_path)

    assert manifest["unique_training_records"] == 200
    assert manifest["weighted_training_samples"] == 300
    assert set(manifest["files"]) == {
        "week5_vlm_train_grounded_internal.json",
        "week5_vlm_train_grounded.json",
        "week5_vlm_train_grounded_weighted.json",
    }
    for name, metadata in manifest["files"].items():
        path = tmp_path / name
        assert metadata["bytes"] == path.stat().st_size
        assert metadata["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_validator_rejects_cross_split_hash_overlap():
    validate = load_script("validate_vlm_dataset")
    facts = make_facts()
    final = next(row for row in facts if row["split"] == "final")
    train = next(row for row in facts if row["split"] == "train")
    final["sha256"] = train["sha256"]

    with pytest.raises(ValueError, match="cross-split.*sha256"):
        validate.validate_split_isolation(facts)


def test_validator_rejects_duplicate_hash_even_inside_one_split():
    validate = load_script("validate_vlm_dataset")
    facts = make_facts()
    train_rows = [row for row in facts if row["split"] == "train"]
    train_rows[1]["sha256"] = train_rows[0]["sha256"]

    with pytest.raises(ValueError, match="duplicate sha256"):
        validate.validate_split_isolation(facts)


def test_validator_detects_day25_prompt_pollution_after_normalization():
    validate = load_script("validate_vlm_dataset")
    prompts = ["请描述：这张图里的内容！", "概括另一张图片"]
    excluded = ["请描述这张图里的内容"]
    collisions = validate.find_prompt_collisions(prompts, excluded)
    assert collisions == [{"prompt": prompts[0], "excluded": excluded[0]}]


def test_validator_rejects_image_path_escape(tmp_path):
    validate = load_script("validate_vlm_dataset")
    (tmp_path / "images").mkdir()
    assert validate.resolve_inside(tmp_path, "images/example.png") == tmp_path / "images/example.png"
    with pytest.raises(ValueError, match="escapes data root"):
        validate.resolve_inside(tmp_path, "../secret.png")


def test_trainable_parameter_audit_rejects_vision_or_projector_lora():
    audit = load_script("verify_trainable_parameters")
    safe = [
        "model.layers.0.self_attn.q_proj.lora_A.default.weight",
        "model.layers.0.mlp.down_proj.lora_B.default.weight",
    ]
    assert audit.audit_trainable_names(safe)["passed"] is True

    unsafe = safe + ["visual.merger.mlp.0.lora_A.default.weight"]
    result = audit.audit_trainable_names(unsafe)
    assert result["passed"] is False
    assert result["forbidden_names"] == [unsafe[-1]]


def test_base_lora_gate_requires_all_three_thresholds():
    score = load_script("score_base_lora")
    rows = []
    for index in range(20):
        base_scores = {
            "factual": 3,
            "instruction": 3,
            "completeness": 3,
            "usefulness": 3,
            "format": 3,
        }
        lora_scores = {name: (4 if index < 12 else 3) for name in base_scores}
        rows.append(
            {
                "case_id": f"case-{index:02d}",
                "base": {"scores": base_scores, "hallucination": index < 4},
                "lora": {"scores": lora_scores, "hallucination": index < 3},
            }
        )

    result = score.score_comparison(rows)
    assert result["mean_gain"] == pytest.approx(0.6)
    assert result["lora_wins"] == 12
    assert result["lora_hallucination_rate"] == pytest.approx(0.15)
    assert result["passed"] is True

    for row in rows[-3:]:
        row["lora"]["scores"] = {name: 2 for name in base_scores}
    assert score.score_comparison(rows)["passed"] is False


def test_blind_scores_are_revealed_only_with_private_key():
    score = load_script("score_base_lora")
    blind = [
        {
            "case_id": "case-01",
            "response_a": {"scores": {name: 4 for name in score.WEIGHTS}, "hallucination": False},
            "response_b": {"scores": {name: 3 for name in score.WEIGHTS}, "hallucination": True},
        }
    ]
    key = [{"case_id": "case-01", "a_model": "lora", "b_model": "base"}]
    revealed = score.reveal_blind_scores(blind, key)
    assert revealed == [
        {
            "case_id": "case-01",
            "lora": blind[0]["response_a"],
            "base": blind[0]["response_b"],
        }
    ]

    with pytest.raises(ValueError, match="case IDs"):
        score.reveal_blind_scores(blind, [{"case_id": "other", "a_model": "lora", "b_model": "base"}])


def test_scoring_protocol_is_loaded_from_the_frozen_evaluation_config():
    score = load_script("score_base_lora")
    protocol = score.load_scoring_protocol(DAY26 / "configs" / "evaluation_config_v6_final.json")

    assert protocol["weights"] == score.WEIGHTS
    assert protocol["thresholds"] == {
        "mean_gain_min": 0.50,
        "lora_wins_min": 12,
        "hallucination_rate_not_worse": True,
    }
    assert protocol["config_sha256"] == hashlib.sha256(
        (DAY26 / "configs" / "evaluation_config_v6_final.json").read_bytes()
    ).hexdigest()


def test_evaluation_config_keeps_final_test_single_use_and_fixed_weights():
    config_path = DAY26 / "configs" / "evaluation_config.json"
    assert config_path.exists()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["final_test_max_runs"] == 1
    assert config["weights"] == {
        "factual": 0.35,
        "instruction": 0.25,
        "completeness": 0.15,
        "usefulness": 0.15,
        "format": 0.10,
    }
    assert config["pass_thresholds"] == {
        "mean_gain_min": 0.50,
        "lora_wins_min": 12,
        "hallucination_rate_not_worse": True,
    }


def test_remediation_final_config_changes_only_blind_seed_and_protocol_label():
    original = json.loads(
        (DAY26 / "configs" / "evaluation_config.json").read_text(encoding="utf-8")
    )
    remediation = json.loads(
        (DAY26 / "configs" / "evaluation_config_v6_final.json").read_text(encoding="utf-8")
    )
    assert remediation["protocol_version"] == "day26-v1-remediation-v6"
    assert remediation["blind_seed"] == 2606
    assert remediation["blind_seed"] != original["blind_seed"]
    for key in ("seed", "weights", "pass_thresholds", "generation"):
        assert remediation[key] == original[key]


def test_training_configs_pin_lineage_and_only_target_language_lora():
    framework = json.loads((DAY26 / "configs" / "framework_lock.json").read_text(encoding="utf-8"))
    assert framework["repository"] == "hiyouga/LLaMA-Factory"
    assert framework["revision"] == "ca75f1edf3cb50343ed1c98605141c3e22075b5f"
    assert framework["tag"] == "v0.9.3"

    expected_targets = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    for name in (
        "qwen2vl_lora_smoke.yaml",
        "qwen2vl_lora_train.yaml",
        "qwen2vl_lora_candidate_v2.yaml",
    ):
        config = json.loads((DAY26 / "configs" / name).read_text(encoding="utf-8"))
        assert config["model_name_or_path"].endswith("/models/Qwen2-VL-7B-Instruct")
        assert config["model_revision"] == "eed13092ef92e448dd6875b2a00151bd3f7db0ac"
        assert config["finetuning_type"] == "lora"
        assert config["freeze_vision_tower"] is True
        assert config["freeze_multi_modal_projector"] is True
        assert set(config["lora_target"].split(",")) == expected_targets
        assert config["template"] == "qwen2_vl"
        assert config["bf16"] is True
        assert config["packing"] is False

    v3 = json.loads((DAY26 / "configs" / "qwen2vl_lora_candidate_v3_qv.yaml").read_text())
    assert set(v3["lora_target"].split(",")) == {"q_proj", "v_proj"}
    assert v3["lora_rank"] == 8
    assert v3["freeze_vision_tower"] is True
    assert v3["freeze_multi_modal_projector"] is True

    v4 = json.loads((DAY26 / "configs" / "qwen2vl_lora_candidate_v4_r8_e2.yaml").read_text())
    assert set(v4["lora_target"].split(",")) == expected_targets
    assert v4["lora_rank"] == 8
    assert v4["num_train_epochs"] == 2.0
    assert v4["freeze_vision_tower"] is True

    v5 = json.loads((DAY26 / "configs" / "qwen2vl_lora_candidate_v5_r16_e2.yaml").read_text())
    assert set(v5["lora_target"].split(",")) == expected_targets
    assert v5["lora_rank"] == 16
    assert v5["learning_rate"] == 0.00002
    assert v5["num_train_epochs"] == 2.0
    assert v5["freeze_vision_tower"] is True
    assert v5["freeze_multi_modal_projector"] is True

    v6 = json.loads((DAY26 / "configs" / "qwen2vl_lora_candidate_v6_grounded.yaml").read_text())
    assert set(v6["lora_target"].split(",")) == expected_targets
    assert v6["dataset"] == "day26_train_grounded_weighted"
    assert v6["lora_rank"] == 16
    assert v6["learning_rate"] == 0.00002
    assert v6["num_train_epochs"] == 1.5
    assert v6["freeze_vision_tower"] is True
    assert v6["freeze_multi_modal_projector"] is True


def test_dataset_info_registers_openai_multimodal_sharegpt():
    info = json.loads((DAY26 / "configs" / "dataset_info.json").read_text(encoding="utf-8"))
    for split in ("train", "dev", "final"):
        row = info[f"day26_{split}"]
        assert row["file_name"] == f"week5_vlm_{split}.json"
        assert row["formatting"] == "sharegpt"
        assert row["columns"] == {"messages": "messages", "images": "images"}
        assert row["tags"] == {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        }

    grounded = info["day26_train_grounded_weighted"]
    assert grounded["file_name"] == "week5_vlm_train_grounded_weighted.json"
    assert grounded["formatting"] == "sharegpt"
    assert grounded["columns"] == {"messages": "messages", "images": "images"}


def test_adapter_audit_rejects_visual_keys_and_accepts_language_lora():
    verify = load_script("verify_adapter")
    safe = [
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
        "base_model.model.model.layers.0.mlp.down_proj.lora_B.weight",
    ]
    result = verify.audit_adapter_keys(safe)
    assert result["passed"] is True
    assert result["forbidden_keys"] == []

    unsafe = safe + ["base_model.model.visual.merger.mlp.0.lora_A.weight"]
    result = verify.audit_adapter_keys(unsafe)
    assert result["passed"] is False
    assert result["forbidden_keys"] == [unsafe[-1]]


def test_eval_runner_validates_frozen_twenty_case_contract(tmp_path):
    runner = load_script("run_base_lora_eval")
    rows = []
    for category in runner.CATEGORIES:
        for image_index in range(2):
            for mode in ("direct", "correction"):
                rows.append(
                    {
                        "id": f"{category}-{image_index}-{mode}",
                        "split": "final",
                        "category": category,
                        "mode": mode,
                        "source_image_id": f"{category}-{image_index}",
                        "image_relative_path": f"images/final/{category}-{image_index}.png",
                        "image_sha256": f"{len(rows) + 1:064x}",
                        "user_instruction": "请回答图像问题。",
                        "reference_answer": "参考答案。",
                    }
                )

    runner.validate_eval_records(rows, "final")
    rows[-1]["mode"] = "direct"
    with pytest.raises(ValueError, match="two modes"):
        runner.validate_eval_records(rows, "final")

    ledger = tmp_path / "final_run.json"
    spec = "a" * 64
    runner.reserve_final_run(ledger, "run-001", run_spec_sha256=spec)
    runner.reserve_final_run(ledger, "run-001", run_spec_sha256=spec, resume=True)
    with pytest.raises(ValueError, match="run spec"):
        runner.reserve_final_run(
            ledger, "run-001", run_spec_sha256="b" * 64, resume=True
        )
    with pytest.raises(ValueError, match="already reserved"):
        runner.reserve_final_run(ledger, "run-002", run_spec_sha256=spec)

    runner.validate_resume_rows(
        [{"run_id": "run-001", "run_spec_sha256": spec}], "run-001", spec
    )
    with pytest.raises(ValueError, match="raw resume rows"):
        runner.validate_resume_rows(
            [{"run_id": "run-001", "run_spec_sha256": "b" * 64}],
            "run-001",
            spec,
        )


def test_eval_runner_can_apply_auditable_lora_scale():
    runner = load_script("run_base_lora_eval")

    class Layer:
        def __init__(self):
            self.scaling = {"default": 2.0}

    class Model:
        def __init__(self):
            self.layers = [Layer(), Layer()]

        def modules(self):
            return iter(self.layers)

    model = Model()
    assert runner.apply_lora_scale(model, 0.5) == 2
    assert [layer.scaling["default"] for layer in model.layers] == [1.0, 1.0]
    assert runner.apply_lora_scale(model, 1.25) == 2
    assert [layer.scaling["default"] for layer in model.layers] == [1.25, 1.25]
    with pytest.raises(ValueError, match="scale"):
        runner.apply_lora_scale(model, 0.0)
    with pytest.raises(ValueError, match="scale"):
        runner.apply_lora_scale(model, 2.01)


def test_eval_runner_binds_run_to_exact_adapter_weight_sha(tmp_path):
    runner = load_script("run_base_lora_eval")
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    weight_path = adapter_dir / "adapter_model.safetensors"
    weight_path.write_bytes(b"selected-adapter")
    expected = hashlib.sha256(weight_path.read_bytes()).hexdigest()

    assert runner.verify_adapter_identity(adapter_dir, expected) == expected
    with pytest.raises(ValueError, match="adapter weight SHA-256 mismatch"):
        runner.verify_adapter_identity(adapter_dir, "0" * 64)


def test_eval_runner_verifies_the_complete_day22_base_model_manifest(tmp_path):
    runner = load_script("run_base_lora_eval")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_bytes(b'{"model_type":"qwen2_vl"}\n')
    (model_dir / "weights.safetensors").write_bytes(b"base-model-weights")
    files = [
        {
            "relative_path": path.name,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(model_dir.iterdir())
    ]
    files_digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest = {
        "status": "downloaded_verified",
        "repository": "Qwen/Qwen2-VL-7B-Instruct",
        "revision": "e" * 40,
        "file_count": len(files),
        "total_bytes": sum(row["bytes"] for row in files),
        "files_digest_sha256": files_digest,
        "files": files,
    }
    manifest_path = tmp_path / "model_manifest.json"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    cache_dir = model_dir / ".cache/huggingface/download"
    cache_dir.mkdir(parents=True)
    (model_dir / ".cache/huggingface/.gitignore").write_text("*\n", encoding="utf-8")
    (cache_dir / "config.json.metadata").write_text("cache metadata\n", encoding="utf-8")

    identity = runner.verify_base_model_identity(model_dir, manifest_path)

    assert identity == {
        "base_model_repository": "Qwen/Qwen2-VL-7B-Instruct",
        "base_model_revision": "e" * 40,
        "base_model_files_digest_sha256": files_digest,
        "base_model_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
    }
    (model_dir / "undeclared.txt").write_text("must fail closed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not exactly match"):
        runner.verify_base_model_identity(model_dir, manifest_path)
    (model_dir / "undeclared.txt").unlink()
    (model_dir / "weights.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="base model file integrity failed"):
        runner.verify_base_model_identity(model_dir, manifest_path)


def test_eval_run_spec_binds_base_adapter_config_and_scale():
    runner = load_script("run_base_lora_eval")
    payload = {
        "run_id": "run-001",
        "split": "final",
        "config_sha256": "1" * 64,
        "cases_sha256": "2" * 64,
        "base_model_repository": "Qwen/Qwen2-VL-7B-Instruct",
        "base_model_revision": "3" * 40,
        "base_model_files_digest_sha256": "4" * 64,
        "base_model_manifest_sha256": "5" * 64,
        "adapter_config_sha256": "6" * 64,
        "adapter_model_sha256": "7" * 64,
        "lora_scale": 0.85,
    }
    original = runner.build_run_spec_sha256(payload)

    for field, replacement in (
        ("base_model_repository", "other/model"),
        ("base_model_revision", "8" * 40),
        ("base_model_files_digest_sha256", "8" * 64),
        ("base_model_manifest_sha256", "9" * 64),
        ("adapter_config_sha256", "a" * 64),
        ("lora_scale", 1.0),
    ):
        changed = dict(payload)
        changed[field] = replacement
        assert runner.build_run_spec_sha256(changed) != original


def test_blind_packet_is_deterministic_and_hides_model_identity():
    runner = load_script("run_base_lora_eval")
    rows = [
        {
            "case_id": f"case-{index:02d}",
            "image_relative_path": f"images/final/case-{index:02d}.png",
            "user_instruction": "请回答。",
            "reference_answer": "参考。",
            "base_response": "base",
            "lora_response": "lora",
        }
        for index in range(20)
    ]
    packet_a, key_a = runner.build_blind_packet(rows, seed=2601)
    packet_b, key_b = runner.build_blind_packet(rows, seed=2601)
    assert packet_a == packet_b
    assert key_a == key_b
    assert all(
        set(row)
        == {
            "case_id",
            "image_relative_path",
            "user_instruction",
            "reference_answer",
            "response_a",
            "response_b",
        }
        for row in packet_a
    )
    assert all(set(row) == {"case_id", "a_model", "b_model"} for row in key_a)
    assert {row["a_model"] for row in key_a} == {"base", "lora"}


def test_final_validator_requires_every_teacher_and_quality_gate():
    validate = load_script("validate_day26")
    evidence = {
        "data_status": "PASS",
        "training_record_count": 200,
        "training_sampling_record_count": 300,
        "dev_record_count": 20,
        "final_record_count": 20,
        "trainable_status": "PASS",
        "smoke_exit": 0,
        "formal_exit": 0,
        "adapter_status": "PASS",
        "paired_response_count": 20,
        "blind_score_count": 20,
        "mean_gain_check": True,
        "lora_wins_check": True,
        "hallucination_check": True,
        "training_input_files_match_manifest": True,
        "training_dataset_lineage_match": True,
        "training_config_lineage_match": True,
        "adapter_lineage_match": True,
        "evaluation_lineage_match": True,
        "scoring_config_lineage_match": True,
    }
    result = validate.evaluate_evidence(evidence)
    assert result["status"] == "PASS"
    assert all(result["checks"].values())

    evidence["mean_gain_check"] = False
    result = validate.evaluate_evidence(evidence)
    assert result["status"] == "FAIL"
    assert result["checks"]["quality_mean_gain"] is False


def test_final_validator_collects_selected_v6_artifacts_from_real_tree():
    validate = load_script("validate_day26")
    evidence = validate.collect_evidence(DAY26)

    assert evidence["data_status"] == "PASS"
    assert evidence["training_record_count"] == 200
    assert evidence["training_sampling_record_count"] == 300
    assert evidence["training_input_files_match_manifest"] is True
    assert evidence["training_dataset_lineage_match"] is True
    assert evidence["training_config_lineage_match"] is True
    assert evidence["adapter_lineage_match"] is True
    assert evidence["evaluation_lineage_match"] is True
    assert all(evidence["evaluation_lineage_checks"].values())
    assert evidence["scoring_config_lineage_match"] is True
    assert evidence["dev_record_count"] == 20
    assert evidence["final_record_count"] == 20
    assert evidence["trainable_status"] == "PASS"
    assert evidence["smoke_exit"] == 0
    assert evidence["formal_exit"] == 0
    assert evidence["adapter_status"] == "PASS"
    assert evidence["paired_response_count"] == 20
    assert evidence["blind_score_count"] == 20
    assert evidence["evaluation_scope"] == "image_disjoint_shared_task_templates"
    assert evidence["prompt_template_overlap_records"] == {"dev": 20, "final": 20}
    assert evidence["target_answer_exact_overlap_records"] == {"dev": 0, "final": 0}
    assert evidence["mean_gain_check"] is False
    assert evidence["lora_wins_check"] is True
    assert evidence["hallucination_check"] is True


def test_evaluation_lineage_rejects_each_tampered_identity_field():
    validate = load_script("validate_day26")
    expected = {
        "generation_config_sha256": "1" * 64,
        "cases_sha256": "2" * 64,
        "adapter_config_sha256": "3" * 64,
        "adapter_model_sha256": "4" * 64,
        "lora_scale": 0.85,
        "base_model_repository": "Qwen/Qwen2-VL-7B-Instruct",
        "base_model_revision": "5" * 40,
        "base_model_files_digest_sha256": "6" * 64,
        "base_model_manifest_sha256": "7" * 64,
    }
    summary = dict(expected)
    assert all(validate.evaluation_lineage_checks(summary, expected).values())

    for field, replacement in (
        ("generation_config_sha256", "8" * 64),
        ("cases_sha256", "8" * 64),
        ("adapter_config_sha256", "8" * 64),
        ("adapter_model_sha256", "8" * 64),
        ("lora_scale", 1.0),
        ("base_model_repository", "other/model"),
        ("base_model_revision", "8" * 40),
        ("base_model_files_digest_sha256", "8" * 64),
        ("base_model_manifest_sha256", "8" * 64),
    ):
        tampered = dict(summary)
        tampered[field] = replacement
        checks = validate.evaluation_lineage_checks(tampered, expected)
        assert checks[field] is False
        assert not all(checks.values())


def test_final_validator_rejects_missing_or_tampered_manifest_files(tmp_path):
    validate = load_script("validate_day26")
    path = tmp_path / "weighted.json"
    path.write_bytes(b"selected-training-input")
    manifest = {
        "files": {
            "weighted.json": {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        }
    }

    assert validate.files_match_manifest(tmp_path, manifest, expected_count=1) is True
    path.write_bytes(b"tampered")
    assert validate.files_match_manifest(tmp_path, manifest, expected_count=1) is False
    path.unlink()
    assert validate.files_match_manifest(tmp_path, manifest, expected_count=1) is False


def test_real_split_audit_discloses_shared_task_prompts_but_unique_answers():
    validate = load_script("validate_vlm_dataset")
    data = DAY26 / "source" / "data"
    rows = {
        split: json.loads(
            (data / f"week5_vlm_{split}_internal.json").read_text(encoding="utf-8")
        )
        for split in ("train", "dev", "final")
    }

    audit = validate.cross_split_text_audit(rows)

    assert audit["prompt_template_overlap_records"] == {"dev": 20, "final": 20}
    assert audit["target_answer_exact_overlap_records"] == {"dev": 0, "final": 0}
    assert audit["evaluation_scope"] == "image_disjoint_shared_task_templates"


def test_teacher_projection_has_recomputable_redacted_hash_and_no_blind_artifacts():
    submission = DAY26.parents[2] / "Submission" / "Week5" / "Day26_VLM_LoRA"
    public_config = submission / "Config" / "Evaluation_Config.json"
    public_sha = hashlib.sha256(public_config.read_bytes()).hexdigest()
    evaluation = json.loads(
        (submission / "Results" / "Evaluation_Run_Summary.json").read_text(
            encoding="utf-8"
        )
    )
    comparison = json.loads(
        (submission / "Results" / "Comparison_Summary.json").read_text(
            encoding="utf-8"
        )
    )
    base_manifest = submission / "Model_Archive" / "Base_Model_Manifest.json"
    base_manifest_sha = hashlib.sha256(base_manifest.read_bytes()).hexdigest()

    assert public_sha == "26ce16e5e6e54879b0ba554c3a83dbdc718d269860834c15947069cde0d635a5"
    assert evaluation["submitted_redacted_evaluation_config_sha256"] == public_sha
    assert comparison["submitted_redacted_evaluation_config_sha256"] == public_sha
    assert evaluation["base_model_manifest_sha256"] == base_manifest_sha
    assert evaluation["base_model_files_digest_sha256"] == json.loads(
        base_manifest.read_text(encoding="utf-8")
    )["files_digest_sha256"]
    assert not (submission / "Results" / "Blind_Packet.json").exists()
    assert not (submission / "Results" / "Blind_Scores.json").exists()
    assert "blind_seed" not in json.loads(public_config.read_text(encoding="utf-8"))


def test_remote_evidence_collector_excludes_private_keys_and_hashes_binary(tmp_path):
    collect = load_script("collect_remote_evidence")
    results = tmp_path / "results"
    adapter = tmp_path / "adapter"
    results.mkdir()
    adapter.mkdir()
    (results / "formal_v6_training.log").write_text("training complete\n", encoding="utf-8")
    (results / "private_blind_key_final.json").write_text("secret\n", encoding="utf-8")
    (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter-bytes")

    payload = collect.collect_evidence(tmp_path, adapter)

    assert "results/formal_v6_training.log" in payload["text_artifacts"]
    assert not any("private" in name for name in payload["text_artifacts"])
    binary = payload["selected_adapter_files"]["adapter_model.safetensors"]
    assert binary["bytes"] == len(b"adapter-bytes")
    assert len(binary["sha256"]) == 64
