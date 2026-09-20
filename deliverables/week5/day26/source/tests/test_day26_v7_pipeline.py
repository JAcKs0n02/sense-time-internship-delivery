import importlib.util
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess

import pytest


DAY26 = Path(__file__).resolve().parents[2]
SCRIPTS = DAY26 / "source" / "scripts"


def load_script(name: str):
    path = SCRIPTS / f"{name}.py"
    assert path.exists(), f"missing Day26 v7 script: {path}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_fact(
    *,
    image_id: str = "ocr-train-00",
    split: str = "train",
    category: str = "ocr",
    index: int = 1,
    claim_supported: bool = True,
) -> dict:
    expected_modes = (
        ("direct", "evidence", "verification", "grounding")
        if split == "train"
        else ("direct", "verification")
    )
    subject = f"{category} subject {image_id}"
    prompts = {
        mode: f"[{image_id}] 请完成 {mode} 视觉任务并仅依据可见内容。"
        for mode in expected_modes
    }
    answers = {
        "direct": f"图中可确认的主体是 {subject}。",
        "verification": {
            "verdict": "supported" if claim_supported else "unsupported",
            "evidence": f"可见证据 {image_id} 支持该判断。",
        },
    }
    if split == "train":
        answers.update(
            {
                "evidence": f"证据包括 {image_id} 的位置、文字与结构。",
                "grounding": f"能够确认 {subject}；未显示的属性不能确认。",
            }
        )
    transcription_required = category in {"ocr", "formula"}
    return {
        "image_id": image_id,
        "split": split,
        "category": category,
        "image_relative_path": f"images_v7/{split}/{image_id}.png",
        "source_page_url": f"https://github.com/example/repo/blob/{index:040x}/{image_id}.png",
        "download_url": f"https://raw.githubusercontent.com/example/repo/{index:040x}/{image_id}.png",
        "source_revision": f"{index:040x}",
        "source_revision_kind": "git_commit",
        "author": "Example Project Authors",
        "license": "Apache-2.0",
        "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
        "sha256": f"{index:064x}",
        "subject": subject,
        "observable_facts": [
            f"可见事实 A {image_id}",
            f"可见事实 B {image_id}",
            f"可见事实 C {image_id}",
        ],
        "gold_transcription": {
            "applicable": transcription_required,
            "value": f"verified text {image_id}" if transcription_required else "",
            "review_status": "double_checked" if transcription_required else "not_applicable",
            "ambiguous_tokens": [],
        },
        "uncertainty_notes": ["图片未展示的信息不作推断。"],
        "verification_claim": f"图片展示了 {subject}。",
        "claim_supported": claim_supported,
        "prompts": prompts,
        "answers": answers,
        "annotation_status": "double_checked",
    }


def make_balanced_facts() -> list[dict]:
    build = load_script("build_v7_dataset")
    facts = []
    ordinal = 1
    split_sizes = {"train": 10, "dev": 2, "final": 2}
    for category in build.CATEGORIES:
        for split, size in split_sizes.items():
            for index in range(size):
                image_id = f"{category}-{split}-{index:02d}"
                facts.append(
                    make_fact(
                        image_id=image_id,
                        split=split,
                        category=category,
                        index=ordinal,
                        claim_supported=index % 2 == 0,
                    )
                )
                ordinal += 1
    return facts


def test_v7_fact_requires_reviewed_traceable_grounded_fields():
    build = load_script("build_v7_dataset")
    fact = make_fact()
    build.validate_v7_fact(fact)

    required = (
        "source_page_url",
        "download_url",
        "source_revision",
        "source_revision_kind",
        "author",
        "license",
        "license_url",
        "sha256",
        "subject",
        "observable_facts",
        "verification_claim",
        "claim_supported",
        "prompts",
        "answers",
        "annotation_status",
    )
    for field in required:
        bad = dict(fact)
        bad[field] = "" if field != "claim_supported" else None
        with pytest.raises(ValueError, match=field):
            build.validate_v7_fact(bad)


def test_v7_fact_rejects_weak_or_ambiguous_visual_gold():
    build = load_script("build_v7_dataset")

    bad = make_fact()
    bad["observable_facts"] = ["only one"]
    with pytest.raises(ValueError, match="observable_facts"):
        build.validate_v7_fact(bad)

    for category in ("ocr", "formula"):
        bad = make_fact(category=category, image_id=f"{category}-train-00")
        bad["gold_transcription"] = {
            "applicable": True,
            "value": "uncertain expression",
            "review_status": "double_checked",
            "ambiguous_tokens": ["?"],
        }
        with pytest.raises(ValueError, match="ambiguous_tokens"):
            build.validate_v7_fact(bad)

        bad = make_fact(category=category, image_id=f"{category}-train-00")
        bad["gold_transcription"]["review_status"] = "single_pass"
        with pytest.raises(ValueError, match="review_status"):
            build.validate_v7_fact(bad)


def test_v7_source_revision_is_honest_for_git_or_dated_web_snapshots():
    build = load_script("build_v7_dataset")
    git_fact = make_fact()
    build.validate_v7_fact(git_fact)

    dated = make_fact()
    dated["source_revision_kind"] = "retrieval_date"
    dated["source_revision"] = "2026-08-21"
    build.validate_v7_fact(dated)

    bad = make_fact()
    bad["source_revision_kind"] = "retrieval_date"
    with pytest.raises(ValueError, match="source_revision"):
        build.validate_v7_fact(bad)


def test_v7_verification_label_must_match_claim_boolean():
    build = load_script("build_v7_dataset")
    bad = make_fact(claim_supported=False)
    bad["answers"]["verification"]["verdict"] = "supported"
    with pytest.raises(ValueError, match="verdict"):
        build.validate_v7_fact(bad)


def test_v7_bundle_requires_50_10_10_images_and_category_balance():
    build = load_script("build_v7_dataset")
    facts = make_balanced_facts()
    build.validate_v7_fact_collection(facts)

    bad = facts[:-1]
    with pytest.raises(ValueError, match="image split counts"):
        build.validate_v7_fact_collection(bad)

    bad = [dict(row) for row in facts]
    bad[0]["category"] = "ui"
    with pytest.raises(ValueError, match="category balance"):
        build.validate_v7_fact_collection(bad)


def test_v7_verification_claims_are_balanced_per_category_and_split():
    build = load_script("build_v7_dataset")
    facts = make_balanced_facts()
    build.validate_v7_fact_collection(facts)

    bad = [dict(row) for row in facts]
    first_category = build.CATEGORIES[0]
    for row in bad:
        if row["split"] == "train" and row["category"] == first_category:
            row["claim_supported"] = True
            row["answers"] = dict(row["answers"])
            row["answers"]["verification"] = {
                "verdict": "supported",
                "evidence": "visible evidence",
            }
    with pytest.raises(ValueError, match="verification balance"):
        build.validate_v7_fact_collection(bad)


def test_v7_builder_produces_unique_200_20_20_mode_contract():
    build = load_script("build_v7_dataset")
    records = build.build_v7_records(make_balanced_facts())

    assert {split: len(rows) for split, rows in records.items()} == {
        "train": 200,
        "dev": 20,
        "final": 20,
    }
    expected_modes = {
        "train": set(build.TRAIN_MODES),
        "dev": set(build.EVAL_MODES),
        "final": set(build.EVAL_MODES),
    }
    all_ids = []
    for split, rows in records.items():
        by_image = defaultdict(set)
        for row in rows:
            all_ids.append(row["id"])
            by_image[row["source_image_id"]].add(row["mode"])
        assert all(modes == expected_modes[split] for modes in by_image.values())
    assert len(all_ids) == len(set(all_ids)) == 240
    assert Counter(row["category"] for row in records["train"]) == Counter(
        {category: 40 for category in build.CATEGORIES}
    )


def test_v7_export_keeps_one_image_token_and_auditable_metadata():
    build = load_script("build_v7_dataset")
    internal = build.build_v7_records(make_balanced_facts())["train"][:4]
    exported = build.to_llamafactory_records(internal)

    for source, target in zip(internal, exported):
        assert target["messages"] == [
            {"role": "user", "content": source["prompt"]},
            {"role": "assistant", "content": source["target_answer"]},
        ]
        assert target["images"] == [source["image_relative_path"]]
        assert source["construction_method"] == "human-double-checked-v7"
        assert source["claim_supported"] in {True, False, None}


def test_v7_acquisition_has_50_reaudited_train_and_20_new_heldout_sources():
    acquire = load_script("acquire_source_images_v7")
    legacy = json.loads((DAY26 / "source/data/source_image_manifest.json").read_text())
    specs = acquire.build_source_specs(legacy)

    assert len(specs) == 70
    assert Counter(row["split"] for row in specs) == Counter(
        {"train": 50, "dev": 10, "final": 10}
    )
    assert Counter(row["category"] for row in specs if row["split"] == "train") == Counter(
        {category: 10 for category in acquire.CATEGORIES}
    )
    assert Counter(row["category"] for row in specs if row["split"] == "dev") == Counter(
        {category: 2 for category in acquire.CATEGORIES}
    )
    assert Counter(row["category"] for row in specs if row["split"] == "final") == Counter(
        {category: 2 for category in acquire.CATEGORIES}
    )
    assert sum(row["provenance_role"] == "reaudited_v6_train_bytes" for row in specs) == 50
    assert sum(row["provenance_role"] == "new_v7_heldout_bytes" for row in specs) == 20
    for row in specs:
        assert row["source_revision_kind"] in {"git_commit", "retrieval_date"}
        if row["source_revision_kind"] == "git_commit":
            assert len(row["source_revision"]) == 40
        else:
            assert row["source_revision"] == "2026-08-21"
        assert row["source_page_url"].startswith("https://")
        assert row["download_url"].startswith("https://")
        assert row["license_url"].startswith("https://")
        assert row["author"].strip()
        assert row["license"].strip()


def test_v7_heldout_inventory_excludes_visual_qa_rejects():
    acquire = load_script("acquire_source_images_v7")
    legacy = json.loads((DAY26 / "source/data/source_image_manifest.json").read_text())
    specs = acquire.build_source_specs(legacy)
    heldout = [row for row in specs if row["split"] in {"dev", "final"}]

    rejected_paths = {
        "skimage/data/cell.png",
        "skimage/data/clock_motion.png",
        "skimage/data/ihc.png",
        "skimage/data/microaneurysms.png",
        "docs/examples/page.png",
    }
    assert rejected_paths.isdisjoint({row["source_path"] for row in heldout})
    reviewed_ids = {
        "natural_scene-v7-dev-03",
        "natural_scene-v7-dev-04",
        "natural_scene-v7-final-03",
        "natural_scene-v7-final-04",
        "formula-v7-final-03",
    }
    assert reviewed_ids <= {row["image_id"] for row in heldout}


def test_v7_curated_facts_match_manifest_and_fix_known_formula_labels():
    curate = load_script("curate_v7_facts")
    build = load_script("build_v7_dataset")
    manifest = json.loads(
        (DAY26 / "source/data/source_image_manifest_v7.json").read_text()
    )
    legacy = json.loads((DAY26 / "source/data/source_image_facts.json").read_text())

    facts = curate.build_curated_facts(manifest, legacy)
    build.validate_v7_fact_collection(facts)
    assert len(facts) == 70
    assert {
        (row["image_id"], row["sha256"], row["image_relative_path"])
        for row in facts
    } == {
        (row["image_id"], row["sha256"], row["image_relative_path"])
        for row in manifest
    }

    by_id = {row["image_id"]: row for row in facts}
    assert "=-\\sum" not in by_id["formula-01"]["gold_transcription"]["value"]
    assert "\\frac{A}{x+1}" in by_id["formula-02"]["gold_transcription"]["value"]
    assert "\\mathbb{E}" in by_id["formula-04"]["gold_transcription"]["value"]
    assert by_id["formula-v7-final-03"]["source_page_url"].startswith(
        "https://commons.wikimedia.org/"
    )

    records = build.build_v7_records(facts)
    prompt_sets = {
        split: {row["user_instruction"] for row in rows}
        for split, rows in records.items()
    }
    assert prompt_sets["train"].isdisjoint(prompt_sets["dev"])
    assert prompt_sets["train"].isdisjoint(prompt_sets["final"])
    assert prompt_sets["dev"].isdisjoint(prompt_sets["final"])


def test_v7_real_artifact_validator_recomputes_every_declared_input():
    validate = load_script("validate_v7_dataset")
    result = validate.validate_artifacts(DAY26 / "source")

    assert result["status"] == "PASS"
    assert result["image_counts"] == {"train": 50, "dev": 10, "final": 10}
    assert result["record_counts"] == {"train": 200, "dev": 20, "final": 20}
    assert result["prompt_collisions"] == {
        "train_dev": [],
        "train_final": [],
        "dev_final": [],
    }
    assert all(result["checks"].values())


def test_v7_declared_file_gate_detects_removal_and_tamper(tmp_path):
    validate = load_script("validate_v7_dataset")
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text('{"records": [1]}\n', encoding="utf-8")
    second.write_text('{"records": [2]}\n', encoding="utf-8")
    manifest = validate.build_file_manifest(tmp_path, ["first.json", "second.json"])

    assert validate.declared_files_match(tmp_path, manifest)
    second.write_text('{"records": [3]}\n', encoding="utf-8")
    assert not validate.declared_files_match(tmp_path, manifest)
    second.unlink()
    assert not validate.declared_files_match(tmp_path, manifest)


def test_v7_candidate_configs_only_vary_preregistered_dimensions():
    names = (
        "qwen2vl_lora_v7_a_lr1e5_e1.yaml",
        "qwen2vl_lora_v7_b_lr2e5_e1.yaml",
        "qwen2vl_lora_v7_c_lr1e5_e15.yaml",
    )
    configs = [json.loads((DAY26 / "configs" / name).read_text()) for name in names]
    assert [(row["learning_rate"], row["num_train_epochs"]) for row in configs] == [
        (0.00001, 1.0),
        (0.00002, 1.0),
        (0.00001, 1.5),
    ]
    for row in configs:
        assert row["model_revision"] == "eed13092ef92e448dd6875b2a00151bd3f7db0ac"
        assert row["dataset"] == "day26_train_v7"
        assert row["eval_dataset"] == "day26_dev_v7"
        assert row["freeze_vision_tower"] is True
        assert row["freeze_multi_modal_projector"] is True
        assert row["freeze_language_model"] is False
        assert row["lora_rank"] == 16
        assert row["lora_alpha"] == 32
        assert row["lora_target"] == "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
        assert row["bf16"] is True
        assert row["seed"] == row["data_seed"] == 42
        assert row["image_max_pixels"] == 262144

    ignored = {"learning_rate", "num_train_epochs", "output_dir", "run_name"}
    normalized = [{key: value for key, value in row.items() if key not in ignored} for row in configs]
    assert normalized[0] == normalized[1] == normalized[2]

    dataset_info = json.loads((DAY26 / "configs/dataset_info.json").read_text())
    assert dataset_info["day26_train_v7"]["file_name"] == "week5_vlm_train_v7.json"
    assert dataset_info["day26_dev_v7"]["file_name"] == "week5_vlm_dev_v7.json"


def test_v7_run_manifest_binds_inputs_and_actual_adapter_weights(tmp_path):
    prepare = load_script("prepare_v7_run")
    config_path = DAY26 / "configs/qwen2vl_lora_v7_a_lr1e5_e1.yaml"
    pre = prepare.build_pre_run_manifest(DAY26, config_path, "v7-a")

    assert pre["stage"] == "pre_run"
    assert pre["candidate_id"] == "v7-a"
    assert pre["base_model_revision"] == "eed13092ef92e448dd6875b2a00151bd3f7db0ac"
    assert {
        "training_dataset",
        "fact_cards",
        "source_manifest",
        "data_manifest",
        "training_config",
        "dataset_info",
        "framework_lock",
    } <= set(pre["input_files"])
    for record in pre["input_files"].values():
        assert record["bytes"] > 0
        assert len(record["sha256"]) == 64

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"r": 16}\n', encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"exact-v7-adapter")
    post = prepare.build_post_run_manifest(pre, adapter)
    assert post["stage"] == "post_run"
    assert post["adapter_files"]["adapter_model.safetensors"]["sha256"] == hashlib.sha256(
        b"exact-v7-adapter"
    ).hexdigest()


def test_v7_eval_fingerprint_and_resume_bind_every_immutable_input():
    runner = load_script("run_base_lora_eval_v7")
    payload = {
        "run_id": "v7-dev-a-scale085",
        "stage": "dev",
        "generation_config_sha256": "1" * 64,
        "scoring_config_sha256": "2" * 64,
        "cases_sha256": "3" * 64,
        "base_model_revision": "4" * 40,
        "base_model_files_digest_sha256": "5" * 64,
        "base_model_manifest_sha256": "6" * 64,
        "adapter_config_sha256": "7" * 64,
        "adapter_model_sha256": "8" * 64,
        "lora_scale": 0.85,
    }
    fingerprint = runner.build_run_fingerprint(payload)
    assert len(fingerprint) == 64
    for key, changed_value in (
        ("generation_config_sha256", "a" * 64),
        ("scoring_config_sha256", "b" * 64),
        ("cases_sha256", "c" * 64),
        ("base_model_revision", "d" * 40),
        ("base_model_files_digest_sha256", "e" * 64),
        ("base_model_manifest_sha256", "f" * 64),
        ("adapter_config_sha256", "0" * 64),
        ("adapter_model_sha256", "9" * 64),
        ("lora_scale", 1.0),
        ("stage", "final"),
    ):
        changed = dict(payload)
        changed[key] = changed_value
        assert runner.build_run_fingerprint(changed) != fingerprint

    runner.validate_resume_rows(
        [{"run_id": payload["run_id"], "run_spec_sha256": fingerprint}],
        payload["run_id"],
        fingerprint,
    )
    with pytest.raises(ValueError, match="immutable run spec"):
        runner.validate_resume_rows(
            [{"run_id": payload["run_id"], "run_spec_sha256": "0" * 64}],
            payload["run_id"],
            fingerprint,
        )


def test_v7_eval_runner_accepts_only_direct_and_verification_cases():
    runner = load_script("run_base_lora_eval_v7")
    dev = json.loads((DAY26 / "source/data/week5_vlm_dev_v7_internal.json").read_text())
    runner.validate_eval_records(dev, "dev")

    bad = [dict(row) for row in dev]
    bad[0]["mode"] = "correction"
    with pytest.raises(ValueError, match="direct and verification"):
        runner.validate_eval_records(bad, "dev")


def test_v7_candidate_selection_uses_dev_only_and_frozen_gate():
    selector = load_script("select_v7_candidate")
    cases_sha = "a" * 64
    summaries = [
        {
            "candidate_id": "v7-a",
            "split": "dev",
            "cases_sha256": cases_sha,
            "lora_scale": 0.85,
            "mean_gain": 0.36,
            "direct_gain": 0.10,
            "lora_wins": 12,
            "base_hallucination_rate": 0.20,
            "lora_hallucination_rate": 0.15,
        },
        {
            "candidate_id": "v7-b",
            "split": "dev",
            "cases_sha256": cases_sha,
            "lora_scale": 1.0,
            "mean_gain": 0.50,
            "direct_gain": -0.05,
            "lora_wins": 14,
            "base_hallucination_rate": 0.20,
            "lora_hallucination_rate": 0.10,
        },
    ]
    result = selector.select_candidate(summaries, expected_dev_cases_sha256=cases_sha)
    assert result["status"] == "PASS"
    assert result["selected_candidate_id"] == "v7-a"
    assert result["selected_lora_scale"] == 0.85
    assert result["candidates"][1]["checks"]["direct_gain"] is False

    final = dict(summaries[0], split="final")
    with pytest.raises(ValueError, match="development"):
        selector.select_candidate([final], expected_dev_cases_sha256=cases_sha)


def test_v7_scoring_reports_direct_gain_categories_and_stage_gate():
    score = load_script("score_base_lora_v7")
    rows = []
    categories = ("natural_scene", "ocr", "chart_table", "ui", "formula")
    for category in categories:
        for mode in ("direct", "verification"):
            for index in range(2):
                base_value = 3
                lora_value = 4 if index == 0 else 3
                rows.append(
                    {
                        "case_id": f"{category}-{mode}-{index}",
                        "category": category,
                        "mode": mode,
                        "base": {
                            "scores": {name: base_value for name in score.WEIGHTS},
                            "hallucination": index == 0,
                        },
                        "lora": {
                            "scores": {name: lora_value for name in score.WEIGHTS},
                            "hallucination": False,
                        },
                    }
                )
    result = score.score_comparison(
        rows,
        {
            "mean_gain_min": 0.35,
            "direct_gain_min": 0.0,
            "lora_wins_min": 10,
            "hallucination_rate_not_worse": True,
        },
        split="dev",
    )
    assert result["passed"] is True
    assert result["mean_gain"] == pytest.approx(0.5)
    assert result["direct_gain"] == pytest.approx(0.5)
    assert result["lora_wins"] == 10
    assert set(result["category_metrics"]) == set(categories)


def test_v8_scoring_uses_only_non_ceiling_base_rows_as_win_opportunities():
    """Catches treating Base-at-5 rows as strict-win opportunities."""

    score = load_script("score_base_lora_v7")
    rows = []
    for category in ("natural_scene", "ocr", "chart_table", "ui", "formula"):
        for mode in ("direct", "verification"):
            rows.extend(
                [
                    {
                        "case_id": f"{category}-{mode}-ceiling",
                        "category": category,
                        "mode": mode,
                        "base": {
                            "scores": {name: 5 for name in score.WEIGHTS},
                            "hallucination": False,
                        },
                        "lora": {
                            "scores": {name: 5 for name in score.WEIGHTS},
                            "hallucination": False,
                        },
                    },
                    {
                        "case_id": f"{category}-{mode}-opportunity",
                        "category": category,
                        "mode": mode,
                        "base": {
                            "scores": {name: 3 for name in score.WEIGHTS},
                            "hallucination": True,
                        },
                        "lora": {
                            "scores": {name: 4 for name in score.WEIGHTS},
                            "hallucination": False,
                        },
                    },
                ]
            )

    result = score.score_comparison(
        rows,
        {
            "mean_gain_min": 0.35,
            "direct_gain_min": 0.0,
            "opportunity_win_rate_min": 0.5,
            "hallucination_rate_not_worse": True,
        },
        split="dev",
    )

    assert result["lora_wins"] == 10
    assert result["opportunity_count"] == 10
    assert result["opportunity_wins"] == 10
    assert result["opportunity_win_rate"] == pytest.approx(1.0)
    assert result["checks"]["opportunity_win_rate"] is True
    assert "lora_wins" not in result["checks"]
    assert result["passed"] is True


def test_v7_evaluation_configs_separate_generation_and_stage_thresholds():
    generation = json.loads((DAY26 / "configs/evaluation_generation_v7.json").read_text())
    dev = json.loads((DAY26 / "configs/evaluation_scoring_v7_dev.json").read_text())
    final = json.loads((DAY26 / "configs/evaluation_scoring_v7_final.json").read_text())
    assert generation["generation"]["do_sample"] is False
    assert generation["generation"]["max_new_tokens"] == 256
    assert dev["split"] == "dev" and final["split"] == "final"
    assert dev["pass_thresholds"]["mean_gain_min"] == 0.35
    assert dev["pass_thresholds"]["direct_gain_min"] == 0.0
    assert final["pass_thresholds"]["mean_gain_min"] == 0.5
    assert final["pass_thresholds"]["lora_wins_min"] == 12


def test_v8_evaluation_configs_freeze_the_same_opportunity_gate_before_results():
    dev = json.loads((DAY26 / "configs/evaluation_scoring_v8_dev.json").read_text())
    final = json.loads((DAY26 / "configs/evaluation_scoring_v8_final.json").read_text())
    expected = {
        "mean_gain_min": 0.35,
        "direct_gain_min": 0.0,
        "opportunity_win_rate_min": 0.5,
        "hallucination_rate_not_worse": True,
    }
    assert dev["split"] == "dev" and final["split"] == "final"
    assert dev["pass_thresholds"] == expected
    assert final["pass_thresholds"] == expected


def test_v7_manifest_rejects_old_dev_final_and_cross_split_bytes():
    acquire = load_script("acquire_source_images_v7")
    legacy = json.loads((DAY26 / "source/data/source_image_manifest.json").read_text())
    specs = acquire.build_source_specs(legacy)
    manifest = []
    for index, row in enumerate(specs, start=1):
        item = dict(row)
        item["sha256"] = row.get("expected_sha256", f"{index + 500:064x}")
        item["file_bytes"] = index + 100
        manifest.append(item)
    acquire.validate_frozen_manifest(manifest, legacy)

    old_heldout = next(row for row in legacy if row["split"] in {"dev", "final"})
    bad = [dict(row) for row in manifest]
    bad[-1]["sha256"] = old_heldout["sha256"]
    with pytest.raises(ValueError, match="prior heldout"):
        acquire.validate_frozen_manifest(bad, legacy)

    bad = [dict(row) for row in manifest]
    bad[-1]["sha256"] = bad[-2]["sha256"]
    with pytest.raises(ValueError, match="duplicate|cross-split"):
        acquire.validate_frozen_manifest(bad, legacy)


def test_v7_downloader_is_bounded_and_never_uses_a_shell(monkeypatch):
    acquire = load_script("acquire_source_images_v7")
    captured = {}

    class Result:
        stdout = b"real-image-bytes"

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return Result()

    monkeypatch.setattr(acquire.subprocess, "run", fake_run)
    assert acquire.download_bytes("https://example.org/image.png") == b"real-image-bytes"
    assert captured["command"][0] == "curl"
    assert "--retry-all-errors" in captured["command"]
    assert "--max-time" in captured["command"]
    assert captured["kwargs"]["check"] is True
    assert "shell" not in captured["kwargs"]


def test_v7_downloader_uses_frozen_direct_fallback_after_redirect_timeout(monkeypatch):
    acquire = load_script("acquire_source_images_v7")
    source_url = next(iter(acquire.DIRECT_DOWNLOAD_FALLBACKS))
    expected_fallback = acquire.DIRECT_DOWNLOAD_FALLBACKS[source_url]
    commands = []

    class Result:
        stdout = b"fallback-image-bytes"

    def fake_run(command, **kwargs):
        commands.append(command)
        if len(commands) == 1:
            raise subprocess.CalledProcessError(28, command)
        return Result()

    monkeypatch.setattr(acquire.subprocess, "run", fake_run)
    assert acquire.download_bytes(source_url) == b"fallback-image-bytes"
    assert commands[0][-1] == source_url
    assert commands[1][-1] == expected_fallback


def test_v7_acquisition_reuses_preseeded_bytes_with_matching_frozen_hash(tmp_path, monkeypatch):
    acquire = load_script("acquire_source_images_v7")
    raw = b"preseeded-image-bytes"
    target = tmp_path / "images_v7/dev/example.png"
    target.parent.mkdir(parents=True)
    target.write_bytes(raw)
    spec = {
        "image_relative_path": "images_v7/dev/example.png",
        "expected_sha256": hashlib.sha256(raw).hexdigest(),
        "download_url": "https://example.org/should-not-be-called.png",
    }

    def unexpected_download(_url):
        raise AssertionError("matching preseeded bytes should bypass the network")

    monkeypatch.setattr(acquire, "download_bytes", unexpected_download)
    assert acquire.obtain_new_bytes(spec, tmp_path) == raw


def test_v7_acquisition_refuses_to_overwrite_different_existing_bytes(tmp_path):
    acquire = load_script("acquire_source_images_v7")
    path = tmp_path / "image.png"
    path.write_bytes(b"frozen-original")
    expected = hashlib.sha256(b"frozen-original").hexdigest()

    assert acquire.ensure_exact_bytes(path, b"frozen-original", expected) == expected
    with pytest.raises(ValueError, match="refusing to overwrite"):
        acquire.ensure_exact_bytes(path, b"different", hashlib.sha256(b"different").hexdigest())


def test_v7_repair_dataset_only_reweights_frozen_training_rows():
    repair = load_script("build_v7_repair_dataset")
    public_rows = json.loads((DAY26 / "source/data/week5_vlm_train_v7.json").read_text())
    internal_rows = json.loads(
        (DAY26 / "source/data/week5_vlm_train_v7_internal.json").read_text()
    )

    weighted = repair.build_repair_dataset(public_rows, internal_rows, seed=2608)
    categories = Counter(
        internal_rows[index]["category"]
        for index in repair.repair_source_indices(internal_rows)
    )

    assert len(weighted) == 280
    assert categories == Counter({"natural_scene": 120, "ui": 120, "ocr": 40})
    assert all(row in public_rows for row in weighted)
    assert not any("dev" in image or "final" in image for row in weighted for image in row["images"])
    assert weighted == repair.build_repair_dataset(public_rows, internal_rows, seed=2608)


def test_v8_dataset_balances_400_train_rows_without_heldout_media():
    """Catches corrective exports that leak held-out media or lose category balance."""

    builder = load_script("build_v8_corrective_dataset")
    public = json.loads((DAY26 / "source/data/week5_vlm_train_v7.json").read_text())
    internal = json.loads(
        (DAY26 / "source/data/week5_vlm_train_v7_internal.json").read_text()
    )

    public_v8, internal_v8 = builder.build_corrective_train(public, internal)

    assert len(public_v8) == len(internal_v8) == 400
    assert public_v8[:200] == public
    assert Counter(row["category"] for row in internal_v8) == Counter(
        {category: 80 for category in builder.CATEGORIES}
    )
    assert Counter(row["variant"] for row in internal_v8) == Counter(
        {"v7-original": 200, "v8-paraphrase": 200}
    )
    assert all(row["split"] == "train" for row in internal_v8)
    assert all(
        len(row["images"]) == 1
        and row["messages"][0]["content"].count("<image>") == 1
        and "/train/" in row["images"][0]
        for row in public_v8
    )
    assert not any(
        "/dev/" in row["images"][0] or "/final/" in row["images"][0]
        for row in public_v8
    )


def test_v8_dataset_promotes_only_the_unopened_v7_final_to_development():
    """Catches reusing the already-exposed v7 dev split as fresh development data."""

    builder = load_script("build_v8_corrective_dataset")
    final_public = json.loads((DAY26 / "source/data/week5_vlm_final_v7.json").read_text())
    final_internal = json.loads(
        (DAY26 / "source/data/week5_vlm_final_v7_internal.json").read_text()
    )
    train_public = json.loads((DAY26 / "source/data/week5_vlm_train_v7.json").read_text())
    train_internal = json.loads(
        (DAY26 / "source/data/week5_vlm_train_v7_internal.json").read_text()
    )

    dev_public, dev_internal = builder.promote_unopened_final(
        final_public, final_internal
    )
    corrective_public, _ = builder.build_corrective_train(train_public, train_internal)

    assert len(dev_public) == len(dev_internal) == 20
    assert Counter(row["category"] for row in dev_internal) == Counter(
        {category: 4 for category in builder.CATEGORIES}
    )
    assert Counter(row["mode"] for row in dev_internal) == Counter(
        {"direct": 10, "verification": 10}
    )
    assert all(row["split"] == "dev" for row in dev_internal)
    assert all("v8-dev" in row["id"] for row in dev_internal)
    assert all("v8-dev" in row["source_image_id"] for row in dev_internal)
    assert {
        row["messages"][0]["content"] for row in corrective_public
    }.isdisjoint({row["messages"][0]["content"] for row in dev_public})
    assert {
        row["image_sha256"] for row in train_internal
    }.isdisjoint({row["image_sha256"] for row in dev_internal})


def test_v8_dataset_refuses_promotion_after_any_v7_final_inference_artifact():
    """Catches relabeling a final split after model outputs have exposed it."""

    builder = load_script("build_v8_corrective_dataset")
    builder.assert_v7_final_unopened([])
    with pytest.raises(ValueError, match="already been opened"):
        builder.assert_v7_final_unopened([Path("results/v7/raw_v7_final.jsonl")])


def test_v8_candidate_configs_and_preregistration_bound_the_search_to_two_runs():
    """Catches adding an unregistered third candidate or changing frozen modules."""

    config_dir = DAY26 / "configs"
    candidate_e = json.loads(
        (config_dir / "qwen2vl_lora_v8_e_balanced_continue.yaml").read_text()
    )
    candidate_f = json.loads(
        (config_dir / "qwen2vl_lora_v8_f_balanced_restart.yaml").read_text()
    )
    prereg = json.loads(
        (DAY26 / "source/results/v8/preregistration.json").read_text()
    )

    for config in (candidate_e, candidate_f):
        assert config["model_revision"] == "eed13092ef92e448dd6875b2a00151bd3f7db0ac"
        assert config["dataset"] == "day26_train_v8_corrective"
        assert config["eval_dataset"] == "day26_dev_v8"
        assert config["freeze_vision_tower"] is True
        assert config["freeze_multi_modal_projector"] is True
        assert config["freeze_language_model"] is False
        assert config["lora_rank"] == 16 and config["lora_alpha"] == 32
        assert config["lora_target"] == (
            "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
        )
        assert config["image_max_pixels"] == 262144
        assert config["seed"] == 42

    assert candidate_e["adapter_name_or_path"].endswith("/runs/v7_b_lr2e5_e1")
    assert candidate_e["learning_rate"] == 0.000002
    assert candidate_e["num_train_epochs"] == 0.5
    assert "adapter_name_or_path" not in candidate_f
    assert candidate_f["learning_rate"] == 0.00001
    assert candidate_f["num_train_epochs"] == 1.0

    assert prereg["status"] == "PRE_REGISTERED"
    assert prereg["max_candidates"] == 2
    assert [row["candidate_id"] for row in prereg["candidates"]] == ["v8-e", "v8-f"]
    assert prereg["candidates"][1]["allowed_only_if"] == "v8-e-dev-fails"
    assert prereg["stop_rule"] == "stop-after-first-dev-pass-or-two-dev-fails"
    assert prereg["final_status"] == "NOT_ACQUIRED_UNTIL_DEV_PASS"
    assert prereg["dev_gate"] == {
        "mean_gain_min": 0.35,
        "direct_gain_min": 0.0,
        "opportunity_win_rate_min": 0.5,
        "hallucination_rate_not_worse": True,
    }
