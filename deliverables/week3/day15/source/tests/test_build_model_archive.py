import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_model_archive.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_model_archive", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_archive_manifest_links_selection_load_gate_and_weight_inventory(tmp_path):
    module = load_module()
    selected = {
        "selected_run_id": "epoch-e5",
        "adapter_path": "/runs/epoch-e5/adapter",
        "adapter_model_sha256": "a" * 64,
    }
    load_validation = {
        "merged_model_load": "PASS",
        "model_path": "/models/best",
    }
    hashes = tmp_path / "files.sha256"
    hashes.write_text(
        "b" * 64
        + "  /models/best/model.safetensors\n"
        + "c" * 64
        + "  /models/best/config.json\n",
        encoding="utf-8",
    )
    inventory = tmp_path / "inventory.csv"
    inventory.write_text("model.safetensors,123\nconfig.json,10\n", encoding="utf-8")

    archive = module.build_archive(
        selected=selected,
        load_validation=load_validation,
        hashes_path=hashes,
        inventory_path=inventory,
        total_bytes=133,
    )

    assert archive["status"] == "archived_and_load_validated"
    assert archive["selected_run_id"] == "epoch-e5"
    assert archive["merged_model_path"] == "/models/best"
    assert archive["merged_model_load"] == "PASS"
    assert archive["merged_model_file_count"] == 2
    assert archive["merged_model_total_bytes"] == 133
    assert len(archive["merged_model_manifest_sha256"]) == 64
