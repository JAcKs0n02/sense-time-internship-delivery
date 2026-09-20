import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_opencompass_smoke.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_opencompass_smoke", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_smoke_payload_keeps_one_subject_and_duplicates_model_backend():
    module = load_module()
    source = {
        "datasets": [{"abbr": "ceval-computer_network"}, {"abbr": "ceval-os"}],
        "models": [
            {
                "abbr": "source",
                "path": "/models/base",
                "type": "opencompass.models.HuggingFacewithChatTemplate",
                "batch_size": 4,
            }
        ],
        "work_dir": "/old",
    }

    smoke = module.build_smoke_payload(
        source,
        base_path="/models/base",
        best_path="/models/best",
    )

    assert smoke["datasets"] == [{"abbr": "ceval-computer_network"}]
    assert [model["abbr"] for model in smoke["models"]] == ["base", "best_sft"]
    assert [model["path"] for model in smoke["models"]] == ["/models/base", "/models/best"]
    assert smoke["models"][0]["type"] == smoke["models"][1]["type"]
    assert "work_dir" not in smoke
