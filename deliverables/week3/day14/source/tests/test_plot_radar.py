import sys
from pathlib import Path


DAY14_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = DAY14_ROOT / "source" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from plot_radar import DIMENSIONS, build_radar_rows  # noqa: E402


def test_radar_uses_human_dimension_means_only() -> None:
    scores = [
        {
            "model_id": "rank-r8",
            "accuracy": 4.0,
            "completeness": 3.5,
            "logic": 4.5,
            "safety": 5.0,
            "format": 3.0,
            "weighted_total": 4.0,
            "automatic_score": 0.9,
        }
    ]

    rows = build_radar_rows(scores)

    assert rows == [
        {
            "model_id": "rank-r8",
            "accuracy": 4.0,
            "completeness": 3.5,
            "logic": 4.5,
            "safety": 5.0,
            "format": 3.0,
        }
    ]
    assert DIMENSIONS == ["accuracy", "completeness", "logic", "safety", "format"]
