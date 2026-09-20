#!/usr/bin/env python3
"""Render deterministic Day24 original/heatmap/overlay triptychs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_heatmap(values: Any) -> np.ndarray:
    """Apply the pre-registered per-case linear min-max display transform."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("heatmap must be a two-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError("heatmap contains non-finite values")
    minimum = float(array.min())
    maximum = float(array.max())
    if maximum <= minimum:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - minimum) / (maximum - minimum)).astype(np.float32, copy=False)


def render_triptych(
    *,
    image_path: Path,
    heatmap: Any,
    output_path: Path,
    title: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    if config.get("normalization") != "linear_minmax_per_case":
        raise ValueError("unsupported or unfrozen normalization")
    alpha = float(config["overlay_alpha"])
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("overlay_alpha must be between 0 and 1")
    normalized = normalize_heatmap(heatmap)

    with Image.open(image_path) as source:
        image = np.asarray(source.convert("RGB"))

    figure = plt.figure(
        figsize=(float(config["figure_width_inches"]), float(config["figure_height_inches"])),
        dpi=int(config["figure_dpi"]),
    )
    axes = figure.subplots(1, 3)
    panels = ("Original", "Attention Heatmap", "Overlay")
    for axis, panel in zip(axes, panels):
        axis.set_title(panel)
        axis.set_axis_off()
    axes[0].imshow(image)
    axes[1].imshow(
        normalized,
        cmap=str(config["colormap"]),
        interpolation=str(config["interpolation"]),
        vmin=0.0,
        vmax=1.0,
    )
    axes[2].imshow(image)
    axes[2].imshow(
        normalized,
        cmap=str(config["colormap"]),
        interpolation=str(config["interpolation"]),
        vmin=0.0,
        vmax=1.0,
        alpha=alpha,
        extent=(0, image.shape[1], image.shape[0], 0),
    )
    figure.suptitle(title, fontsize=10)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.86, wspace=0.04)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=int(config["figure_dpi"]), format="png")
    plt.close(figure)

    with Image.open(output_path) as rendered:
        width, height = rendered.size
    return {
        "path": str(output_path),
        "width": width,
        "height": height,
        "sha256": sha256_file(output_path),
    }


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL line {line_number} must be an object")
        rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    serialized = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")


def render_all(
    *,
    data_root: Path,
    results_root: Path,
    visualization_config_path: Path,
) -> dict[str, Any]:
    config = read_json(visualization_config_path)
    metadata_path = results_root / "attention_metadata.jsonl"
    metadata_sha_before_render = sha256_file(metadata_path)
    rows = read_jsonl(metadata_path)
    if not rows:
        raise ValueError("attention metadata is empty")

    rendered_rows: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") != "PASS":
            raise ValueError(f"cannot render failed case: {row.get('case_id')}")
        image_path = data_root / str(row["image_relative_path"])
        if not image_path.is_file() or sha256_file(image_path) != row.get("image_sha256"):
            raise ValueError(f"image integrity check failed: {row.get('case_id')}")
        mean_path = results_root / str(row["array_paths"]["head_mean"])
        if not mean_path.is_file() or sha256_file(mean_path) != row["array_sha256"]["head_mean"]:
            raise ValueError(f"head-mean attention integrity check failed: {row.get('case_id')}")
        heatmap = np.load(mean_path, allow_pickle=False)
        if list(heatmap.shape) != list(row["head_mean_shape"]):
            raise ValueError(f"head-mean attention shape changed: {row.get('case_id')}")
        output_path = results_root / "heatmaps" / f"{row['case_id']}-triptych.png"
        title = (
            f"{row['case_id']} | target={row['target_display_label']} | "
            f"layer={row['layer_number_1_based']} (index={row['layer_index_0_based']})"
        )
        rendered = render_triptych(
            image_path=image_path,
            heatmap=heatmap,
            output_path=output_path,
            title=title,
            config=config,
        )
        rendered_rows.append(
            {
                **row,
                "visualization_config_sha256": sha256_file(visualization_config_path),
                "heatmap_path": str(output_path.relative_to(results_root)),
                "heatmap_sha256": rendered["sha256"],
                "heatmap_width": rendered["width"],
                "heatmap_height": rendered["height"],
            }
        )

    write_jsonl(metadata_path, rendered_rows)
    metadata_sha_after_render = sha256_file(metadata_path)
    run_summary_path = results_root / "run_summary.json"
    run_summary = read_json(run_summary_path)
    run_summary["metadata_sha256_at_extraction"] = run_summary.get(
        "metadata_sha256_at_extraction",
        run_summary.get("metadata_sha256", metadata_sha_before_render),
    )
    run_summary["metadata_sha256"] = metadata_sha_after_render
    run_summary["metadata_sha256_after_render"] = metadata_sha_after_render
    write_json(run_summary_path, run_summary)
    summary = {
        "schema_version": "1.0",
        "status": "PASS",
        "rendered_case_count": len(rendered_rows),
        "normalization": config["normalization"],
        "colormap": config["colormap"],
        "overlay_alpha": config["overlay_alpha"],
        "visualization_config_sha256": sha256_file(visualization_config_path),
        "metadata_sha256": metadata_sha_after_render,
    }
    write_json(results_root / "render_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--visualization-config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = render_all(
        data_root=args.data_root,
        results_root=args.results_root,
        visualization_config_path=args.visualization_config,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
