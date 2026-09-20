#!/usr/bin/env python3
"""Independently validate the Week 2 Day 6 delivery."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_EXPECTED_COUNTS = {
    "alpaca_gpt4_zh": 2_000,
    "coig_pc": 2_000,
    "sharegpt_zh": 1_000,
}
RAW_FILES = {
    "alpaca_gpt4_zh": "alpaca_gpt4_zh_2k.jsonl",
    "coig_pc": "coig_pc_2k.jsonl",
    "sharegpt_zh": "sharegpt_zh_1k.jsonl",
}
MANIFEST_FIELDS = (
    "dataset_name",
    "source_url",
    "revision",
    "split",
    "license",
    "downloaded_at",
    "upstream_count",
    "sample_count",
    "sampling_method",
    "raw_file",
    "file_size_bytes",
    "sha256",
    "notes",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    return records


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_raw_manifests(root: Path) -> dict[str, Path]:
    """Generate CSV and Markdown manifests from verified collection metadata."""
    root = Path(root)
    manifest_dir = root / "source" / "manifests"
    metadata_path = manifest_dir / "collection_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows = []
    for source in metadata["sources"]:
        raw_path = root / source["raw_file"]
        actual_size = raw_path.stat().st_size
        actual_sha256 = _sha256_file(raw_path)
        if actual_size != source["file_size_bytes"]:
            raise ValueError(f"{source['source_name']}: raw file size changed")
        if actual_sha256 != source["sha256"]:
            raise ValueError(f"{source['source_name']}: raw file SHA-256 changed")
        notes = (
            f"source_name={source['source_name']}; "
            f"config={source.get('config_name') or 'default'}; "
            f"gated={str(source.get('gated', False)).lower()}; "
            f"upstream_file={source.get('upstream_file') or 'dataset_loader'}; "
            "upstream_file_size_bytes="
            f"{source.get('upstream_file_size_bytes') or 'not_recorded'}; "
            "upstream_file_sha256="
            f"{source.get('upstream_file_sha256') or 'not_recorded'}"
        )
        rows.append(
            {
                "dataset_name": source["dataset_name"],
                "source_url": source["source_url"],
                "revision": source["revision"],
                "split": source["split"],
                "license": source["license_text"],
                "downloaded_at": source["downloaded_at"],
                "upstream_count": source["upstream_count"],
                "sample_count": source["sample_count"],
                "sampling_method": source["sampling_method"],
                "raw_file": source["raw_file"],
                "file_size_bytes": actual_size,
                "sha256": actual_sha256,
                "notes": notes,
            }
        )

    csv_path = manifest_dir / "raw_data_manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=MANIFEST_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = manifest_dir / "raw_data_manifest.md"
    table_lines = [
        "# Day 6 原始数据归档清单",
        "",
        f"- 固定抽样种子：`{metadata['seed']}`",
        f"- 归档生成时间：`{metadata['generated_at']}`",
        f"- 抽样语义样本总数：`{metadata['total_sample_count']}`",
        "",
        "| 数据集 | revision | split | 许可/条件 | 上游条数 | 抽样条数 | 原始子集 | SHA-256 |",
        "|---|---|---|---|---:|---:|---|---|",
    ]
    for row in rows:
        table_lines.append(
            "| [{dataset_name}]({source_url}) | `{revision}` | `{split}` | {license} | "
            "{upstream_count} | {sample_count} | `{raw_file}` | `{sha256}` |".format(
                **row
            )
        )
    table_lines.extend(
        [
            "",
            "## 上游文件固定信息",
            "",
            "| 数据集 | 上游文件 | 文件大小（bytes） | 上游文件 SHA-256 |",
            "|---|---|---:|---|",
        ]
    )
    for source in metadata["sources"]:
        table_lines.append(
            "| {dataset_name} | `{upstream_file}` | {upstream_size} | "
            "`{upstream_sha256}` |".format(
                dataset_name=source["dataset_name"],
                upstream_file=source.get("upstream_file") or "dataset_loader",
                upstream_size=source.get("upstream_file_size_bytes")
                or "not_recorded",
                upstream_sha256=source.get("upstream_file_sha256")
                or "not_recorded",
            )
        )
    table_lines.extend(
        [
            "",
            "## 抽样口径",
            "",
            rows[0]["sampling_method"] if rows else "",
            "",
            "所有 `raw/` 文件都是固定 revision 上游记录的确定性子集；",
            "Day 6 未执行 HTML 清理、控制字符清理、token 截断或去重。",
            "",
        ]
    )
    markdown_path.write_text("\n".join(table_lines), encoding="utf-8")
    return {"csv": csv_path, "markdown": markdown_path}


def write_sha256_manifest(
    root: Path, paths: list[Path], output_path: Path
) -> Path:
    """Write deterministic SHA-256 lines using delivery-relative paths."""
    root = Path(root).resolve()
    output_path = Path(output_path)
    lines = []
    for path in sorted((Path(path).resolve() for path in paths), key=str):
        relative_path = path.relative_to(root).as_posix()
        lines.append(f"{_sha256_file(path)}  {relative_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _validate_alpaca_record(record: dict[str, Any], label: str) -> list[str]:
    errors = []
    if set(record) != {"instruction", "input", "output"}:
        errors.append(f"{label}: Alpaca fields must be instruction,input,output only")
        return errors
    for field in ("instruction", "input", "output"):
        if not isinstance(record[field], str):
            errors.append(f"{label}: {field} must be a string")
    if isinstance(record["instruction"], str) and not record["instruction"]:
        errors.append(f"{label}: instruction must be non-empty")
    if isinstance(record["output"], str) and not record["output"]:
        errors.append(f"{label}: output must be non-empty")
    return errors


def _validate_sharegpt_record(record: dict[str, Any], label: str) -> list[str]:
    if set(record) != {"conversations"}:
        return [f"{label}: ShareGPT record must contain conversations only"]
    messages = record["conversations"]
    if not isinstance(messages, list) or not messages or len(messages) % 2:
        return [f"{label}: conversations must contain complete message pairs"]
    errors = []
    for index, message in enumerate(messages):
        expected_role = "human" if index % 2 == 0 else "gpt"
        if not isinstance(message, dict) or set(message) != {"from", "value"}:
            errors.append(f"{label}: message {index} has invalid fields")
            continue
        if message["from"] != expected_role:
            errors.append(
                f"{label}: message {index} role is {message['from']!r}, "
                f"expected {expected_role!r}"
            )
        if not isinstance(message["value"], str) or not message["value"]:
            errors.append(f"{label}: message {index} value must be a non-empty string")
    return errors


def validate_delivery(
    root: Path,
    expected_counts: dict[str, int] | None = None,
    require_manifest: bool = True,
) -> dict[str, Any]:
    """Return a complete validation report without mutating the delivery."""
    root = Path(root)
    expected_counts = dict(expected_counts or DEFAULT_EXPECTED_COUNTS)
    raw_dir = root / "source" / "data" / "raw"
    formatted_dir = root / "source" / "data" / "formatted"
    interim_dir = root / "source" / "data" / "interim"
    errors: list[str] = []
    counts: dict[str, int] = {}

    for source, filename in RAW_FILES.items():
        path = raw_dir / filename
        try:
            records = _read_jsonl(path)
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
            records = []
        counts[source] = len(records)
        if len(records) != expected_counts[source]:
            errors.append(
                f"{source}: expected {expected_counts[source]} raw records, "
                f"found {len(records)}"
            )
        seen_source_indexes = set()
        for index, record in enumerate(records):
            provenance = record.get("_provenance")
            if not isinstance(provenance, dict):
                errors.append(f"{source}[{index}]: missing _provenance")
                continue
            if provenance.get("source") != source:
                errors.append(f"{source}[{index}]: provenance source mismatch")
            source_index = provenance.get("source_index")
            if not isinstance(source_index, int):
                errors.append(f"{source}[{index}]: source_index must be an integer")
            elif source_index in seen_source_indexes:
                errors.append(f"{source}[{index}]: duplicate source_index")
            else:
                seen_source_indexes.add(source_index)
            if not isinstance(provenance.get("revision"), str):
                errors.append(f"{source}[{index}]: revision must be a string")

    try:
        alpaca_records = _read_jsonl(
            formatted_dir / "week2_5k_alpaca.jsonl"
        )
    except (OSError, ValueError) as exc:
        errors.append(str(exc))
        alpaca_records = []
    try:
        sharegpt_records = _read_jsonl(
            formatted_dir / "week2_5k_sharegpt.jsonl"
        )
    except (OSError, ValueError) as exc:
        errors.append(str(exc))
        sharegpt_records = []
    try:
        provenance_records = _read_jsonl(interim_dir / "day6_provenance.jsonl")
    except (OSError, ValueError) as exc:
        errors.append(str(exc))
        provenance_records = []

    expected_total = sum(expected_counts.values())
    counts["formatted_alpaca"] = len(alpaca_records)
    counts["formatted_sharegpt"] = len(sharegpt_records)
    counts["provenance"] = len(provenance_records)
    for label, count in (
        ("formatted_alpaca", len(alpaca_records)),
        ("formatted_sharegpt", len(sharegpt_records)),
        ("provenance", len(provenance_records)),
    ):
        if count != expected_total:
            errors.append(f"{label}: expected {expected_total}, found {count}")

    for index, record in enumerate(alpaca_records):
        errors.extend(_validate_alpaca_record(record, f"alpaca[{index}]"))
    for index, record in enumerate(sharegpt_records):
        errors.extend(_validate_sharegpt_record(record, f"sharegpt[{index}]"))

    sample_ids = [record.get("sample_id") for record in provenance_records]
    if any(not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids):
        errors.append("provenance: every sample_id must be a non-empty string")
    if len(set(sample_ids)) != len(sample_ids):
        errors.append("provenance: duplicate sample_id values")

    provenance_counts = {
        source: sum(record.get("source") == source for record in provenance_records)
        for source in expected_counts
    }
    if provenance_counts != expected_counts:
        errors.append(
            f"provenance source counts mismatch: {provenance_counts!r} "
            f"!= {expected_counts!r}"
        )

    if require_manifest:
        manifest_dir = root / "source" / "manifests"
        csv_path = manifest_dir / "raw_data_manifest.csv"
        markdown_path = manifest_dir / "raw_data_manifest.md"
        if not csv_path.is_file():
            errors.append("missing manifest: raw_data_manifest.csv")
        else:
            try:
                with csv_path.open(newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    manifest_rows = list(reader)
                    if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
                        errors.append("manifest CSV fields do not match required fields")
                if len(manifest_rows) != len(expected_counts):
                    errors.append(
                        f"manifest dataset rows: expected {len(expected_counts)}, "
                        f"found {len(manifest_rows)}"
                    )
                for index, row in enumerate(manifest_rows):
                    for field in MANIFEST_FIELDS:
                        if not row.get(field):
                            errors.append(f"manifest row {index}: empty {field}")
                    raw_file = row.get("raw_file", "")
                    raw_path = root / raw_file
                    if not raw_path.is_file():
                        errors.append(f"manifest row {index}: missing raw file {raw_file}")
                        continue
                    if row.get("file_size_bytes") != str(raw_path.stat().st_size):
                        errors.append(f"manifest row {index}: file size mismatch")
                    if row.get("sha256") != _sha256_file(raw_path):
                        errors.append(f"manifest row {index}: SHA-256 mismatch")
            except (OSError, csv.Error) as exc:
                errors.append(f"manifest CSV error: {exc}")
        if not markdown_path.is_file():
            errors.append("missing manifest: raw_data_manifest.md")
        elif not markdown_path.read_text(encoding="utf-8").strip():
            errors.append("manifest Markdown is empty")

    return {
        "valid": not errors,
        "counts": counts,
        "expected_counts": expected_counts,
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    write_raw_manifests(args.root)
    report = validate_delivery(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    hash_path = args.root / "source" / "results" / "day6_files.sha256"
    hash_inputs = [
        path
        for path in (args.root / "source").rglob("*")
        if path.is_file()
        and path != hash_path
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    ]
    write_sha256_manifest(args.root, hash_inputs, hash_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
