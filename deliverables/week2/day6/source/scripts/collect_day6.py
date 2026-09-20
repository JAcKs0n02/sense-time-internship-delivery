#!/usr/bin/env python3
"""Collect deterministic fixed-revision subsets for Week 2 Day 6."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class SourceSpec:
    source_name: str
    dataset_name: str
    repo_id: str
    revision: str
    split: str
    config_name: str | None
    sample_count: int
    output_filename: str
    license_text: str
    gated: bool
    source_filename: str | None = None
    file_format: str | None = None
    upstream_file_size_bytes: int | None = None
    upstream_file_sha256: str | None = None


SOURCE_SPECS = (
    SourceSpec(
        source_name="alpaca_gpt4_zh",
        dataset_name="Alpaca-GPT4-zh",
        repo_id="llamafactory/alpaca_gpt4_zh",
        revision="065394ee242c43928298de8f43a8748ffd16f3e3",
        split="train",
        config_name=None,
        sample_count=2_000,
        output_filename="alpaca_gpt4_zh_2k.jsonl",
        license_text="Apache-2.0",
        gated=False,
        source_filename="alpaca_gpt4_data_zh.json",
        file_format="json",
        upstream_file_size_bytes=27_862_380,
        upstream_file_sha256=(
            "628c3b100e32af85ec8d5338c42745f6b82900f97cf4deeda307b98f94ab34ab"
        ),
    ),
    SourceSpec(
        source_name="coig_pc",
        dataset_name="COIG-PC Top200PerTask",
        repo_id="BAAI/COIG-PC",
        revision="4ca2778d69d7a9c17a0a6c1668c6e989b102176b",
        split="Top200PerTask",
        config_name=None,
        sample_count=2_000,
        output_filename="coig_pc_2k.jsonl",
        license_text=(
            "Sub-dataset license takes precedence; Apache-2.0 applies by "
            "default when a sub-dataset has no declared license; repository "
            "access conditions also apply"
        ),
        gated=True,
        source_filename=(
            "data/Top200PerTask-00000-of-00001-fe36897020230691.parquet"
        ),
        file_format="parquet",
        upstream_file_size_bytes=160_399_530,
        upstream_file_sha256=(
            "59a961b3517c3a46c572c6f031436552f863d3ee0331728901725ac668105219"
        ),
    ),
    SourceSpec(
        source_name="sharegpt_zh",
        dataset_name="ShareGPT Chinese",
        repo_id="FreedomIntelligence/sharegpt-chinese",
        revision="ef0a611d7c8c3cd7b17c152d4166c67a92015c2b",
        split="train",
        config_name=None,
        sample_count=1_000,
        output_filename="sharegpt_zh_1k.jsonl",
        license_text="Apache-2.0",
        gated=False,
        source_filename="sharegpt-chinese.json",
        file_format="json",
        upstream_file_size_bytes=220_056_589,
        upstream_file_sha256=(
            "bacd961782259b43364306885a984487964379d5f7aeec54c25814574ffe6855"
        ),
    ),
)


class _JsonArrayRecords:
    def __init__(self, path: Path):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, list) or not all(
            isinstance(record, dict) for record in value
        ):
            raise ValueError(f"{path}: expected a JSON array of objects")
        self._records = value

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        return iter(self._records)


class _ParquetRecords:
    def __init__(self, path: Path):
        import pyarrow.parquet as pq

        self._parquet_file = pq.ParquetFile(path)

    def __len__(self) -> int:
        return self._parquet_file.metadata.num_rows

    def __iter__(self):
        for batch in self._parquet_file.iter_batches(batch_size=1024):
            yield from batch.to_pylist()


def load_local_records(path: Path, file_format: str):
    """Return a sized iterable for a fixed JSON-array or Parquet source file."""
    if file_format == "json":
        return _JsonArrayRecords(path)
    if file_format == "parquet":
        return _ParquetRecords(path)
    raise ValueError(f"unsupported source file format: {file_format}")


def stable_sample(
    records: Iterable[dict[str, Any]],
    sample_count: int,
    seed: int,
    source_name: str,
    revision: str,
) -> list[dict[str, Any]]:
    """Select the lowest stable SHA-256 priorities and restore source order."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")

    heap: list[tuple[int, int, int, dict[str, Any], str]] = []
    upstream_count = 0
    for source_index, record in enumerate(records):
        upstream_count += 1
        priority_hex = hashlib.sha256(
            f"{seed}:{source_name}:{source_index}".encode("utf-8")
        ).hexdigest()
        priority = int(priority_hex, 16)
        item = (-priority, -source_index, source_index, record, priority_hex)
        if len(heap) < sample_count:
            heapq.heappush(heap, item)
            continue

        worst_priority = -heap[0][0]
        worst_index = -heap[0][1]
        if (priority, source_index) < (worst_priority, worst_index):
            heapq.heapreplace(heap, item)

    if upstream_count < sample_count:
        raise ValueError(
            f"requested {sample_count} records but upstream has {upstream_count}"
        )

    selected = []
    for _, _, source_index, record, priority_hex in sorted(
        heap, key=lambda item: item[2]
    ):
        output_record = dict(record)
        output_record["_provenance"] = {
            "source": source_name,
            "source_index": source_index,
            "revision": revision,
            "seed": seed,
            "sampling_priority_sha256": priority_hex,
        }
        selected.append(output_record)
    return selected


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_source(
    spec: SourceSpec,
    raw_dir: Path,
    cache_dir: Path,
    seed: int,
    dataset_loader=None,
    file_downloader=None,
) -> dict[str, Any]:
    """Load one pinned dataset, sample it, and archive the raw subset."""
    if dataset_loader is not None:
        loader_kwargs: dict[str, Any] = {
            "split": spec.split,
            "revision": spec.revision,
            "cache_dir": str(cache_dir),
        }
        if spec.config_name is not None:
            loader_kwargs["name"] = spec.config_name
        if spec.gated:
            loader_kwargs["token"] = True
        dataset = dataset_loader(spec.repo_id, **loader_kwargs)
    else:
        if not spec.source_filename or not spec.file_format:
            raise ValueError(f"{spec.source_name}: source file metadata is missing")
        if file_downloader is None:
            from huggingface_hub import hf_hub_download

            file_downloader = hf_hub_download
        local_source_path = file_downloader(
            repo_id=spec.repo_id,
            filename=spec.source_filename,
            repo_type="dataset",
            revision=spec.revision,
            cache_dir=str(cache_dir),
            token=True if spec.gated else None,
            resume_download=True,
        )
        dataset = load_local_records(Path(local_source_path), spec.file_format)
    upstream_count = len(dataset)
    selected = stable_sample(
        iter(dataset),
        sample_count=spec.sample_count,
        seed=seed,
        source_name=spec.source_name,
        revision=spec.revision,
    )

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / spec.output_filename
    with output_path.open("w", encoding="utf-8") as handle:
        for record in selected:
            handle.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )

    metadata = asdict(spec)
    metadata.update(
        {
            "source_url": f"https://huggingface.co/datasets/{spec.repo_id}",
            "downloaded_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(
                timespec="seconds"
            ),
            "upstream_count": upstream_count,
            "upstream_file": spec.source_filename,
            "sample_count": len(selected),
            "sampling_method": (
                f"lowest SHA-256 priority of '{seed}:<source>:<source_index>', "
                "then source_index ascending"
            ),
            "raw_file": f"source/data/raw/{spec.output_filename}",
            "file_size_bytes": output_path.stat().st_size,
            "sha256": _sha256_file(output_path),
        }
    )
    return metadata


def collect_all(
    specs: Iterable[SourceSpec],
    raw_dir: Path,
    cache_dir: Path,
    manifest_json: Path,
    seed: int,
    dataset_loader=None,
) -> dict[str, Any]:
    """Collect all requested sources and save exact machine-readable metadata."""
    source_metadata = [
        collect_source(
            spec,
            raw_dir=raw_dir,
            cache_dir=cache_dir,
            seed=seed,
            dataset_loader=dataset_loader,
        )
        for spec in specs
    ]
    result = {
        "pipeline": "Week 2 Day 6 deterministic collection",
        "seed": seed,
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(
            timespec="seconds"
        ),
        "total_sample_count": sum(item["sample_count"] for item in source_metadata),
        "sources": source_metadata,
    }
    manifest_json = Path(manifest_json)
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    manifest_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = collect_all(
        specs=SOURCE_SPECS,
        raw_dir=args.raw_dir,
        cache_dir=args.cache_dir,
        manifest_json=args.manifest_json,
        seed=args.seed,
    )
    summary = {
        "seed": result["seed"],
        "total_sample_count": result["total_sample_count"],
        "sources": {
            item["source_name"]: {
                "upstream_count": item["upstream_count"],
                "sample_count": item["sample_count"],
                "sha256": item["sha256"],
            }
            for item in result["sources"]
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
