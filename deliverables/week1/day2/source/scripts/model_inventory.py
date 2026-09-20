from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    root = args.model
    required = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    ]
    for name in required:
        path = root / name
        assert path.is_file() and path.stat().st_size > 0, name

    index = json.loads(
        (root / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    shards = sorted(set(index["weight_map"].values()))
    assert len(shards) == 4, shards
    for shard in shards:
        path = root / shard
        assert path.is_file() and path.stat().st_size > 1_000_000_000, shard

    partials = [
        path
        for path in root.rglob("*")
        if "incomplete" in path.name.lower()
    ]
    assert not partials, partials

    config = AutoConfig.from_pretrained(root, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    print("model_type", config.model_type)
    print("layers", config.num_hidden_layers)
    print("hidden_size", config.hidden_size)
    print("vocab", len(tokenizer))
    for shard in shards:
        path = root / shard
        print(path.name, path.stat().st_size)
    print("PASS model inventory")


if __name__ == "__main__":
    main()
