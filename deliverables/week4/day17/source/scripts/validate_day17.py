#!/usr/bin/env python3
"""Validate the frozen Week 4 Day 17 preference-data contract."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_TYPES = (
    "factuality",
    "safety",
    "completeness",
    "helpfulness",
    "format",
)
TAXONOMY_REQUIRED_LISTS = (
    "chosen_criteria",
    "rejected_patterns",
    "boundary_rules",
    "quality_checks",
    "example_ids",
)
EXAMPLE_REQUIRED_FIELDS = (
    "id",
    "preference_type",
    "conversations",
    "chosen",
    "rejected",
    "construction_reason",
    "source",
    "quality_review",
)


def non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def valid_message(value: Any, expected_role: str) -> bool:
    return (
        isinstance(value, dict)
        and value.get("from") == expected_role
        and non_empty_string(value.get("value"))
    )


def validate_taxonomy(taxonomy: Any, errors: set[str]) -> dict[str, dict[str, Any]]:
    if not isinstance(taxonomy, dict):
        errors.add("taxonomy.not_object")
        return {}
    items = taxonomy.get("preference_types")
    if not isinstance(items, list):
        errors.add("taxonomy.preference_types_not_list")
        return {}

    by_id: dict[str, dict[str, Any]] = {}
    ids: list[str] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            errors.add(f"taxonomy.item_not_object:{index}")
            continue
        category = item.get("id")
        if not non_empty_string(category):
            errors.add(f"taxonomy.id_missing:{index}")
            continue
        ids.append(category)
        if category in by_id:
            errors.add(f"taxonomy.duplicate_id:{category}")
        by_id[category] = item
        if not non_empty_string(item.get("name_zh")):
            errors.add(f"taxonomy.name_zh_missing:{category}")
        if not non_empty_string(item.get("definition")):
            errors.add(f"taxonomy.definition_missing:{category}")
        for field in TAXONOMY_REQUIRED_LISTS:
            value = item.get(field)
            if not isinstance(value, list) or not value or not all(non_empty_string(entry) for entry in value):
                errors.add(f"taxonomy.{field}_invalid:{category}")

    if set(ids) != set(EXPECTED_TYPES) or len(ids) != len(EXPECTED_TYPES):
        errors.add("taxonomy.category_set_mismatch")
    return by_id


def validate_examples(examples_payload: Any, errors: set[str]) -> tuple[Counter[str], list[dict[str, Any]]]:
    if not isinstance(examples_payload, dict):
        errors.add("examples.not_object")
        return Counter(), []
    if examples_payload.get("format") != "llamafactory_sharegpt_ranking":
        errors.add("examples.format_invalid")
    examples = examples_payload.get("examples")
    if not isinstance(examples, list):
        errors.add("examples.examples_not_list")
        return Counter(), []

    ids: set[str] = set()
    counts: Counter[str] = Counter()
    safety_behaviors: set[str] = set()
    valid_objects: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        if not isinstance(example, dict):
            errors.add(f"examples.item_not_object:{index}")
            continue
        valid_objects.append(example)
        example_id = example.get("id")
        label = example_id if non_empty_string(example_id) else str(index)
        for field in EXAMPLE_REQUIRED_FIELDS:
            if field not in example:
                errors.add(f"examples.field_missing:{label}:{field}")

        if not non_empty_string(example_id):
            errors.add(f"examples.id_missing:{index}")
        elif example_id in ids:
            errors.add(f"examples.duplicate_id:{example_id}")
        else:
            ids.add(example_id)

        category = example.get("preference_type")
        if category not in EXPECTED_TYPES:
            errors.add(f"examples.category_invalid:{label}")
        else:
            counts[category] += 1

        conversations = example.get("conversations")
        if (
            not isinstance(conversations, list)
            or not conversations
            or not all(valid_message(message, "human") for message in conversations)
        ):
            errors.add(f"examples.conversations_invalid:{label}")
        if not valid_message(example.get("chosen"), "gpt"):
            errors.add(f"examples.chosen_invalid:{label}")
        if not valid_message(example.get("rejected"), "gpt"):
            errors.add(f"examples.rejected_invalid:{label}")

        chosen = example.get("chosen")
        rejected = example.get("rejected")
        if (
            isinstance(chosen, dict)
            and isinstance(rejected, dict)
            and isinstance(chosen.get("value"), str)
            and isinstance(rejected.get("value"), str)
            and chosen["value"].strip() == rejected["value"].strip()
        ):
            errors.add(f"examples.responses_not_distinct:{label}")
        if not non_empty_string(example.get("construction_reason")):
            errors.add(f"examples.construction_reason_missing:{label}")

        source = example.get("source")
        if not isinstance(source, dict) or not non_empty_string(source.get("type")):
            errors.add(f"examples.source_invalid:{label}")
        review = example.get("quality_review")
        if not isinstance(review, dict):
            errors.add(f"examples.quality_review_invalid:{label}")
        else:
            if review.get("status") != "approved":
                errors.add(f"examples.review_not_approved:{label}")
            if review.get("single_dimension") is not True:
                errors.add(f"examples.not_single_dimension:{label}")
            if review.get("safe_to_publish") is not True:
                errors.add(f"examples.not_safe_to_publish:{label}")

        if category == "safety":
            safety_behavior = example.get("safety_behavior")
            if safety_behavior not in {"refuse_harmful", "assist_benign"}:
                errors.add(f"examples.safety_behavior_invalid:{label}")
            else:
                safety_behaviors.add(safety_behavior)

    for category in EXPECTED_TYPES:
        if counts[category] < 2:
            errors.add(f"examples.category_below_minimum:{category}")
    if len(examples) < 10:
        errors.add("examples.total_below_minimum")
    if safety_behaviors != {"refuse_harmful", "assist_benign"}:
        errors.add("examples.safety_behavior_coverage_mismatch")
    return counts, valid_objects


def validate_cross_references(
    taxonomy_by_id: dict[str, dict[str, Any]],
    examples: list[dict[str, Any]],
    errors: set[str],
) -> None:
    actual: dict[str, set[str]] = {category: set() for category in EXPECTED_TYPES}
    for example in examples:
        category = example.get("preference_type")
        example_id = example.get("id")
        if category in actual and non_empty_string(example_id):
            actual[category].add(example_id)
    for category in EXPECTED_TYPES:
        item = taxonomy_by_id.get(category)
        if item is None:
            continue
        declared = item.get("example_ids")
        if isinstance(declared, list) and set(declared) != actual[category]:
            errors.add(f"taxonomy.example_ids_mismatch:{category}")


def validate_payloads(taxonomy: Any, examples_payload: Any) -> dict[str, Any]:
    """Return a stable, machine-readable validation summary."""
    errors: set[str] = set()
    taxonomy_by_id = validate_taxonomy(taxonomy, errors)
    counts, examples = validate_examples(examples_payload, errors)
    validate_cross_references(taxonomy_by_id, examples, errors)
    ordered_counts = {category: counts[category] for category in sorted(EXPECTED_TYPES)}
    return {
        "valid": not errors,
        "errors": sorted(errors),
        "category_counts": ordered_counts,
        "example_count": len(examples),
    }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = validate_payloads(load_json(args.taxonomy), load_json(args.examples))
    except (OSError, json.JSONDecodeError) as exc:
        result = {
            "valid": False,
            "errors": [f"input.read_error:{exc}"],
            "category_counts": {category: 0 for category in sorted(EXPECTED_TYPES)},
            "example_count": 0,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
