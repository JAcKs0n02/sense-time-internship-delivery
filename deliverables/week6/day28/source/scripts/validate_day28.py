"""Run the independent Day 28 acceptance checks and write their actual result."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable


DAY28_SOURCE = Path(__file__).resolve().parents[1]
WEEK6_SOURCE = DAY28_SOURCE.parents[1] / "source"
RESULT_PATH = DAY28_SOURCE / "results" / "day28_validation.json"
KNOWLEDGE_BASE_PATH = DAY28_SOURCE / "data" / "knowledge_base.json"
sys.path.insert(0, str(WEEK6_SOURCE))

_REQUIRED_ROOT_FIELDS = {"schema_version", "products_sha256", "products"}
_REQUIRED_PRODUCT_FIELDS = {
    "product_id",
    "name",
    "aliases",
    "keywords",
    "price_cny",
    "shipping_cny",
    "stock",
    "description",
}


def _run_check(name: str, check: Callable[[], Any]) -> tuple[dict[str, Any], Any | None]:
    try:
        actual = check()
        return {"name": name, "status": "PASS", "actual": actual}, actual
    except Exception as error:  # validator must report failed checks rather than hide them
        return {"name": name, "status": "FAIL", "error": str(error)}, None


def _canonical_products_sha256(products: list[object]) -> str:
    encoded = json.dumps(
        products, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_knowledge_base(path: Path) -> dict[str, Any]:
    """Independently validate the frozen local JSON and recompute its evidence."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid_knowledge_base") from error
    if not isinstance(payload, dict) or set(payload) != _REQUIRED_ROOT_FIELDS:
        raise ValueError("invalid_root_schema")
    if payload["schema_version"] != "1.0":
        raise ValueError("invalid_schema_version")
    products = payload["products"]
    if not isinstance(products, list):
        raise ValueError("invalid_products")
    declared_sha = payload["products_sha256"]
    if (
        not isinstance(declared_sha, str)
        or len(declared_sha) != 64
        or any(character not in "0123456789abcdef" for character in declared_sha)
    ):
        raise ValueError("invalid_knowledge_base_sha256")

    product_ids: set[str] = set()
    for product in products:
        if not isinstance(product, dict) or set(product) != _REQUIRED_PRODUCT_FIELDS:
            raise ValueError("invalid_product_schema")
        product_id = product["product_id"]
        if not isinstance(product_id, str) or not product_id.strip():
            raise ValueError("invalid_product_id")
        if product_id in product_ids:
            raise ValueError("duplicate_product_id")
        product_ids.add(product_id)
        for field in ("name", "description"):
            if not isinstance(product[field], str) or not product[field].strip():
                raise ValueError(f"invalid_{field}")
        for field in ("aliases", "keywords"):
            values = product[field]
            if not isinstance(values, list) or not values or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise ValueError(f"invalid_{field}")
        for field in ("price_cny", "shipping_cny"):
            value = product[field]
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"invalid_{field}")
        if type(product["stock"]) is not int or product["stock"] < 0:
            raise ValueError("invalid_stock")

    recomputed_sha = _canonical_products_sha256(products)
    if declared_sha != recomputed_sha:
        raise ValueError("knowledge_base_sha256_mismatch")
    return {
        "schema_version": payload["schema_version"],
        "product_count": len(products),
        "declared_products_sha256": declared_sha,
        "recomputed_products_sha256": recomputed_sha,
    }


def _write_result(result: dict[str, Any]) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    checks: list[dict[str, Any]] = []
    knowledge_base_sha256: str | None = None

    knowledge_check, knowledge_evidence = _run_check(
        "knowledge_base", lambda: validate_knowledge_base(KNOWLEDGE_BASE_PATH)
    )
    checks.append(knowledge_check)
    if isinstance(knowledge_evidence, dict):
        knowledge_base_sha256 = knowledge_evidence["recomputed_products_sha256"]

    try:
        from week6_agent.tools.calculator import CalculatorTool
        from week6_agent.tools.knowledge_retrieval import KnowledgeRetrievalTool

        checks.append(
            {
                "name": "imports",
                "status": "PASS",
                "actual": [CalculatorTool.__name__, KnowledgeRetrievalTool.__name__],
            }
        )
    except Exception as error:
        checks.append({"name": "imports", "status": "FAIL", "error": str(error)})
        result = {
            "schema_version": "1.0",
            "status": "FAIL",
            "knowledge_base_sha256": knowledge_base_sha256,
            "checks": checks,
        }
        _write_result(result)
        return 1

    checks.append(
        _run_check(
            "calculator_safe_case",
            lambda: CalculatorTool().run("123 * 456").data["value"] == 56088,
        )[0]
    )
    checks.append(
        _run_check(
            "calculator_unsafe_case",
            lambda: (
                CalculatorTool().run("__import__('os').system('id')").status == "rejected"
                and CalculatorTool().run("__import__('os').system('id')").error_code
                == "forbidden_node"
            ),
        )[0]
    )
    checks.append(
        _run_check(
            "calculator_limit_case",
            lambda: CalculatorTool().run("2 ** 65").error_code == "limit_exceeded",
        )[0]
    )
    checks.append(
        _run_check(
            "retrieval_alias_case",
            lambda: (
                KnowledgeRetrievalTool(KNOWLEDGE_BASE_PATH).search("星云键盘").status == "success"
                and KnowledgeRetrievalTool(KNOWLEDGE_BASE_PATH)
                .search("星云键盘")
                .data["price_cny"]
                >= 0
            ),
        )[0]
    )
    checks.append(
        _run_check(
            "retrieval_not_found_case",
            lambda: KnowledgeRetrievalTool(KNOWLEDGE_BASE_PATH).search("不存在的产品").status
            == "not_found",
        )[0]
    )

    status = "PASS" if all(check["status"] == "PASS" and check.get("actual") is not False for check in checks) else "FAIL"
    result = {
        "schema_version": "1.0",
        "status": status,
        "knowledge_base_sha256": knowledge_base_sha256,
        "checks": checks,
    }
    _write_result(result)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
