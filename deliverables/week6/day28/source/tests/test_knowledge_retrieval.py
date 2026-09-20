from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


WEEK6_SOURCE = Path(__file__).resolve().parents[3] / "source"
sys.path.insert(0, str(WEEK6_SOURCE))

from week6_agent.tools.knowledge_retrieval import KnowledgeRetrievalTool  # noqa: E402


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "knowledge_base.json"


def make_knowledge_base(products: list[dict[str, object]]) -> dict[str, object]:
    """Build a complete hand-authored fixture with its required content hash."""
    encoded = json.dumps(
        products, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "schema_version": "1.0",
        "products_sha256": hashlib.sha256(encoded).hexdigest(),
        "products": products,
    }


def write_knowledge_base(path: Path, products: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps(make_knowledge_base(products), ensure_ascii=False), encoding="utf-8"
    )


def fixture_products() -> list[dict[str, object]]:
    return [
        {
            "product_id": "nebula-keyboard",
            "name": "星云机械键盘",
            "aliases": ["星云键盘", "Nebula Keyboard"],
            "keywords": ["键盘", "机械", "无线"],
            "price_cny": 699,
            "shipping_cny": 20,
            "stock": 12,
            "description": "75% 配列的无线机械键盘。",
        },
        {
            "product_id": "aurora-keyboard",
            "name": "极光键盘",
            "aliases": ["Aurora Keyboard"],
            "keywords": ["键盘", "机械", "有线"],
            "price_cny": 799,
            "shipping_cny": 0,
            "stock": 8,
            "description": "全尺寸机械键盘。",
        },
    ]


def test_retrieval_returns_price_and_shipping_for_exact_alias():
    """Catches a lookup that ignores the required alias or omits product facts."""
    result = KnowledgeRetrievalTool(DATA_PATH).search("星云键盘")

    assert result.status == "success"
    assert result.error_code is None
    assert result.data["match_type"] == "exact_alias"
    assert result.data["product_id"] == "nebula-keyboard"
    assert result.data["price_cny"] >= 0
    assert result.data["shipping_cny"] >= 0


def test_retrieval_exposes_a_langchain_structured_tool():
    """Catches Day 29 receiving a wrapper that cannot invoke the fixed database."""
    result = KnowledgeRetrievalTool(DATA_PATH).as_langchain_tool().invoke(
        {"query": "星云键盘"}
    )

    assert result["status"] == "success"
    assert result["data"]["product_id"] == "nebula-keyboard"
    assert result["error_code"] is None


def test_retrieval_returns_not_found_without_inventing_a_product():
    """Catches a fallback that fabricates an answer for an unknown query."""
    result = KnowledgeRetrievalTool(DATA_PATH).search("不存在的产品")

    assert result.status == "not_found"
    assert result.error_code == "not_found"
    assert result.data == {"query": "不存在的产品"}


def test_retrieval_result_mutation_cannot_change_future_searches():
    """Catches outward aliases sharing the frozen tool's internal product list."""
    tool = KnowledgeRetrievalTool(DATA_PATH)
    result = tool.search("星云键盘")
    result.data["aliases"].append("injected alias")

    assert tool.search("injected alias").status == "not_found"
    assert "injected alias" not in tool.search("星云键盘").data["aliases"]


def test_retrieval_returns_sorted_candidates_for_keyword_tie(tmp_path: Path):
    """Catches a ranking rule that guesses one product on equal keyword scores."""
    path = tmp_path / "knowledge_base.json"
    write_knowledge_base(path, fixture_products())

    result = KnowledgeRetrievalTool(path).search("键盘")

    assert result.status == "ambiguous"
    assert result.error_code == "ambiguous_match"
    assert result.data["match_type"] == "keyword"
    assert [candidate["product_id"] for candidate in result.data["candidates"]] == [
        "aurora-keyboard",
        "nebula-keyboard",
    ]


@pytest.mark.parametrize(
    ("mutate", "expected_error"),
    [
        (
            lambda products: products.append({**products[0], "product_id": "nebula-keyboard"}),
            "duplicate_product_id",
        ),
        (lambda products: products[0].update(price_cny=-1), "invalid_price_cny"),
    ],
)
def test_retrieval_rejects_invalid_product_schema(
    tmp_path: Path, mutate, expected_error: str
):
    """Catches invalid inventory data being accepted at tool initialization."""
    path = tmp_path / "knowledge_base.json"
    products = fixture_products()
    mutate(products)
    write_knowledge_base(path, products)

    with pytest.raises(ValueError, match=expected_error):
        KnowledgeRetrievalTool(path)


def test_retrieval_rejects_a_knowledge_base_with_wrong_sha(tmp_path: Path):
    """Catches use of a knowledge base whose products changed after it was frozen."""
    path = tmp_path / "knowledge_base.json"
    payload = make_knowledge_base(fixture_products())
    payload["products_sha256"] = "0" * 64
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="knowledge_base_sha256_mismatch"):
        KnowledgeRetrievalTool(path)
