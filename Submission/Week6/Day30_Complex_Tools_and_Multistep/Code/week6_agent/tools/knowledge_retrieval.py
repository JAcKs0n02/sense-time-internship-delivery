"""Deterministic retrieval over the fixed, local Day 28 product database."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from week6_agent.tool_result import ToolResult


class KnowledgeRetrievalInput(BaseModel):
    """Input accepted by the LangChain-compatible retrieval wrapper."""

    query: str = Field(description="A product name, alias, or product keyword query.")


class KnowledgeRetrievalTool:
    """Search a constructor-fixed JSON file without path or network inputs."""

    _REQUIRED_FIELDS = {
        "product_id",
        "name",
        "aliases",
        "keywords",
        "price_cny",
        "shipping_cny",
        "stock",
        "description",
    }
    _DEFAULT_KNOWLEDGE_BASE = (
        Path(__file__).resolve().parents[3] / "day28" / "source" / "data" / "knowledge_base.json"
    )

    def __init__(self, knowledge_base_path: str | Path | None = None) -> None:
        self.knowledge_base_path = Path(knowledge_base_path or self._DEFAULT_KNOWLEDGE_BASE)
        self._products, self.knowledge_base_sha256 = self._load_and_validate()

    def search(self, query: str) -> ToolResult:
        """Search names, aliases, then keyword sets in a deterministic order."""
        if not isinstance(query, str) or not query.strip():
            return ToolResult("rejected", error_code="invalid_query")
        normalized_query = self._normalize(query)

        exact_matches: list[tuple[dict[str, Any], str]] = []
        for product in self._products:
            if normalized_query == self._normalize(product["name"]):
                exact_matches.append((product, "exact_name"))
            elif any(
                normalized_query == self._normalize(alias) for alias in product["aliases"]
            ):
                exact_matches.append((product, "exact_alias"))
        if exact_matches:
            return self._resolve_matches(query, exact_matches)

        query_terms = tuple(term for term in normalized_query.split(" ") if term)
        scored: list[tuple[dict[str, Any], int]] = []
        for product in self._products:
            product_keywords = {self._normalize(keyword) for keyword in product["keywords"]}
            score = sum(term in product_keywords for term in query_terms)
            if score:
                scored.append((product, score))
        if not scored:
            return ToolResult("not_found", data={"query": query}, error_code="not_found")

        best_score = max(score for _, score in scored)
        best = [(product, "keyword") for product, score in scored if score == best_score]
        return self._resolve_matches(query, best)

    def as_langchain_tool(self) -> StructuredTool:
        """Return the stable wrapper Day 29 binds to its ReAct agent."""
        return StructuredTool.from_function(
            func=self._invoke_for_langchain,
            name="knowledge_retrieval",
            description="Looks up product price, shipping, stock, and descriptions in local JSON.",
            args_schema=KnowledgeRetrievalInput,
        )

    def _invoke_for_langchain(self, query: str) -> dict[str, Any]:
        return self.search(query).to_dict()

    def _load_and_validate(self) -> tuple[list[dict[str, Any]], str]:
        try:
            payload = json.loads(self.knowledge_base_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("invalid_knowledge_base") from error
        if not isinstance(payload, dict) or payload.get("schema_version") != "1.0":
            raise ValueError("invalid_schema_version")
        products = payload.get("products")
        if not isinstance(products, list):
            raise ValueError("invalid_products")
        declared_sha = payload.get("products_sha256")
        if not isinstance(declared_sha, str) or len(declared_sha) != 64:
            raise ValueError("invalid_knowledge_base_sha256")
        actual_sha = self._products_sha256(products)
        if declared_sha != actual_sha:
            raise ValueError("knowledge_base_sha256_mismatch")
        self._validate_products(products)
        return products, actual_sha

    def _validate_products(self, products: list[object]) -> None:
        product_ids: set[str] = set()
        for product in products:
            if not isinstance(product, dict) or set(product) != self._REQUIRED_FIELDS:
                raise ValueError("invalid_product_schema")
            product_id = product["product_id"]
            if not isinstance(product_id, str) or not product_id:
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

    def _resolve_matches(
        self, query: str, matches: list[tuple[dict[str, Any], str]]
    ) -> ToolResult:
        matches.sort(key=lambda item: item[0]["product_id"])
        if len(matches) == 1:
            product, match_type = matches[0]
            data = self._public_product(product)
            data.update(match_type=match_type, knowledge_base_sha256=self.knowledge_base_sha256)
            return ToolResult("success", data=data)
        return ToolResult(
            "ambiguous",
            data={
                "query": query,
                "match_type": matches[0][1],
                "candidates": [self._public_product(product) for product, _ in matches],
            },
            error_code="ambiguous_match",
        )

    @classmethod
    def _products_sha256(cls, products: list[object]) -> str:
        encoded = json.dumps(
            products, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(value.casefold().split())

    @staticmethod
    def _public_product(product: dict[str, Any]) -> dict[str, Any]:
        return {
            field: list(product[field]) if field == "aliases" else product[field]
            for field in (
                "product_id",
                "name",
                "aliases",
                "price_cny",
                "shipping_cny",
                "stock",
                "description",
            )
        }
