#!/usr/bin/env python3
"""Build the deterministic Day31 ShareGPT tool-call train and frozen eval sets.

The only source of an observation is ``execute_tool`` below, which calls the
approved Day28-Day30 tool implementations.  The audit representation is kept
alongside (but separate from) the LLaMA-Factory ``conversations`` projection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


DAY31_SOURCE = Path(__file__).resolve().parents[1]
REPO_ROOT = DAY31_SOURCE.parents[3]
WEEK6_SOURCE = REPO_ROOT / "deliverables/week6/source"
KNOWLEDGE_BASE = REPO_ROOT / "deliverables/week6/day28/source/data/knowledge_base.json"
KNOWLEDGE_BASE_V2 = DAY31_SOURCE / "data/knowledge_base_v2.json"
TOOL_SOURCE_PATHS = {
    "calculator": WEEK6_SOURCE / "week6_agent/tools/calculator.py",
    "knowledge_retrieval": WEEK6_SOURCE / "week6_agent/tools/knowledge_retrieval.py",
    "code_executor": WEEK6_SOURCE / "week6_agent/tools/code_executor.py",
}
EXPECTED_TRAIN_COUNTS = {
    "calculator": 20,
    "knowledge_retrieval": 20,
    "code_executor": 20,
    "knowledge_plus_calculator": 25,
    "recovery_or_termination": 15,
}
sys.path.insert(0, str(WEEK6_SOURCE))


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tools(knowledge_base_path: Path = KNOWLEDGE_BASE) -> dict[str, Any]:
    from week6_agent.tools.calculator import CalculatorTool
    from week6_agent.tools.code_executor import CodeExecutor
    from week6_agent.tools.knowledge_retrieval import KnowledgeRetrievalTool

    return {
        "calculator": CalculatorTool(),
        "knowledge_retrieval": KnowledgeRetrievalTool(knowledge_base_path),
        "code_executor": CodeExecutor(),
    }


def execute_tool(
    name: str,
    arguments: dict[str, Any],
    knowledge_base_path: Path = KNOWLEDGE_BASE,
) -> dict[str, Any]:
    """Call a frozen local tool and return its unedited JSON object."""
    tool = _tools(knowledge_base_path).get(name)
    if tool is None:
        raise ValueError(f"unregistered tool: {name}")
    if name == "calculator":
        result = tool.run(**arguments)
    elif name == "knowledge_retrieval":
        result = tool.search(**arguments)
    else:
        result = tool.inspect(**arguments)
    return result.to_dict()


def tool_definitions() -> list[dict[str, Any]]:
    """Return the official function-tool projection with schemas from real tools."""
    definitions: list[dict[str, Any]] = []
    for name, tool in _tools().items():
        langchain_tool = tool.as_langchain_tool()
        definitions.append(
            {
                "name": langchain_tool.name,
                "description": langchain_tool.description,
                "parameters": langchain_tool.args_schema.model_json_schema(),
            }
        )
    return definitions


def _call_value(name: str, arguments: dict[str, Any]) -> str:
    return canonical_json({"name": name, "arguments": arguments})


def _final_for(sequence: list[tuple[str, dict[str, Any]]], observations: list[dict[str, Any]], final: str) -> str:
    """Avoid accepting an ungrounded template final response at build time."""
    if not sequence or len(sequence) != len(observations) or not final.strip():
        raise ValueError("incomplete grounded example")
    return final


def make_row(
    *,
    row_id: str,
    split: str,
    category: str,
    prompt: str,
    sequence: list[tuple[str, dict[str, Any], str]],
    final: str,
    success_predicates: dict[str, Any] | None = None,
    knowledge_base_path: Path = KNOWLEDGE_BASE,
) -> dict[str, Any]:
    """Create one audit row and its one-to-one ShareGPT conversation view."""
    conversations: list[dict[str, str]] = [{"from": "human", "value": prompt}]
    audit_react: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    for name, arguments, thought in sequence:
        observation = execute_tool(name, arguments, knowledge_base_path)
        observations.append(observation)
        conversations.extend(
            [
                {"from": "function_call", "value": _call_value(name, arguments)},
                {"from": "observation", "value": canonical_json(observation)},
            ]
        )
        audit_react.append(
            {
                "Thought": thought,
                "Action": name,
                "Action Input": arguments,
                "Observation": observation,
            }
        )
    conversations.append(
        {
            "from": "gpt",
            "value": _final_for([(name, args) for name, args, _ in sequence], observations, final),
        }
    )
    row: dict[str, Any] = {
        "id": row_id,
        "split": split,
        "category": category,
        "conversations": conversations,
        "tools": canonical_json(tool_definitions()),
        "audit": {"prompt": prompt, "react": audit_react, "final": final},
    }
    if success_predicates is not None:
        row["success_predicates"] = success_predicates
    return row


def _calculator_rows() -> list[dict[str, Any]]:
    expressions = [
        "7 + 8", "12 * 9", "144 / 12", "(18 - 5) * 3", "2 ** 8",
        "99 % 7", "-15 + 22", "81 // 9", "3.5 * 4", "100 - 37",
        "(6 + 4) * (9 - 2)", "250 / 5", "17 * 6 - 4", "72 + 19",
        "5 ** 3", "48 // 5", "11 * 11", "90 / 4", "34 + 66", "63 - 28",
    ]
    rows = []
    for index, expression in enumerate(expressions, 1):
        result = execute_tool("calculator", {"expression": expression})["data"]["value"]
        rows.append(make_row(
            row_id=f"train-calculator-{index:02d}", split="train", category="calculator",
            prompt=f"训练算术任务 {index:02d}：请用计算器计算表达式 {expression}。",
            sequence=[("calculator", {"expression": expression}, "这是纯算术，应调用 calculator。")],
            final=f"计算结果是 {result}。",
        ))
    return rows


def _knowledge_rows() -> list[dict[str, Any]]:
    queries = [
        "星云机械键盘", "星云键盘", "Nebula Keyboard", "键盘 无线", "无线",
        "彗星鼠标", "Comet Mouse", "无线鼠标", "鼠标", "办公",
        "轨道耳机", "Orbit Headphones", "降噪耳机", "耳机", "无线 降噪",
        "不存在的设备", "火星显示器", "木星相机", "银杏显示器", "海盐路由器",
    ]
    rows = []
    for index, query in enumerate(queries, 1):
        observation = execute_tool("knowledge_retrieval", {"query": query})
        if observation["status"] == "success":
            final = f"查询到 {observation['data']['name']}，价格为 {observation['data']['price_cny']} 元。"
        elif observation["status"] == "ambiguous":
            final = "查询结果不唯一，请提供更具体的产品名称。"
        else:
            final = f"本地目录中没有找到“{query}”。"
        rows.append(make_row(
            row_id=f"train-knowledge-{index:02d}", split="train", category="knowledge_retrieval",
            prompt=f"训练目录检索 {index:02d}：请在本地商品目录中查找“{query}”。",
            sequence=[("knowledge_retrieval", {"query": query}, "需要固定目录事实，应调用 knowledge_retrieval。")],
            final=final,
        ))
    return rows


def _code_rows() -> list[dict[str, Any]]:
    sources = [
        "count = 3\nnext_count = count + 1", "for item in [1, 2]:\n    total = item",
        "if True:\n    label = 'ok'", "values = [1, 2, 3]", "result = {'safe': True}",
        "def square(value):\n    return value * value", "class Marker:\n    pass", "flag = 5 > 2",
        "text = 'offline review'", "total = 42", "open('x.txt', 'w')", "import os",
        "object.value = 3", "with open('x') as handle:\n    text = handle.read()", "broken =",
        "print('hello')", "from math import sqrt", "global shared", "async with lock:\n    pass", "data.append(1)",
    ]
    rows = []
    for index, source in enumerate(sources, 1):
        observation = execute_tool("code_executor", {"source": source})
        final = (
            "代码语法有效且未发现受限风险节点。" if observation["status"] == "success"
            else f"代码未执行；检查结果为 {observation['status']}（{observation['error_code']}）。"
        )
        rows.append(make_row(
            row_id=f"train-code-{index:02d}", split="train", category="code_executor",
            prompt=f"训练代码审查 {index:02d}：仅检查以下 Python 源码，不要执行：\n{source}",
            sequence=[("code_executor", {"source": source}, "用户要求静态检查，调用不会执行代码的 code_executor。")],
            final=final,
        ))
    return rows


def _product_row(row_id: str, split: str, category: str, query: str, budget: int, prompt: str, predicates: dict[str, Any] | None = None) -> dict[str, Any]:
    retrieved = execute_tool("knowledge_retrieval", {"query": query})
    if retrieved["status"] != "success":
        raise ValueError(f"product prompt must resolve uniquely: {query}")
    data = retrieved["data"]
    expression = f"{data['price_cny']} + {data['shipping_cny']}"
    calculated = execute_tool("calculator", {"expression": expression})
    total = calculated["data"]["value"]
    difference = abs(total - budget)
    relation = "超过" if total > budget else "没有超过"
    final = f"{data['name']}：商品 {data['price_cny']} 元，运费 {data['shipping_cny']} 元，总价 {total} 元，{relation} {budget} 元预算，相差 {difference} 元。"
    return make_row(
        row_id=row_id, split=split, category=category, prompt=prompt,
        sequence=[
            ("knowledge_retrieval", {"query": query}, "先读取商品价格和运费。"),
            ("calculator", {"expression": expression}, "已获得两个金额，必须计算总价。"),
        ],
        final=final, success_predicates=predicates,
    )


def _multistep_rows() -> list[dict[str, Any]]:
    # Aurora is intentionally reserved for the frozen evaluation product tasks.
    products = ["星云机械键盘", "彗星鼠标", "轨道耳机"]
    budgets = [650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850]
    return [
        _product_row(
            f"train-multistep-{index:02d}", "train", "knowledge_plus_calculator", query, budget,
            f"训练预算核对 {index:02d}：查“{query}”的商品价和配送费，算出到手价后和 {budget} 元额度比较。",
        )
        for index, (query, budget) in enumerate(zip((products * 9)[:25], budgets), 1)
    ]


def _recovery_rows() -> list[dict[str, Any]]:
    cases: list[tuple[str, list[tuple[str, dict[str, Any], str]], str]] = []
    for index, expression in enumerate(["1 / 0", "2 +", "abs(4)", "10 ** 999", "'x' + 1"], 1):
        cases.append((
            f"训练恢复算术 {index:02d}：请计算 {expression}；如果工具拒绝，解释原因并停止。",
            [("calculator", {"expression": expression}, "先执行请求；若被拒绝，不要猜测结果或循环调用。")],
            "计算器拒绝了该表达式，因此不能提供一个编造的数值。",
        ))
    for index, query in enumerate(["木星平板", "银杏显示器", "纸飞机扫描仪", "海盐路由器", "蓝鲸投影仪"], 1):
        cases.append((
            f"训练恢复目录 {index:02d}：在固定商品目录搜索“{query}”；若无结果，直接说明并终止。",
            [("knowledge_retrieval", {"query": query}, "先查固定目录；查无结果时安全结束。")],
            f"本地商品目录没有“{query}”，不会改用不相关工具猜测。",
        ))
    for index, source in enumerate(["open('a')", "import sys", "item.value", "for", "exec('x')"], 1):
        cases.append((
            f"训练恢复代码 {index:02d}：只审查这段 Python 源码，拒绝时解释并不要检索商品：\n{source}",
            [("code_executor", {"source": source}, "这是静态代码检查；结果拒绝时不能追加无关检索。")],
            "代码没有执行；工具的拒绝结果已足以安全终止本次检查。",
        ))
    rows = []
    for index, (prompt, sequence, final) in enumerate(cases, 1):
        rows.append(make_row(
            row_id=f"train-recovery-{index:02d}", split="train", category="recovery_or_termination",
            prompt=prompt, sequence=sequence, final=final,
        ))
    return rows


def build_train_rows() -> list[dict[str, Any]]:
    rows = _calculator_rows() + _knowledge_rows() + _code_rows() + _multistep_rows() + _recovery_rows()
    if Counter(row["category"] for row in rows) != EXPECTED_TRAIN_COUNTS:
        raise ValueError("train category construction mismatch")
    return rows


def _eval_predicates(sequence: list[str], **extra: Any) -> dict[str, Any]:
    return {"expected_tool_sequence": sequence, "tool_selection_exact": True, "arguments_schema_valid": True, **extra}


def build_eval_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expressions = ["13 * 7", "256 / 8", "(14 + 6) * 5", "77 - 29", "9 ** 3", "101 % 9", "-8 * 11", "64 // 7", "45 + 55", "7.25 * 4"]
    for index, expression in enumerate(expressions, 1):
        value = execute_tool("calculator", {"expression": expression})["data"]["value"]
        rows.append(make_row(
            row_id=f"eval-calculator-{index:02d}", split="eval_frozen", category="calculator",
            prompt=f"冻结评测算式 {index:02d}：仅返回算式 {expression} 的计算结果。",
            sequence=[("calculator", {"expression": expression}, "该请求只需受限算术工具。")], final=f"结果为 {value}。",
            success_predicates=_eval_predicates(["calculator"]),
        ))
    # Keep every frozen product case on Aurora. Training never retrieves its
    # product identity, aliases, or calculator total, so this is independent at
    # the knowledge-case level while still using real fixed-knowledge-base data.
    product_cases = [
        ("极光键盘", 800), ("Aurora Keyboard", 805), ("极光机械键盘", 810),
        ("有线", 820), ("键盘 有线", 830), ("机械 有线", 840),
        ("有线 键盘", 850), ("有线 机械 键盘", 860),
    ]
    for index, (query, budget) in enumerate(product_cases, 1):
        extra = {"budget_cny": budget}
        if query == "极光键盘":
            extra["requires_calculator_when_shipping_zero"] = True
            row_id = "eval-aurora-zero-shipping-calculator-required"
        else:
            row_id = f"eval-budget-{index:02d}"
        rows.append(_product_row(
            row_id, "eval_frozen", "knowledge_plus_calculator", query, budget,
            f"冻结购物评测 {index:02d}：核对“{query}”的落地总额是否高于 {budget} 元；必须先查目录再计算。",
            _eval_predicates(["knowledge_retrieval", "calculator"], **extra),
        ))
    code_sources = [
        "total = 1 + 2", "open('/tmp/not-run')", "value =", "import pathlib", "record.field = 2", "for n in range(2):\n    pass",
    ]
    for index, source in enumerate(code_sources, 1):
        observation = execute_tool("code_executor", {"source": source})
        rows.append(make_row(
            row_id="eval-code-no-unnecessary-knowledge" if index == 1 else f"eval-code-{index:02d}",
            split="eval_frozen", category="code_executor",
            prompt=f"冻结代码评测 {index:02d}：只做 AST 安全审查，绝不执行，也不要查商品目录。\n{source}",
            sequence=[("code_executor", {"source": source}, "代码检查已足够，不应追加 knowledge_retrieval。")],
            final=f"静态检查状态：{observation['status']}。",
            success_predicates=_eval_predicates(["code_executor"], no_knowledge_retrieval_after_code_check=True),
        ))
    recovery_cases = [
        ("calculator", {"expression": "0 / 0"}, "冻结恢复算术：表达式被拒绝时停止，不要重试同一调用。"),
        ("calculator", {"expression": "lambda x: x"}, "冻结恢复算术：遇到禁止节点时准确报告工具拒绝。"),
        ("knowledge_retrieval", {"query": "雾港摄像机"}, "冻结恢复目录：没有此商品时结束，不替换成相似商品。"),
        ("knowledge_retrieval", {"query": "松风音箱"}, "冻结恢复目录：目录未命中时不要编造库存。"),
        ("code_executor", {"source": "compile('x', '<frozen>', 'exec')"}, "冻结恢复代码：风险节点被拒绝时不得调用商品工具。"),
        ("code_executor", {"source": "if"}, "冻结恢复代码：语法错误时安全终止。"),
    ]
    for index, (name, arguments, prompt) in enumerate(recovery_cases, 1):
        observation = execute_tool(name, arguments)
        rows.append(make_row(
            row_id=f"eval-recovery-{index:02d}", split="eval_frozen", category="recovery_or_termination",
            prompt=prompt, sequence=[(name, arguments, "执行一次真实工具调用；拒绝后停止。")],
            final=f"工具状态为 {observation['status']}，本次请求安全结束。",
            success_predicates=_eval_predicates([name], terminates_after_tool_error=True),
        ))
    if len(rows) != 30:
        raise ValueError("eval construction mismatch")
    return rows


V2_PROMPT_TEMPLATES = {
    "calculator": (
        "请调用计算器求出 {payload} 的结果。",
        "需要精确计算表达式 {payload}，请使用算术工具。",
        "不要心算：用 calculator 处理 {payload}。",
        "帮我核算 {payload}，返回工具给出的数值。",
    ),
    "knowledge": (
        "请在本地商品目录查询“{payload}”。",
        "我需要“{payload}”的目录信息，请使用检索工具。",
        "帮我查一下固定商品库中的“{payload}”。",
        "请核对产品“{payload}”的价格与库存记录。",
    ),
    "code": (
        "只做 Python AST 静态检查，不执行下面源码：\n{payload}",
        "请用 CodeExecutor 审查这段源码，禁止运行：\n{payload}",
        "核对以下 Python 代码的语法和风险节点：\n{payload}",
        "对下面内容进行非执行式 AST 检查：\n{payload}",
    ),
    "budget": (
        "查询“{payload}”的商品价和运费，用计算器求总价，并与 {budget} 元预算比较。",
        "请先检索“{payload}”，再计算落地价，判断是否超过 {budget} 元。",
        "核对“{payload}”到手总额：目录检索后必须调用 calculator，并对照 {budget} 元预算。",
        "我有 {budget} 元预算；请查“{payload}”并计算商品价加运费后的差额。",
    ),
}


def _v2_prompt(kind: str, index: int, payload: str, budget: int | None = None) -> str:
    template = V2_PROMPT_TEMPLATES[kind][index % len(V2_PROMPT_TEMPLATES[kind])]
    return template.format(payload=payload, budget=budget)


def _v2_products() -> list[dict[str, Any]]:
    payload = json.loads(KNOWLEDGE_BASE_V2.read_text(encoding="utf-8"))
    return payload["products"]


def _v2_calculator_rows(prefix: str, split: str, count: int, offset: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(count):
        left = 17 + (offset + index) * 3
        right = 2 + (offset + index) % 9
        factor = 2 + (offset + index) % 5
        expression = f"({left} + {right}) * {factor}"
        observation = execute_tool("calculator", {"expression": expression})
        value = observation["data"]["value"]
        rows.append(make_row(
            row_id=f"{prefix}-calculator-{index + 1:03d}",
            split=split,
            category="calculator",
            prompt=_v2_prompt("calculator", index, expression),
            sequence=[("calculator", {"expression": expression}, "使用受限计算器获得精确数值。")],
            final=f"计算器返回 {value}。",
            success_predicates={
                "expected_tool_sequence": ["calculator"],
                "expected_answer": {"kind": "calculator", "value": value},
            },
        ))
    return rows


def _v2_knowledge_rows(
    prefix: str,
    split: str,
    count: int,
    products: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(count):
        product = products[index % len(products)]
        query = product["name"] if index % 3 == 0 else product["aliases"][index % len(product["aliases"])]
        observation = execute_tool(
            "knowledge_retrieval", {"query": query}, KNOWLEDGE_BASE_V2
        )
        data = observation["data"]
        rows.append(make_row(
            row_id=f"{prefix}-knowledge-{index + 1:03d}",
            split=split,
            category="knowledge_retrieval",
            prompt=_v2_prompt("knowledge", index, query),
            sequence=[("knowledge_retrieval", {"query": query}, "固定目录事实必须由检索工具提供。")],
            final=f"目录记录：{data['name']}，价格 {data['price_cny']} 元，库存 {data['stock']}。",
            success_predicates={
                "expected_tool_sequence": ["knowledge_retrieval"],
                "expected_answer": {
                    "kind": "knowledge",
                    "status": "success",
                    "product_id": data["product_id"],
                    "name": data["name"],
                    "price_cny": data["price_cny"],
                    "stock": data["stock"],
                },
            },
            knowledge_base_path=KNOWLEDGE_BASE_V2,
        ))
    return rows


def _v2_source(index: int, offset: int) -> str:
    value = index + offset
    patterns = (
        f"total_{value} = {value} + 2\nlabel_{value} = 'ok'",
        f"def scale_{value}(number):\n    return number * {value % 7 + 1}",
        f"items_{value} = [{value}, {value + 1}]\nsize_{value} = len(items_{value})",
        f"value_{value} =",
        f"import os\nmarker_{value} = 'static-only'",
        f"record_{value}.field = {value}",
    )
    return patterns[index % len(patterns)]


def _v2_code_rows(prefix: str, split: str, count: int, offset: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(count):
        source = _v2_source(index, offset)
        observation = execute_tool("code_executor", {"source": source})
        data = observation["data"]
        rows.append(make_row(
            row_id=f"{prefix}-code-{index + 1:03d}",
            split=split,
            category="code_executor",
            prompt=_v2_prompt("code", index, source),
            sequence=[("code_executor", {"source": source}, "源码只能进行 AST 静态检查。")],
            final=(
                f"静态检查状态为 {observation['status']}，语法有效性为 "
                f"{data['syntax_valid']}，错误码为 {observation['error_code']}，"
                f"风险节点为 {data['risk_nodes'] if data['risk_nodes'] else '[]'}。"
            ),
            success_predicates={
                "expected_tool_sequence": ["code_executor"],
                "expected_answer": {
                    "kind": "code",
                    "status": observation["status"],
                    "error_code": observation["error_code"],
                    "syntax_valid": data["syntax_valid"],
                    "risk_nodes": data["risk_nodes"],
                },
            },
        ))
    return rows


def _v2_budget_rows(
    prefix: str,
    split: str,
    count: int,
    products: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    budget_offsets = (-120, 0, 85, 160)
    for index in range(count):
        product = products[index % len(products)]
        query = product["name"] if index % 2 == 0 else product["aliases"][0]
        retrieved = execute_tool(
            "knowledge_retrieval", {"query": query}, KNOWLEDGE_BASE_V2
        )
        data = retrieved["data"]
        expression = f"{data['price_cny']} + {data['shipping_cny']}"
        calculated = execute_tool("calculator", {"expression": expression})
        total = calculated["data"]["value"]
        budget = int(total + budget_offsets[index % len(budget_offsets)])
        relation = "above" if total > budget else "equal" if total == budget else "below"
        difference = abs(total - budget)
        rows.append(make_row(
            row_id=f"{prefix}-budget-{index + 1:03d}",
            split=split,
            category="knowledge_plus_calculator",
            prompt=_v2_prompt("budget", index, query, budget),
            sequence=[
                ("knowledge_retrieval", {"query": query}, "先读取商品价和运费。"),
                ("calculator", {"expression": expression}, "使用检索结果中的两个金额计算总价。"),
            ],
            final=(
                f"{data['name']}商品价 {data['price_cny']} 元，运费 {data['shipping_cny']} 元，"
                f"总价 {total} 元，与 {budget} 元预算关系为 {relation}，差额 {difference} 元。"
            ),
            success_predicates={
                "expected_tool_sequence": ["knowledge_retrieval", "calculator"],
                "expected_answer": {
                    "kind": "budget",
                    "product_id": data["product_id"],
                    "name": data["name"],
                    "price_cny": data["price_cny"],
                    "shipping_cny": data["shipping_cny"],
                    "total_cny": total,
                    "budget_cny": budget,
                    "relation": relation,
                    "difference_cny": difference,
                },
            },
            knowledge_base_path=KNOWLEDGE_BASE_V2,
        ))
    return rows


def _v2_recovery_rows(prefix: str, split: str, count: int, offset: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(count):
        kind = index % 3
        if kind == 0:
            payload = f"{offset + index + 1} / 0"
            name = "calculator"
            arguments = {"expression": payload}
            prompt = f"请用计算器处理 {payload}；若被拒绝，报告错误码并停止。"
            thought = "执行一次真实计算器调用，拒绝后终止。"
        elif kind == 1:
            payload = f"未收录设备-{prefix}-{offset + index + 1}"
            name = "knowledge_retrieval"
            arguments = {"query": payload}
            prompt = f"在本地目录检索“{payload}”；若未找到，报告状态并停止。"
            thought = "执行一次真实目录检索，未命中后终止。"
        else:
            payload = f"import os\nflag_{offset + index + 1} = 'audit'"
            name = "code_executor"
            arguments = {"source": payload}
            prompt = f"只做 AST 静态检查，不执行下面源码；若拒绝，报告错误码并停止：\n{payload}"
            thought = "执行一次真实 AST 检查，拒绝后终止。"
        observation = execute_tool(name, arguments, KNOWLEDGE_BASE_V2)
        rows.append(make_row(
            row_id=f"{prefix}-recovery-{index + 1:03d}",
            split=split,
            category="recovery_or_termination",
            prompt=prompt,
            sequence=[(name, arguments, thought)],
            final=(
                f"工具状态为 {observation['status']}，错误码为 "
                f"{observation['error_code']}，任务已终止。"
            ),
            success_predicates={
                "expected_tool_sequence": [name],
                "terminates_after_tool_error": True,
                "expected_answer": {
                    "kind": "recovery",
                    "status": observation["status"],
                    "error_code": observation["error_code"],
                },
            },
            knowledge_base_path=KNOWLEDGE_BASE_V2,
        ))
    return rows


def _v2_split(
    prefix: str,
    split: str,
    products: list[dict[str, Any]],
    counts: dict[str, int],
    offset: int,
) -> list[dict[str, Any]]:
    return [
        *_v2_calculator_rows(prefix, split, counts["calculator"], offset),
        *_v2_knowledge_rows(prefix, split, counts["knowledge_retrieval"], products),
        *_v2_code_rows(prefix, split, counts["code_executor"], offset),
        *_v2_budget_rows(prefix, split, counts["knowledge_plus_calculator"], products),
        *_v2_recovery_rows(prefix, split, counts["recovery_or_termination"], offset),
    ]


def build_dataset_bundle() -> dict[str, list[dict[str, Any]]]:
    """Build the official versioned train/dev/test data without hidden payloads."""
    products = _v2_products()
    train_products, dev_products, test_products = products[:12], products[12:18], products[18:24]
    core_counts = dict(EXPECTED_TRAIN_COUNTS)
    expanded_counts = {
        "calculator": 40,
        "knowledge_retrieval": 40,
        "code_executor": 40,
        "knowledge_plus_calculator": 50,
        "recovery_or_termination": 30,
    }
    dev_counts = {
        "calculator": 8,
        "knowledge_retrieval": 8,
        "code_executor": 8,
        "knowledge_plus_calculator": 10,
        "recovery_or_termination": 6,
    }
    test_counts = dict(EXPECTED_TRAIN_COUNTS)
    return {
        "train_core_100": _v2_split("core", "train_core", train_products, core_counts, 0),
        "train_expanded": _v2_split("expanded", "train_expanded", train_products, expanded_counts, 100),
        "dev_v2": _v2_split("dev", "dev", dev_products, dev_counts, 500),
        "test_v2": _v2_split("test", "test", test_products, test_counts, 900),
    }


def build_manifest(
    train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]], data_directory: Path | None = None
) -> dict[str, Any]:
    definitions = tool_definitions()
    sources = [
        {"name": name, "path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_path(path)}
        for name, path in TOOL_SOURCE_PATHS.items()
    ]
    schemas = {definition["name"]: definition["parameters"] for definition in definitions}
    return {
        "schema_version": "1.0",
        "generator": {"path": str(Path(__file__).relative_to(REPO_ROOT)), "sha256": sha256_path(Path(__file__))},
        "tool_names": [definition["name"] for definition in definitions],
        "tool_sources": sources,
        "tool_schemas_sha256": hashlib.sha256(canonical_json(schemas).encode()).hexdigest(),
        "knowledge_base": {"path": str(KNOWLEDGE_BASE.relative_to(REPO_ROOT)), "sha256": sha256_path(KNOWLEDGE_BASE)},
        "train": {
            "file_name": "tool_sft_train_100.json",
            "row_count": len(train_rows),
            "canonical_content_sha256": hashlib.sha256(canonical_json(train_rows).encode()).hexdigest(),
            "file_sha256": sha256_path(data_directory / "tool_sft_train_100.json") if data_directory else None,
            "category_counts": dict(sorted(Counter(row["category"] for row in train_rows).items())),
        },
        "eval_frozen": {
            "file_name": "tool_sft_eval_frozen.json",
            "row_count": len(eval_rows),
            "canonical_content_sha256": hashlib.sha256(canonical_json(eval_rows).encode()).hexdigest(),
            "file_sha256": sha256_path(data_directory / "tool_sft_eval_frozen.json") if data_directory else None,
            "frozen": True,
        },
        "llamafactory": {"formatting": "sharegpt", "message_column": "conversations", "tools_column": "tools", "tool_call_roles": ["human", "function_call", "observation", "gpt"], "formal_runtime_revision": "pending_autodl_preflight"},
    }


def write_dataset(output_directory: Path) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    train_rows, eval_rows = build_train_rows(), build_eval_rows()
    for name, payload in {
        "tool_sft_train_100.json": train_rows,
        "tool_sft_eval_frozen.json": eval_rows,
    }.items():
        (output_directory / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = build_manifest(train_rows, eval_rows, output_directory)
    (output_directory / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _v2_manifest_entry(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "file_name": path.name,
        "row_count": len(rows),
        "file_sha256": sha256_path(path),
        "canonical_content_sha256": hashlib.sha256(
            canonical_json(rows).encode("utf-8")
        ).hexdigest(),
        "category_counts": dict(
            sorted(Counter(row["category"] for row in rows).items())
        ),
    }


def write_dataset_bundle(output_directory: Path) -> dict[str, Any]:
    """Materialize the official v2 bundle and bind every split byte stream."""
    output_directory.mkdir(parents=True, exist_ok=True)
    bundle = build_dataset_bundle()
    train_v2 = [*bundle["train_core_100"], *bundle["train_expanded"]]
    files = {
        "train_core_100": ("tool_sft_train_core_100.json", bundle["train_core_100"]),
        "train_expanded": ("tool_sft_train_expanded.json", bundle["train_expanded"]),
        "train_v2": ("tool_sft_train_v2.json", train_v2),
        "dev_v2": ("tool_sft_dev_v2.json", bundle["dev_v2"]),
        "test_v2": ("tool_sft_test_v2.json", bundle["test_v2"]),
    }
    for file_name, rows in files.values():
        (output_directory / file_name).write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    manifest = {
        "schema_version": "2.0",
        "profile": "week6_agent_formal_v2",
        "generator": {
            "path": str(Path(__file__).relative_to(REPO_ROOT)),
            "sha256": sha256_path(Path(__file__)),
        },
        "knowledge_base": {
            "path": str(KNOWLEDGE_BASE_V2.relative_to(REPO_ROOT)),
            "sha256": sha256_path(KNOWLEDGE_BASE_V2),
            "products_sha256": json.loads(
                KNOWLEDGE_BASE_V2.read_text(encoding="utf-8")
            )["products_sha256"],
        },
        "tool_names": [definition["name"] for definition in tool_definitions()],
        "splits": {
            name: _v2_manifest_entry(output_directory / file_name, rows)
            for name, (file_name, rows) in files.items()
        },
        "selection_policy": {
            "dev_only_for_checkpoint_and_prompt_selection": True,
            "test_v2_reserved_for_final_evaluation": True,
            "raw_and_guarded_modes_reported_separately": True,
        },
    }
    (output_directory / "dataset_manifest_v2.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=DAY31_SOURCE / "data")
    parser.add_argument("--profile", choices=("legacy", "v2"), default="legacy")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.profile == "v2":
        manifest = write_dataset_bundle(args.output_directory)
        print(canonical_json({
            "status": "built",
            "profile": "v2",
            "train_rows": manifest["splits"]["train_v2"]["row_count"],
            "dev_rows": manifest["splits"]["dev_v2"]["row_count"],
            "test_rows": manifest["splits"]["test_v2"]["row_count"],
        }))
        return 0
    manifest = write_dataset(args.output_directory)
    print(canonical_json({"status": "built", "profile": "legacy", "train_rows": manifest["train"]["row_count"], "eval_rows": manifest["eval_frozen"]["row_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
