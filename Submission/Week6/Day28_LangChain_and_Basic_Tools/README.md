# Day 28：LangChain 与两个基础工具

## 交付结论

已实现并验证两个 LangChain 工具类：

- `CalculatorTool`：使用 AST 白名单解释算术表达式，不调用 `eval`；
- `KnowledgeRetrievalTool`：检索固定本地 JSON 知识库，不联网、不编造结果；
- Day 28 独立验证结果为 `PASS`；
- 18 项 Day 28 测试通过。

## 文件说明

| 目录/文件 | 内容 |
|---|---|
| `Code/week6_agent/tools/` | 老师要求的两个工具类 |
| `Code/week6_agent/limits.py` | 输入和资源限制 |
| `Code/week6_agent/tool_result.py` | 统一结构化返回 |
| `Data/knowledge_base.json` | 固定本地知识库 |
| `Results/day28_validation.json` | 验证结果（PASS） |
| `Results/environment.json` | 实际环境记录 |
| `Submission_Map.json` | 扁平文件与正式实验源文件的哈希映射 |

依赖版本见 [`requirements.txt`](requirements.txt)。提交中的代码与正式实验源码保持
字节一致，映射关系可通过 [`Submission_Map.json`](Submission_Map.json) 核验。

从本目录进行最小调用时，应显式传入扁平目录中的知识库路径：

```bash
PYTHONPATH=Code python - <<'PY'
from pathlib import Path
from week6_agent.tools.calculator import CalculatorTool
from week6_agent.tools.knowledge_retrieval import KnowledgeRetrievalTool

assert CalculatorTool().run("123 * 456").data["value"] == 56088
retriever = KnowledgeRetrievalTool(Path("Data/knowledge_base.json"))
PY
```

项目内部的完整测试和验证器保留在正式工程目录中，不重复放入教师提交。
