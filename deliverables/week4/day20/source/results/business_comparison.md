# Day 20 Blinded Business Comparison

评审者类型：`codex_review`。候选 A/B 在 `seed=42` 下生成，评分完成并冻结 SHA-256 后才使用私有 mapping 解盲。

| Prompt | SFT-only | SFT+DPO | Outcome |
|---|---:|---:|---|
| `w4-eval-business-01` | 3.275 | 3.400 | SFT+DPO |
| `w4-eval-business-02` | 3.800 | 3.425 | SFT-only |
| `w4-eval-business-03` | 3.325 | 2.625 | SFT-only |
| `w4-eval-business-04` | 3.375 | 3.275 | SFT-only |
| `w4-eval-business-05` | 4.050 | 4.050 | Tie |
| **Mean** | **3.565** | **3.355** | **SFT-only +0.210** |

五题只提供描述性证据。逐候选五维分数和理由见 [`business_comparison.csv`](business_comparison.csv)，未改写的模型回答见两份原始 JSONL。
