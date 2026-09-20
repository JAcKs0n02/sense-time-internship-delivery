# Week6 最终 Agent 模型引用归档

最终组合：

- 基座：已核验的 Week4 DPO merged model；
- 增量权重：Day31 `checkpoint-38` Tool-call SFT LoRA；
- System Prompt：main；
- 工具：Calculator、KnowledgeRetrieval、AST-only CodeExecutor；
- 推理协议：确定性生成，`max_new_tokens=384`，`seed=42`，最大 6 步。

checkpoint-38 位于：

`/root/autodl-tmp/qwen25-week6/formal-v2/runs/tool-sft-lora-v2/checkpoint-38`

权重 SHA-256：`a5e348ad180f8183479d3138b40613ead7163f3b2089c9b271063b64be5cdc5c`。

本目录和教师提交只保存引用 manifest、adapter 配置与哈希，不复制大权重。Prompt 消融的两组结果均保留，最终采用开发集严格成功率更高的 main Prompt。
