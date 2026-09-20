# Week8 第一阶段任务1–2验收记录

日期：2026-09-14。状态：输入与处理方法已核定；不是DATA_READY。

- `input_inventory.json`：11份数据版本/血缘文件、15份评估相关文件、5份tokenizer文件、13份原审核依赖的实际哈希和角色。
- `verification.json`：34项输入身份检查通过。远端JSONL检查解析后的完整内容及顺序，不要求不同空格排版的字节哈希一致。
- `tokenizer_probe.json`：冻结的fast tokenizer离线加载与一个合成对话模板化成功，34 token；未运行真实LLaMA-Factory加载。
- `audit_inputs.py`：可在仓库执行 `python3 reports/week8/phase1/audit_inputs.py` 复核；仅重写本目录的输入清单与核验结果，不改源文件或派生训练集。
- `artifact_verification.json`：协议路径/哈希、文档引用及完成范围一致性检查结果。

本次没有生成本周数据、训练模型、启动GPU、调用远端裁判或推送仓库。数据脚本尚未接入新协议。下一次应先落实任务3的数据处理入口和机器校验，再分别执行任务4–6。

审核通过指沿用已有共同KEEP选择，不代表所有事实经本次人工复查。历史技术运行不能冒充本周运行。裁判身份、benchmark数据快照和OpenCompass具体配置必须在首次新训练前锁定并重做污染检查；本次完成的是方法冻结。

任务3更正：原清单对items/prompts结构的三份评估集误计0，现已改为200/200/15，旧清单保存为`input_inventory_v1_before_count_fix.json`。v1方法和机器协议保存为`frozen_rules_v1.md`、`frozen_protocol_v1.json`。任务3详情见`../phase1_task3/`，本目录的验收状态仍表示任务1–2完成时的范围。
