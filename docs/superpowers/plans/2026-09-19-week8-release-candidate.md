# Week8 独立发布候选执行计划

> **For agentic workers:** 使用 superpowers:executing-plans 的逐项核验方式在当前会话执行；本次是独立导出与文档整理，不改训练算法、不新建开发分支、不提交推送。

**Goal:** 生成有明确白名单、可追溯脱敏和独立运行证据的本地发布候选。
**Architecture:** 原仓库只新增说明和证据；候选目录位于被忽略的 outputs/week8-release-candidate-20260919/repository。完整历史身份回执保留，脱敏副本使用独立新清单；冻结正式输入必须保持原哈希。
**Tech Stack:** Python 3.11、标准库导出/哈希、隔离 venv、现有 CPU 数据准备和评分核验入口。
**Spec:** docs/week8_release_review.md 的后续步骤2及3（本轮限CPU，不覆盖CUDA）。

## Global Constraints

- 不复制 .git、凭据、环境缓存、权重或压缩归档；仅显式列出的文字源文件、必要冻结输入和已核验的技术报告附件进入候选。
- 不修改9个原始/派生数据源；在导出副本中替换已知凭据字面量并记录文件、行号、数量和前后哈希，禁止记录凭据原文。
- 不改写历史评分、训练结果或冻结哈希以掩盖脱敏差异。正式训练输入如命中凭据则停止该入口放行。
- 不启动GPU、付费裁判、不提交推送、不删除原始模型或历史文件。

## Task 1: 文件范围与依赖
- [x] 读取配置中的冻结路径，结合现有脚本/审查记录建立逐文件白名单。
- [x] 为排除文件记录类别，明确历史附件不在本候选中的范围。

## Task 2: 可追溯导出
- [x] 在 reports/week8/release_candidate_20260919/保存导出脚本、白名单和转换映射。
- [x] 新建 outputs/week8-release-candidate-20260919/repository，按白名单复制；拒绝越界及符号链接。
- [x] 比对原始9文件哈希不变，导出JSONL行数/结构不变，改动仅限已知字面量。
- [x] 生成候选校验表，原始历史清单保留并说明其适用范围。

## Task 3: 独立环境验收
- [x] 创建全新Python venv，安装 configs/requirements-data.txt 与 pytest，记录版本。
- [x] 在候选目录运行 quick、score-only all、冻结正式数据准备及现有相关测试；禁止测试进程读原仓库及外网连接。
- [x] 如发现导出遗漏，按真实访问/配置依赖补入白名单，重新导出并重跑受影响检查，不降低原有哈希门槛。

## Task 4: 最终审查与说明
- [x] 对候选全量文本扫描已知凭据及密钥模式，检查私钥/环境文件和大文件，保留扫描边界。
- [x] 核对候选文件清单、原源完整性及运行回执，生成 verification.json 和独立证据清单。
- [x] 写 docs/week8_release_candidate.md，更新当前入口，说明CPU通过范围与未完成CUDA/正式发布项。

## 执行结果

全部170项测试通过，三个入口通过；全量扫描、清单和原始源完整性最终核对见 reports/week8/release_candidate_20260919/verification.json。本轮未提交推送。
