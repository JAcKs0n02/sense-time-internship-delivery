# Week 4 Day 17 Submission and Pages Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将已验证的 Day 17 工程成果投影为教师提交包，同步原生 Pages 完成状态，并把全部授权改动提交、推送到当前分支。

**Architecture:** `deliverables/week4/day17/` 继续保存完整工程归档；`Submission/Week4/Day17_Preference_Data_Methodology/` 只保存老师要求的指南和少量必要支持证据。`Submission/SHA256SUMS.txt` 作为统一完整性清单；`.pages` 只追加 Day 17 完成标记和仓库交付路径，不改写老师原始要求。

**Tech Stack:** Markdown、JSON、Python 3.11、pytest、SHA-256、Apple Pages、Git/GitHub。

## Global Constraints

- `Submission/` 内所有目录名和文件名使用 ASCII 英文。
- Day 17 教师核心交付必须包含《偏好数据构造指南》。
- 支持材料限定为 taxonomy、10 组样例、验证结果和提交说明，不复制缓存或模型权重。
- 保留老师 `.pages` 的原始要求，只在 Day 17 交付行后追加完成状态。
- 不创建 Day 18–Day 21 的虚假结果或占位教师交付。
- 提交前验证 JSON、Markdown 链接、SHA-256、pytest 和 Pages 重渲染。

---

### Task 1: 修复 Submission 清单对本地生成压缩包的边界

**Files:**
- Modify: `deliverables/week3/day16/source/tests/test_validate_week3.py`
- Modify: `deliverables/week3/day16/source/scripts/validate_week3.py`

**Interfaces:**
- Consumes: `Submission/SHA256SUMS.txt` 和 Submission 文件树。
- Produces: 只验证正式源文件、忽略 `.gitignore` 已定义的生成压缩包。

- [ ] 添加临时 Submission 中含 `Week3.zip` 但清单只列正式 README 的回归测试。
- [ ] 运行该测试，确认旧逻辑报告清单缺失。
- [ ] 在清单枚举中排除 `.zip` 生成归档，不放宽其他未知文件检查。
- [ ] 运行 Day 16 测试确认回归通过。

### Task 2: 创建 Week 4 Day 17 教师提交包

**Files:**
- Create: `Submission/Week4/README.md`
- Create: `Submission/Week4/Day17_Preference_Data_Methodology/README.md`
- Create: `Submission/Week4/Day17_Preference_Data_Methodology/Preference_Data_Construction_Guide.md`
- Create: `Submission/Week4/Day17_Preference_Data_Methodology/Preference_Taxonomy.json`
- Create: `Submission/Week4/Day17_Preference_Data_Methodology/Preference_Pair_Examples.json`
- Create: `Submission/Week4/Day17_Preference_Data_Methodology/Day17_Validation.json`
- Modify: `Submission/README.md`

**Interfaces:**
- Consumes: `deliverables/week4/day17/` 中已验证的正式成果。
- Produces: 教师可从 `Submission/Week4/README.md` 进入的英文路径提交包。

- [ ] 复制指南、taxonomy、样例和验证结果；三个 JSON 逐字节一致，教师版指南只重写两个包内相对链接。
- [ ] 编写 Day 17 提交说明，逐项映射老师要求与文件入口。
- [ ] 编写 Week 4 索引，只把 Day 17 标为完成。
- [ ] 在统一 Submission README 增加 Week 4 行和目录政策说明。

### Task 3: 更新老师 Pages 文档

**Files:**
- Modify: `商汤实习需求.pages`

**Interfaces:**
- Consumes: Week 4 Day 17 原始交付行。
- Produces: 原始要求后追加的完成状态、验证数字和 Submission 路径。

- [ ] 使用 Apple Pages 打开当前文件并定位 Week 4 Day 17。
- [ ] 在“交付：《偏好数据构造指南》文档。”后追加 Day 17 完成记录。
- [ ] 保存并通过 Quick Look 重渲染，确认 Week 4 页面文字可读、Day 18–Day 21 原文未改变。

### Task 4: 重建统一清单并验证

**Files:**
- Modify: `Submission/SHA256SUMS.txt`
- Modify: `README.md`

**Interfaces:**
- Consumes: 最终 Submission 文件树和 Pages 更新结果。
- Produces: 排序、无重复、哈希匹配的统一清单和根索引。

- [ ] 更新根 README 的教师提交入口与 Week 4 状态。
- [ ] 对 Submission 正式文件按相对路径排序生成 SHA-256，清单不列自身和生成 zip。
- [ ] 运行 Day 17 focused tests、Day 16 Submission 合同、JSON/链接/whitespace/秘密扫描和 Pages 渲染检查。
- [ ] 运行仓库测试；Day 2 GPU 测试在本机缺少 torch 时使用已记录的排除命令复核其余测试。

### Task 5: 提交并推送

**Files:**
- Stage: 用户明确授权的全部仓库改动，包括 `.pages`。

**Interfaces:**
- Consumes: 全部验证通过的工作树。
- Produces: 当前 `codex/week4-day17` 分支上的单一可追溯提交和 origin 推送。

- [ ] 检查 `git status` 与 staged diff，确认没有缓存、权重、凭据或未授权文件。
- [ ] 执行 `git add -A`、创建描述 Day 17 与 Submission 的提交。
- [ ] 推送 `codex/week4-day17` 到 `origin`，核对本地 HEAD 与远端分支 SHA 一致。
