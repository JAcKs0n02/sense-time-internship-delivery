#!/usr/bin/env python3
"""Project the verified Week6 evidence into a shallow teacher submission."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
SUBMISSION = REPO_ROOT / "Submission/Week6"

DAY31_FILES = {
    "Config/dataset_info_v2.json": "deliverables/week6/day31/configs/dataset_info_v2.json",
    "Config/runtime_versions_v2.json": "deliverables/week6/day31/configs/runtime_versions_v2.json",
    "Config/week6_tool_sft_lora_v2.yaml": "deliverables/week6/day31/configs/week6_tool_sft_lora_v2.yaml",
    "Config/week6_tool_sft_lora_v2_effective.yaml": "deliverables/week6/day31/source/results/week6_tool_sft_lora_v2_effective.yaml",
    "Data/dataset_manifest_v2.json": "deliverables/week6/day31/source/data/dataset_manifest_v2.json",
    "Data/tool_sft_train_core_100.json": "deliverables/week6/day31/source/data/tool_sft_train_core_100.json",
    "Data/tool_sft_train_v2.json": "deliverables/week6/day31/source/data/tool_sft_train_v2.json",
    "Data/tool_sft_dev_v2.json": "deliverables/week6/day31/source/data/tool_sft_dev_v2.json",
    "Data/tool_sft_test_v2.json": "deliverables/week6/day31/source/data/tool_sft_test_v2.json",
    "Model_Archive/adapter_config.json": "deliverables/week6/day31/model_archive/adapter_config.json",
    "Model_Archive/adapter_manifest.json": "deliverables/week6/day31/source/results/v2_adapter_manifest.json",
    "Results/data_validation_v2.json": "deliverables/week6/day31/source/results/data_validation_v2.json",
    "Results/official_parser_receipt_v2.json": "deliverables/week6/day31/source/results/official_parser_receipt_v2.json",
    "Results/formal_preflight_v2.json": "deliverables/week6/day31/source/results/formal_preflight_v2.json",
    "Results/launch_command_v2.json": "deliverables/week6/day31/source/results/launch_command_v2.json",
    "Results/exit_status_v2.json": "deliverables/week6/day31/source/results/exit_status_v2.json",
    "Results/trainable_parameters_v2.json": "deliverables/week6/day31/source/results/trainable_parameters_v2.json",
    "Results/training_summary_v2.json": "deliverables/week6/day31/source/results/v2_training_summary.json",
    "Results/training_log_v2.txt": "deliverables/week6/day31/source/results/v2_training_log.txt",
    "Results/checkpoint_selection_v2.json": "deliverables/week6/day31/source/results/v2_checkpoint_selection.json",
    "Results/frozen_release_v2.json": "deliverables/week6/day31/source/results/v2_frozen_release.json",
    "Results/final_test_raw_v2.json": "deliverables/week6/day31/source/results/v2_final_test_raw.json",
    "Results/final_test_guarded_v2.json": "deliverables/week6/day31/source/results/v2_final_test_guarded.json",
    "Results/final_test_summary_v2.json": "deliverables/week6/day31/source/results/v2_final_test_summary.json",
    "Scripts/build_tool_sft_dataset.py": "deliverables/week6/day31/source/scripts/build_tool_sft_dataset.py",
    "Scripts/day31_harness.py": "deliverables/week6/day31/source/scripts/day31_harness.py",
    "Scripts/select_v2_checkpoint.py": "deliverables/week6/day31/source/scripts/select_v2_checkpoint.py",
    "Scripts/validate_tool_sft_dataset.py": "deliverables/week6/day31/source/scripts/validate_tool_sft_dataset.py",
}

DAY32_FILES = {
    "Code/day32_analysis.py": "deliverables/week6/source/week6_agent/day32_analysis.py",
    "Config/prompt_manifest_v2.json": "deliverables/week6/day32/source/configs/prompt_manifest_v2.json",
    "Config/system_prompt_main.txt": "deliverables/week6/day32/source/configs/system_prompt_main.txt",
    "Config/system_prompt_ablation_input_fidelity.txt": "deliverables/week6/day32/source/configs/system_prompt_ablation_input_fidelity.txt",
    "Data/tool_sft_dev_v2.json": "deliverables/week6/day31/source/data/tool_sft_dev_v2.json",
    "Results/prompt_main_eval_v2.json": "deliverables/week6/day32/source/results/v2_dev_prompt_main.json",
    "Results/prompt_ablation_eval_v2.json": "deliverables/week6/day32/source/results/v2_dev_prompt_ablation.json",
    "Results/prompt_selection_v2.json": "deliverables/week6/day32/source/results/v2_prompt_selection.json",
    "Results/main_error_analysis.json": "deliverables/week6/day32/source/results/v2_analysis/baseline_error_analysis.json",
    "Results/main_error_cases.csv": "deliverables/week6/day32/source/results/v2_analysis/baseline_error_cases.csv",
    "Results/ablation_error_analysis.json": "deliverables/week6/day32/source/results/v2_analysis/optimized_error_analysis.json",
    "Results/ablation_error_cases.csv": "deliverables/week6/day32/source/results/v2_analysis/optimized_error_cases.csv",
    "Results/prompt_comparison.json": "deliverables/week6/day32/source/results/v2_analysis/prompt_comparison.json",
    "Results/Agent_Error_Mode_Analysis_Report.md": "deliverables/week6/day32/source/results/AGENT_ERROR_ANALYSIS.md",
    "Results/day32_validation.json": "deliverables/week6/day32/source/results/day32_validation.json",
    "Scripts/analyze_failures.py": "deliverables/week6/day32/source/scripts/analyze_failures.py",
    "Scripts/validate_day32.py": "deliverables/week6/day32/source/scripts/validate_day32.py",
}

DAY33_FILES = {
    "Code/day33_reporting.py": "deliverables/week6/source/week6_agent/day33_reporting.py",
    "Model_Archive/adapter_config.json": "deliverables/week6/day31/model_archive/adapter_config.json",
    "Model_Archive/adapter_manifest.json": "deliverables/week6/day31/source/results/v2_adapter_manifest.json",
    "Report/Week6_Agent_Development_Report.md": "deliverables/week6/day33/REPORT.md",
    "Results/day33_validation.json": "deliverables/week6/day33/source/results/day33_validation.json",
    "Results/final_agent_archive.json": "deliverables/week6/day33/source/results/final_agent_archive.json",
    "Results/report_input_manifest.json": "deliverables/week6/day33/source/results/report_input_manifest.json",
    "Results/week6_acceptance_matrix.csv": "deliverables/week6/day33/source/results/week6_acceptance_matrix.csv",
    "Scripts/build_day33.py": "deliverables/week6/day33/source/scripts/build_day33.py",
    "Scripts/validate_day33.py": "deliverables/week6/day33/source/scripts/validate_day33.py",
}

DAY31_README = """# Day31 工具调用 SFT

本目录包含 100 条核心训练样本、200 条扩展样本、40 条开发集、100 条最终测试集，以及正式 LoRA 训练、checkpoint 选择和冻结测试证据。

- 训练：300 条，2 epoch，76 steps，退出码 0；最终 eval loss 0.0638。
- 开发集：checkpoint-38 的严格成功率最高（45.0%），因此被冻结为最终候选。
- 最终测试（raw）：首工具 91.0%，完整工具序列 91.0%，参数正确率 89.0%，严格成功率 39.0%。
- guarded 结果单独报告；它只执行安全策略，不替代 raw 模型能力指标。

模型大权重不进入 Git。`Model_Archive/adapter_manifest.json` 给出路径、大小和 SHA-256，`Results/frozen_release_v2.json` 固定 checkpoint、Prompt、测试集与生成参数。
"""

DAY32_README = """# Day32 Agent 错误分析与 Prompt 消融

在同一 checkpoint-38、同一 40 条开发集和相同生成参数下，只比较 main Prompt 与单规则 input-fidelity ablation。

- main：严格成功率 45.0%，语义正确率 50.0%，失败 22/40；
- ablation：严格成功率 35.0%，语义正确率 47.5%，失败 26/40；
- 选择 main：1 条改善、5 条回归，未观察到死循环。

`Results/Agent_Error_Mode_Analysis_Report.md` 记录分类规则、数量、代表案例、根因和结论；两份原始评测、逐题 CSV 与比较 JSON 可独立复核。
"""

DAY33_README = """# Day33 第 6 周周报与归档

本目录包含《第 6 周：Agent 智能体开发报告》、老师四项验收矩阵、报告输入哈希和最终 Agent 引用归档。

最终组合：Week4 DPO merged model + Day31 checkpoint-38 LoRA + main Prompt + 三个固定工具。模型权重保存在持久化模型目录；教师包只交付配置、manifest、哈希和可复核结果。
"""

WEEK6_README = """# Week6 教师提交包

本目录是精简教师交付，不包含整个仓库。

| 日期 | 交付内容 |
|---|---|
| Day28 | Calculator、KnowledgeRetrieval 与验证记录 |
| Day29 | ReAct Agent 与单轮调用日志 |
| Day30 | AST-only CodeExecutor 与三阶段复杂任务 |
| Day31 | 工具调用 SFT 数据、训练证据、checkpoint 选择与最终测试 |
| Day32 | 错误分类、单因素 Prompt 消融与分析报告 |
| Day33 | Week6 周报、验收矩阵和最终模型引用归档 |

每个日期目录使用浅层结构，`Submission_Map.json` 绑定正式源文件，根目录 `SHA256SUMS.txt` 覆盖除其自身外的全部文件。提交包不含 checkpoint、大权重、缓存、凭据或平台运维记录。
"""

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _project(day_name: str, files: dict[str, str], readme: str) -> None:
    day = SUBMISSION / day_name
    if day.exists():
        shutil.rmtree(day)
    records: list[dict[str, object]] = []
    for destination, source_name in sorted(files.items()):
        source = REPO_ROOT / source_name
        if not source.is_file():
            raise FileNotFoundError(source)
        target = day / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append({"submission_path": destination, "formal_source": source_name, "bytes": source.stat().st_size, "sha256": _sha(source)})
    day.mkdir(parents=True, exist_ok=True)
    (day / "README.md").write_text(readme, encoding="utf-8")
    (day / "Submission_Map.json").write_text(json.dumps({"schema_version": "1.0", "files": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def _write_checksums() -> None:
    checksum = SUBMISSION / "SHA256SUMS.txt"
    paths = sorted(path for path in SUBMISSION.rglob("*") if path.is_file() and path != checksum)
    checksum.write_text("".join(f"{_sha(path)}  ./{path.relative_to(SUBMISSION).as_posix()}\n" for path in paths), encoding="utf-8")

def main() -> int:
    _project("Day31_Tool_Call_SFT", DAY31_FILES, DAY31_README)
    _project("Day32_Error_Analysis", DAY32_FILES, DAY32_README)
    _project("Day33_Weekly_Report", DAY33_FILES, DAY33_README)
    shutil.copy2(REPO_ROOT / "deliverables/week6/day31/model_archive/README.md", SUBMISSION / "Day31_Tool_Call_SFT/Model_Archive/README.md")
    shutil.copy2(REPO_ROOT / "deliverables/week6/day33/model_archive/README.md", SUBMISSION / "Day33_Weekly_Report/Model_Archive/README.md")
    (SUBMISSION / "README.md").write_text(WEEK6_README, encoding="utf-8")
    _write_checksums()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
