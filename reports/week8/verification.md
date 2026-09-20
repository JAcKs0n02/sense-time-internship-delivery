# 第8周本轮验证收据

日期2026-09-14。该收据区分本地已经运行的内容与需要GPU实测的内容。

## 已实际运行

- 使用独立`/tmp/internship-week8-verify-20260914` Python环境安装Transformers4.50.0、Jinja2 3.1.6、PyYAML6.0.2；完整解析版本见local_data_environment.txt。环境为macOS Python3.14，不声称已验证Linux CUDA依赖。
- 真实Qwen tokenizer加载本地AWQ归档中的tokenizer资产，读取Day6原始5000条数据，清洗后4999条，训练4499/验证500，问题交叉0，最长完整模板2048token。统计与输出哈希见data_statistics.json；完整数据及逐条审计在logs/week8-formal-data-20260914/data。
- 实际运行`run_pipeline.sh --quick`，完成原始数据处理和历史评分复算；日志见quick_run.txt。评估是`archived_evidence_replay`，没有新模型推理。
- SFT/DPO `--dry-run`已生成真实参数配置：Week3 epoch-e5与Week4 corrective40step，新DPO输入为本次SFT合并路径。配置留在logs/week8-config-review-20260914/planned_configs.json，状态planned_only，没有训练。
- 9项unittest回归通过，完整输出见unit_tests.txt；覆盖双格式/分组无交叉、评分数值拒绝、OOM有效batch保持、非OOM不重试、空adapter不误报、训练拒绝quick数据、服务进程异常、未知参数失败及quick端到端。
- 新脚本与回收Week7 Python源码语法解析，Bash入口语法检查通过。没有执行生产模型，也没有将测试用子进程当作LLaMA-Factory训练。
- Week7本地AWQ资产及四段录屏逐文件重新计算SHA-256，清单在Submission/Week7/assets_manifest.json。它证明文件完整性，不证明模型本轮独立加载。
- 综合报告八章、前四章和全文中文字符数见report_validation.json；Word经渲染检查后导出PDF。修正了中文字体缺字及旧240step图误引用的问题，最终采用Submission的40step曲线。

## 未进行或未满足的验收

| 项目 | 状态 | 原因及下一步 |
|---|---|---|
| 新SFT/DPO训练及合并后独立加载 | NOT_RUN_GPU | 本轮本地macOS没有CUDA；需指定训练GPU环境和模型资产 |
| 新模型OpenCompass与20题AI评分 | NOT_RUN_GPU | 需实际模型及显式独立裁判端点；历史复算不替代 |
| Week4教师到0.5B学生两轮蒸馏 | NOT_RUN_GPU | 生成/训练/比较脚本及配置已写，尚无训练日志或效果表 |
| 新部署监督器联调 | NOT_RUN_GPU | 复用Week7最终应用，仍需真实服务启动/停止与健康检查 |
| 干净Linux CUDA全链路验收 | PENDING | 依赖清单已整理，本轮只验证CPU环境与数据入口 |
| 实验终稿 | DRAFT | 已有八章及足够字数；缺Week8 GPU结果，保留草稿标记 |
| checkpoint删除、Git推送、教师提交 | NOT_DONE | 本轮保留本地可审阅修改；没有删除历史模型或对外发送 |

本地可执行部分与待GPU部分均已列入docs/week8_execution_plan.md。不能将上述NOT_RUN_GPU改写为PASS。
