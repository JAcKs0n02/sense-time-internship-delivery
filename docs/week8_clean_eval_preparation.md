# Week8 干净环境数据准备与真实评估：准备方案

> 2026-09-19后续：新增[干净环境评估与Gemini入口](week8_clean_eval_integration.md)，已完成本地接入；旧fresh分支作为历史入口保留。新的独立发布候选与目标GPU实测仍待完成，当前候选未放行。

日期：2026-09-19。状态：步骤1（验收口径修正）和步骤2（执行准备）完成；新评估尚未放行。没有启动320、加载GPU模型、调用付费裁判或重复训练。本方案是工程选择，不是新增教师要求。

## 老师原话与本轮范围

《实习需求.docx》验收①：“自动化Pipeline在干净环境中（提供依赖清单）可无报错运行（至少包括数据准备和评估步骤）。”

Day41.1：“编写 step3_eval.py：自动加载合并后的模型，运行OpenCompass（CEval/CMMLU）并解析结果，同时执行自定义的20题人工评估集（自动计算5维度分数），输出汇总表（CSV）。”

Day41.3：“编写主控 run_pipeline.sh，支持 --skip-train、--skip-eval 等参数，实现分段执行。”

老师同时要求封装前7周流程，训练和部署入口仍属交付内容；它们已有单独实测证据。文档未要求将Linux作为独立验收条件，也未要求在本项干净环境验收中重新训练全部模型、重跑蒸馏与部署。使用Linux CUDA是现有GPU执行方案。原DOCX SHA256为`a9d6d1cd16f3f5acdf6c96f1500f3597e337b08ea276021645d49e05639bc526`，与[原始来源记录](../reports/week8/requirements_source.json)一致。

已有独立发布候选完成真实tokenizer数据准备、quick和既有评分核验；quick/score-only都是历史结果回放或复核，不替代一次新生成答案的真实评估。因此下一次仅补充：独立环境 → 正式数据准备 → 加载已有合并模型 → 两个基准完整推理 → custom20新答案与五维评分 → 新CSV及来源记录。

## 固定模型、数据与评分口径

| 项目 | 本方案选择及核验要求 |
|---|---|
| 模型 | Week8 final_dpo，作为一个已训练合并模型验证工程流程；不称它质量最好，也不据此扩大上线 |
| 本地完整备份 | `models/backups/week8-20260919/week8_final_dpo`；14文件、15,247,180,172字节；备份哈希及独立CPU加载证据见model_backup_20260919目录 |
| 320远端旧位置 | `/root/autodl-tmp/week8-full-training-runtime-20260916/logs/full-run-01/models/final_dpo`；仅是9月19日已有记录，下一次执行前重新确认并逐文件核对，不能凭旧路径认定当前存在 |
| 正式数据 | `configs/week8_data_protocol.json`冻结协议，1580条输入，1422/158划分，历史83条隔离；使用协议绑定的真实tokenizer |
| 基准 | 冻结CEval 52科1346题、CMMLU 67科11582题，共119科12928题；来自phase2_preflight锁及local_data_manifest，不重新下载替换版本 |
| 推理协议 | OpenCompass 0.5.3；BF16、batch=1、seed=42、max_seq_len=2048、基准max_out_len=32、确定性生成；沿用已修正的非递归模板，不切换CoT题型 |
| custom20 | 原固定20题及五维rubric；新答案独立保存、检查截断，沿用已审核生成协议（max_new_tokens=512）；截断则停下检查，不把残缺答案当正常结果 |
| 裁判 | 沿用已校准的Gemini `gemini-3.1-pro-preview`；Google原生generateContent接口、high thinking、maxOutputTokens=16384；20题每题一次，无自动重试 |
| 身份记录 | 新run_id、代码与数据哈希、模型清单、逐题答案哈希、新评分计划、原始响应和token用量；不覆盖旧成绩，不复用绑定旧答案的评分计划 |

裁判模型名来自已有实现和校准记录，不表示本轮已联网确认该预览版本仍可用。实际执行前核对服务可用性和价格；若版本失效，不静默替换，另做兼容与校准检查。密钥继续通过已有本机配置读取，不写入文档、仓库或命令行。

## 环境与可执行的准备命令

独立发布候选的CPU验证环境使用Python3.11、torch2.5.1、Transformers4.50.0、OpenCompass0.5.3、MMEngine0.10.7；完整快照见[CPU环境记录](../reports/week8/release_candidate_20260919/environment_freeze.txt)。这不是Linux CUDA锁文件。GPU目标拟采用Python3.10、torch2.5.1+cu121及仓库`configs/requirements-training.txt`，复用既有兼容路线；本任务不执行其中的训练功能。先在独立目录准备环境、保存安装日志，`pip check`及实际导入通过后导出目标机完整freeze。

以下为后续目标环境安装命令，本轮未执行：

```bash
conda env create -f environment.yml
conda activate internship-pipeline
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r configs/requirements-training.txt
python -m pip check
python -m pip freeze > target_environment_freeze.txt
```

当前已存在、可独立运行的数据准备入口（在仓库根目录、已安装数据依赖的环境中执行，每次使用新目录）：

```bash
unset DATA_INPUT TOKENIZER_PATH
bash run_pipeline.sh --skip-train --skip-eval --run-dir logs/clean-eval-data-001
```

当前已存在的配置生成入口：

```bash
python scripts/build_week8_formal_eval_config.py \
  --model "$WEEK8_MODEL_ABSOLUTE_PATH" \
  --output-dir reports/week8/clean-eval-target-config-001
```

该命令只生成配置并核验冻结输入，输出`verified=false`，不加载权重、不访问裁判。变量必须先设为核对后的绝对路径；目标仓库根目录改变时必须重新生成，不能直接复制本机的绝对路径配置。本轮已生成本机备份模型对应的候选配置，见[准备回执目录](../reports/week8/clean_eval_preparation_20260919/)。

## 已查明的入口缺口：必须先完成的本地工作

1. `run_pipeline.sh --skip-train`的正式评估目前仍进入`step3_eval.py fresh`。该分支按旧DeepSeek协议加载judge profile和余额检查，不能通过把模型名改成Gemini就直接使用Google接口。
2. `configs/week8_evaluation_preflight.json`没有当前可放行的顶层`runtime_lock`；历史嵌套回执、旧目标机路径和已过期的会话截止时间均不能冒充新锁。
3. 现有Gemini score-only入口绑定旧答案，用于既有成绩核验/恢复。要评估新生成的20题，必须生成新的评分计划、校验答案身份，并让主控连接新推理、评分与CSV汇总。

下一步先实现并测试上述单模型新评估接入。要求本地模拟覆盖Google请求协议、20题身份绑定、缺失学科/评分拒收、超时及未知付费状态停止、历史成绩不被覆盖；不进行真实收费调用。完成后重新导出包含本次修订的独立候选，再核验目标GPU环境、模型身份与实际prompt，生成新运行锁。

**因此目前没有一条已经放行、可直接启动的fresh端到端命令。** 不提供会错误进入旧裁判分支的启动示例，也不把配置候选的`verified`手工改为true。之后真正的OpenCompass阶段应使用现有`opencompass_command`构造命令（配置路径、独立work-dir、max-num-workers=1、dump-eval-details），经主控受控调用。

## 执行顺序、时间与预算

1. 本地完成上述接入和失败分支检查，生成独立候选及新运行计划。
2. 320使用隔离仓库和新环境，验证依赖、模型清单、冻结数据与prompt；在未核验前不启动完整推理。
3. 在同一次运行记录下完成数据准备，跳过训练；加载final_dpo，完成119科基准和20题答案生成。
4. 回收并逐文件校验全部推理输出，停止GPU并确认实例释放；本机完成20次Gemini新评分和CSV汇总。
5. 独立重算基准题数/正确数、五维加权成绩，核对数据准备产物、环境清单和全链路退出状态，再更新验收状态。

时间采用保守资源窗口，而非已测速度承诺：基准沿用历史4小时超时；加安装、模型核验和custom20，提议单模型GPU总窗口最多6小时。若无卡准备可完成安装与文件检查，GPU计费窗口可以缩短。到期停止并保留进度，不自动延长或重跑；后续必须把窗口落实到新监督器/平台关机设置，当前尚未配置。API最多20次顺序请求，总时长取决于响应速度。

费用按启动页面实际单价p元/小时计算，拟定GPU窗口预算为6p元。历史记录曾为1.48元/小时，代入示例为8.88元；这不是当前报价。裁判旧方案采用20题估算准入上限6美元，可作为本轮预算建议，但不是当前费用报价或供应商硬扣费上限；启动前按实际价格和token预算重算，保存用量。未知付费请求不自动重试。无须为工程验收再运行其余两个7B模型，也不重复已完成训练。

## 通过标准与停止条件

通过标准：独立环境安装无错误且依赖检查通过；正式数据1422/158及隔离核验通过；模型身份/模板匹配；119科完整、CEval1346和CMMLU11582答案可独立重算；custom20恰好20条新答案与20条合法五维评分，保留依据和原始响应；新CSV可从原始结果重建；退出码、阶段日志、依赖清单和产物哈希完整。低分本身不是工程失败，必须如实记录，不能为使分数变高而改评分协议。

任何输入/模型/依赖哈希不符、裁判身份改变、OOM、超时、缺失科目或题目、截断或无效评分、预算不足、请求状态不明，都停止当前阶段并保存证据。不得填0伪造缺失结果、覆盖旧分数、无控制地重试或启动其他实例。

历史发布候选及全部原始实验JSON回执保持不变。本次文档修订不等于新候选已经复验，也不等于GPU真实评估完成。最终发布与具体检查点清理仍是后续工作。
