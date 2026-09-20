# 环境和配置说明

> 2026-09-19入口更新：现行命令见[主README](../README.md#当前分段流程)。`scripts/pipeline/` 复用已验证配置；新入口已在320实测119题基准样本＋完整custom20，并完成本地Gemini评分；复用既有CUDA环境，未从零安装。

`conda env create -f environment.yml` 创建本地流水线环境。`--quick`只依赖Python标准库，在无CUDA和不联网条件下运行数据处理与历史评分复算；不等于正式模型评估。

## 现行数据与评估安装

Linux NVIDIA GPU环境中另建Python3.10环境，按顺序安装：

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r configs/requirements-evaluation.txt
python -m pip check
```

`requirements-evaluation.txt`用于数据准备和合并模型评估，固定320成功运行时的核心版本：torch2.5.1、Transformers4.50.0、accelerate1.2.1、datasets3.2.0、OpenCompass0.5.3及mmengine-lite0.10.7。它是直接依赖清单，不是跨平台全量lock。CUDA轮子必须先按上面的cu121地址安装。OpenCompass数据与配置由仓库的冻结输入提供；自定义20题使用已固定Gemini裁判，Key放在仓库外。

在macOS上可新建Python3.11环境，直接安装此清单（不执行cu121命令），运行真实tokenizer数据准备、评估配置生成和历史评分核验；CUDA推理worker仍要求Linux NVIDIA GPU。不要同时安装旧的`requirements-data.txt`或`requirements-loader-cpu.txt`来覆盖此环境的版本。

实际CUDA环境的完整版本见[运行依赖快照](../reports/week8/delivery_eval_20260919/retrieved/environment_freeze.txt)。该快照包含vLLM、Jupyter及本机可编辑安装路径，仅供审计，不应直接作为可移植requirements安装。现行清单与实测记录的版本对齐，不等于已从零重建CUDA环境。

新安装macOS Python3.11环境已通过安装、`pip check`、正式数据15文件逐字节复现、评估配置及6项入口检查；完整结果和未覆盖的CUDA范围见[本轮交付检查](../docs/week8_dependency_delivery.md)。

## 正式训练安装

如需重新训练，在上述Linux CUDA环境继续安装：

```bash
python -m pip install -r configs/requirements-training-cuda.txt
python -m pip check
```

该清单补充LLaMA-Factory0.9.3、PEFT0.15.1、TRL0.9.6和bitsandbytes0.43.3，版本依据[正式训练预检](../reports/week8/phase2_full_training_target/evidence/logs/full-run-01/target_preflight.json)。原`requirements-training.txt`由旧运行计划绑定哈希，保留原字节作为历史证据，不再作为现行安装入口。`requirements-loader-cpu.txt`同样仅记录此前CPU加载环境。本次交付不需要重复训练。

现行训练入口读取 `week8_full_training_candidate/sft.yaml` 和 `dpo.yaml`，仅替换本次模型、数据和输出路径。参数源于Week3/4方案，并已在Week8正式训练中实测：SFT rank8、alpha16、lr1e-4、5epoch、seed42；DPO beta0.1、lr2e-6、40steps。DPO引用本次合并SFT，adapter禁用提供隐式参考。生成计划不等于再次训练通过。

## 部署环境

文字服务：Python3.10、torch2.5.1+cu121、vLLM0.6.4.post1、Transformers4.50.0、NumPy1.26.4。视觉服务单独环境，Transformers4.46.3；不要将两环境混装。UI版本见requirements-ui.txt。已验证旧部署详见 Submission/Week7/本地部署与操作手册.md；本周监督启动器已完成GPU联调，见[监督部署实测](../docs/week8_supervised_deployment.md)。

## 蒸馏

`distillation.json`固定0.5B学生和2轮，采用离线序列级教师目标。温度0.7是教师采样温度，不是KL软标签温度。理论上软标签蒸馏通常使用温度缩放分布并用T²缩放KL项；本实现没有该项，不能把生成后SFT称作logit蒸馏。[理论实现参考](https://huggingface.co/docs/transformers/v4.48.1/en/tasks/knowledge_distillation_for_image_classification)。

## 已验证的CPU数据加载环境

任务5已使用Python3.11.14、LLaMA-Factory0.9.3、Transformers4.50.0和Torch2.5.1，在独立macOS CPU环境完成真实加载验收。核心依赖见`requirements-loader-cpu.txt`，全量解析快照见`../reports/week8/phase1_task5/environment_freeze.txt`。这不替代上文Linux CUDA环境的实际复验，详情见[任务5结果](../docs/week8_phase1_task5_result.md)。
