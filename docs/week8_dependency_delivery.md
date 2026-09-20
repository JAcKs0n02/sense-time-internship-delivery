# 现行依赖与独立交付副本（2026-09-19）

本轮完成依赖版本对齐和本地独立运行检查，没有重训、启动GPU、调用裁判API或删除原仓库文件。交付副本为 `outputs/week8-submission-20260919/repository/`，后续文档同步与仓库整理继续使用这一份副本。

## 安装与验证结果

现行数据和评估使用 `configs/requirements-evaluation.txt`；Linux训练补充使用 `configs/requirements-training-cuda.txt`。共22条直接依赖声明与320实际版本记录一致，CUDA torch的`+cu121`后缀由安装源确定。旧`requirements-training.txt`与CPU加载清单保留历史原字节，不再混用。具体命令见[依赖说明](../configs/README.md)。这些是直接依赖清单，未冒称跨平台全量lock。

从空白macOS Python3.11环境安装评估清单成功，`pip check`通过，OpenCompass的HuggingFaceCausalLM及主要依赖可导入。安装结束后，运行检查使用空HOME，并阻止读取原仓库和访问外网：

| 检查 | 结果 |
|---|---|
| quick入口 | 通过；数据处理及历史结果回放 |
| 正式数据准备 | 1422训练／158验证；15个产物与冻结基准逐字节一致 |
| 现行评估配置 | 119题基准样本＋20题自定义题；仅生成配置 |
| 三模型已有裁判评分核验 | 通过；新增API请求0 |
| 现行入口测试 | 6项通过，含训练配置、CSV计算、缓存及损坏输入拒绝 |
| 最新真实评估证据的只读核验 | 366个生成文件、305个依赖、119题基准和20题裁判结果通过；custom20仍为3.775/5 |

[验证收据](../reports/week8/dependency_delivery_20260919/verification.json)、[安装日志](../reports/week8/dependency_delivery_20260919/install.log)、[版本对齐](../reports/week8/dependency_delivery_20260919/dependency_alignment.json)和[macOS实际依赖快照](../reports/week8/dependency_delivery_20260919/environment_freeze_macos.txt)保留完整边界。

## 已评分目录的迁移限制

付费评分执行记录绑定输出目录的绝对路径。额外尝试将已完成的评分目录搬到新位置再执行`--phase score`时，被`unclaimed output already exists`保护拒绝，没有产生API请求；该失败日志保留。这里没有删除执行记录、改写路径绑定或强行重试。

已评分完成的迁移材料可在副本根目录进行只读核验：

```bash
python reports/week8/dependency_delivery_20260919/check_saved_evaluation.py
```

此命令检查已有模型答案和裁判响应，不运行新模型或重新收费。GPU仅生成、尚未开始评分的运行目录仍可按主README流程迁移到本地首次评分。

## 尚未完成的范围

本轮是新安装macOS CPU环境的检查，不是从零安装Linux CUDA后的真实推理。现行训练清单仅按实测版本对齐，本轮未安装LLaMA-Factory／bitsandbytes或重新训练。已有320真实评估仍是复用既有CUDA环境的记录，不能与本轮CPU验证合并描述为“全新CUDA环境全程通过”。

副本沿用已脱敏候选并补齐现行入口及最新评估证据；历史哈希绑定路径暂时保留，尚不是最终整理后的Week1–8目录。下一步同步最终报告与提交说明，再完成各周导航整理和Git发布。检查点删除按用户决定暂不处理。
