# Day 26：VLM LoRA 微调提交

## 交付结论

本目录已完成老师要求的 200 条图文指令数据、LLaMA-Factory VLM LoRA、冻结 ViT 视觉编码器、微调日志和 Base/LoRA 效果对比。

工程训练成功，但最终效果联合门槛未通过：LoRA 在 20 题中胜出 12 题，幻觉率由 65% 降至 55%，但均分只从 3.7825 提升到 3.7950（`+0.0125`），未达到预注册的 `+0.50`。因此本提交准确标记为“实验完成；效果验收 FAIL”，不将训练 loss 下降冒充质量显著提升。

## 老师要求对应关系

| 老师要求 | 完成情况 | 证据 |
|---|---|---|
| 构造 200 条图文指令数据 | 完成：正好 200 条唯一记录，5 类各 40 条 | `Data/Training_Dataset_200.json` |
| 使用 LLaMA-Factory 做 VLM LoRA | 完成：LLaMA-Factory 0.9.3，113 steps，exit 0 | `Config/LLaMAFactory_LoRA_Config.json`、`Results/Formal_Training_Log.txt` |
| 冻结 ViT，只训练 LLM LoRA | 完成：392 个 LoRA tensor；trainer 冻结日志、可训练参数审计与 adapter 键检查均无视觉/投影训练项；基座参考张量在加载 adapter 前后 SHA 一致 | `Results/Formal_Training_Log.txt`、`Results/Trainable_Parameters.json`、`Results/Adapter_Manifest.json` |
| 交付微调后的 VLM | 完成 adapter 归档和文件哈希；随附 Day22 基座逐文件 manifest；大权重保留 AutoDL 持久化盘 | `Model_Archive/Model_Manifest.json`、`Model_Archive/Base_Model_Manifest.json` |
| 交付微调日志 | 完成 | `Results/Formal_Training_Log.txt`、`Results/Training_Summary.json` |
| 比较微调效果 | 完成两轮最终盲评；纠偏候选胜题/幻觉率通过，均分增量不通过 | `Results/Effect_Comparison.md`、`Results/Comparison_Summary.json` |

## 数据说明

- 70 张真实开源图片：50 train、10 dev、10 final；
- 来源为 scikit-image、PaddleOCR、Pix2Text 和 OmniParser；
- manifest 包含来源页、原始 URL、作者、许可、固定 revision、尺寸、文件字节和 SHA-256；
- 作者与许可字段按对应开源仓库/项目记录，用于技术追溯，不作为逐图片法律授权意见；
- 训练、开发、最终图片 SHA-256 零重叠；
- 与 Day 25 的图片和 Prompt 无污染；
- `Training_Dataset_200.json` 是便于审计的 200 条唯一内部格式；
- `Training_Dataset_LLaMAFactory.json` 是实际 LLaMA-Factory ShareGPT 多模态格式；
- v6 训练对 direct/correction 做确定性重复采样，实际采样 300 条，但没有把教师数据集虚报成 300 条唯一数据。
- `Training_Dataset_Weighted_300.json` 是正式训练实际读取的精确采样序列，`Training_Input_Manifest.json` 记录 200/300 两种视图的 SHA-256；
- 开发/最终图片与训练图片 SHA 零重叠，目标答案精确重叠为 0；三组沿用相同的通用任务 Prompt 模板，因此效果对比只代表新图片上的受控任务泛化，不代表新指令措辞泛化。

图片没有重复复制进本提交，以避免目录膨胀；`Source_Image_Manifest` 提供可重现 URL 与哈希，`Acquire_Source_Images.py` 对新下载或已存在文件都强制核对随附 manifest 的字节数与 SHA，不匹配时失败退出。

## 最终训练配置

| 字段 | 值 |
|---|---|
| Base | Qwen/Qwen2-VL-7B-Instruct |
| Base revision | `eed13092ef92e448dd6875b2a00151bd3f7db0ac` |
| Framework | LLaMA-Factory 0.9.3 (`ca75f1e…`) |
| Dataset | 200 unique / 300 weighted samples |
| LoRA target | q/k/v/o/gate/up/down projection |
| Rank / alpha / dropout | 16 / 32 / 0.05 |
| Learning rate / epoch | 2e-5 / 1.5 |
| Optimizer steps | 113 |
| Train / eval loss | 2.188334 / 1.943 |
| Selected checkpoint | checkpoint-84 |
| Inference LoRA scale | 0.85 |
| Trainable params | 40,370,176 / 8,331,745,792 |
| Adapter | 161,533,192 bytes；SHA `fb36ad94…aa423` |

## 最终盲评

匿名 A/B 评分使用 5 个维度：视觉事实 35%、指令完成 25%、完整性 15%、实用性 15%、格式 10%。评分文件在读取私有映射前冻结，SHA-256 为：

```text
2d3ea02814cedb9ce3b15ddb79b41c7df7f3bc4a029673047a779c3422a5a846
```

| 指标 | Base | LoRA | 结果 |
|---|---:|---:|---|
| 加权均分 | 3.7825 | 3.7950 | +0.0125，未达到 +0.50 |
| LoRA 胜题 | — | 12/20 | PASS |
| 幻觉率 | 65% | 55% | PASS |
| 三项联合门槛 | — | — | FAIL |

匿名 packet、匿名评分、盲化种子和私有 A/B 映射仅留工程区。本提交只提供 `Revealed_Scores.json`、`Base_LoRA_Final_Comparison.json`、汇总指标及评分前冻结的文件哈希，既便于老师复核，也不能从提交目录重建匿名映射。

运行时私有配置（含盲化参数）哈希为 `2dfb8b…a6355`；提交中的脱敏 `Evaluation_Config.json` 哈希为 `26ce16…35a5`。结果文件分别标注两种哈希，公开设置可直接从本目录重算，私有哈希仅作为运行血缘标识。最终验证还将评测摘要逐项绑定至 Day22 基座 repository/revision/文件集合摘要及 manifest SHA、最终样本 SHA、adapter config/权重 SHA 和 LoRA scale。现存 v6 结果的完整基座关联属于运行后与 Day22 已验证 manifest 的追溯绑定；更新后的评测脚本会在未来运行前逐文件校验，并用不可变 run spec 阻止混合续跑。

## 目录结构

```text
Day26_VLM_LoRA/
├── README.md
├── Data/
│   ├── Training_Dataset_200.json
│   ├── Training_Dataset_LLaMAFactory.json
│   ├── Training_Dataset_Weighted_300.json
│   ├── Training_Input_Manifest.json
│   ├── Source_Image_Manifest.csv
│   ├── Source_Image_Manifest.json
│   ├── Data_Manifest.json
│   ├── Split_Manifest.json
│   └── Acquire_Source_Images.py
├── Config/
│   ├── LLaMAFactory_LoRA_Config.json
│   └── Evaluation_Config.json
├── Results/
│   ├── Formal_Training_Log.txt
│   ├── Training_Summary.json
│   ├── Trainable_Parameters.json
│   ├── Adapter_Manifest.json
│   ├── Revealed_Scores.json
│   ├── Base_LoRA_Final_Comparison.json
│   ├── Comparison_Summary.json
│   ├── Candidate_Selection.md
│   ├── Effect_Comparison.md
│   └── Day26_Validation.json
└── Model_Archive/
    ├── Base_Model_Manifest.json
    ├── README.md
    ├── Selected_Adapter_SHA256.txt
    └── Model_Manifest.json
```

本提交不包含账号、密码、盲评私钥、平台运维记录、checkpoint/optimizer state 或缓存。
