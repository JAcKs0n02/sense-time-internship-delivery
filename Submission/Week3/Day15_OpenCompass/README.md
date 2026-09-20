# Day 15 OpenCompass Evaluation

本目录是 Day 15 教师交付。OpenCompass 0.5.3 在同一 RTX 3090 上依次评测基座模型与 Day 14 已选出的最优 SFT `epoch-e5`，覆盖 CEval 52 学科、1,398 题和 CMMLU 67 学科、11,649 题。两个模型均退出码为 0。

| Model | Dataset | Sample-weighted accuracy (%) | Macro subject score (%) | Delta vs base (pp) |
|---|---|---:|---:|---:|
| Base | CEval | 77.542262 | 77.001569 | 0.000000 |
| Base | CMMLU | 0.647412 | 0.622482 | 0.000000 |
| Best SFT (`epoch-e5`) | CEval | 79.189426 | 78.977255 | +1.647164 |
| Best SFT (`epoch-e5`) | CMMLU | 5.205187 | 5.003287 | +4.557775 |

完整四行结果见 `OpenCompass_Scores.csv`。OpenCompass 不参与超参数选择，只验证 Day 14 已选模型的公开基准泛化能力；CMMLU 的绝对分数仍低，不应扩张解释为稳定的通用知识能力。
