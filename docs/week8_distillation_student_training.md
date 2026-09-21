# 学生蒸馏训练

采用Week4 DPO 7B教师与Qwen2.5-0.5B-Instruct学生，执行序列级教师答案监督训练。学生revision为`7ae557604adf67be50417f59c2c2f167def9a775`；不是逐token软标签KL训练。

137条训练、15条验证；BF16、LoRA rank8/alpha16、学习率5e-5、batch1、梯度累积8、cutoff1024、seed42。实际完成2轮36步，LoRA合并、张量检查和CUDA冷加载通过。

| 指标 | 结果 |
|---|---:|
| 首条/末条训练Loss | 1.6044 / 0.6115 |
| 第1轮验证Loss | 1.7604635954 |
| 第2轮验证Loss | 1.7532078028 |
| Adapter张量数 | 336 |
| 合并模型张量数 | 290 |

Loss下降仅说明优化过程，不能证明泛化提升。前后CEval评估已完成，准确率下降0.5944个百分点，速度基本持平，详见[比较分析](week8_distillation_comparison.md)。

[实际配置、数据及训练记录](../reports/week8/phase3_distillation_student_gpu/retrieved/student/)包含`distill_train.yaml`、Trainer状态、训练日志、输入绑定、合并及冷加载回执。最终模型已独立备份，见[模型清单](../models/README.md)。
