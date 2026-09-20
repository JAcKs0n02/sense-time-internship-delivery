# Day32 Agent 错误分析与 Prompt 消融

在同一 checkpoint-38、同一 40 条开发集和相同生成参数下，只比较 main Prompt 与单规则 input-fidelity ablation。

- main：严格成功率 45.0%，语义正确率 50.0%，失败 22/40；
- ablation：严格成功率 35.0%，语义正确率 47.5%，失败 26/40；
- 选择 main：1 条改善、5 条回归，未观察到死循环。

`Results/Agent_Error_Mode_Analysis_Report.md` 记录分类规则、数量、代表案例、根因和结论；两份原始评测、逐题 CSV 与比较 JSON 可独立复核。
