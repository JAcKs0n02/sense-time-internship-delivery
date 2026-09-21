# 正式数据协议

[数据目录](../data/README.md) · [冻结协议JSON](../configs/week8_data_protocol.json) · [原始规则快照](../data/provenance/data_rules_20260914.md)

正式输入为1580条已核验对话，使用固定本地tokenizer、seed42和2048 token上限。处理包含格式校验、重复检测、历史评估题隔离、问题组划分及Alpaca/ShareGPT双格式输出。超长记录整条排除，不截断答案；来源和sample_id在独立血缘文件中保存。

划分以关联问题组为单位，避免同组提示跨越训练和验证。目标比例9:1；当前结果为1422条训练、158条验证，83条历史保留样本及其他已暴露测试题继续隔离。归一化仅用于匹配，不回写对话正文。

精确匹配规则、Jaccard阈值、分组与划分算法、tokenizer及全部输入哈希以冻结JSON和原始规则快照为准。本次整理仅调整物理位置，`data/input_paths.json`负责解析历史名称，文件内容仍按原协议逐个校验。原始规则快照中的阶段待办只描述冻结当天，不代表现行完成状态。

公开基准快照位于 `data/evaluation/benchmarks/`；当前Gemini五维协议与已完成评估分别见[校准记录](week8_gemini_dimension_calibration.md)、[结果说明](week8_three_model_evaluation.md)。
