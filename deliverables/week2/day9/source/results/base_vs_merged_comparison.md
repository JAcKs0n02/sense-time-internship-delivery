# Base vs Merged Model Comparison

The questions and deterministic generation settings were frozen before inference. Scoring used five dimensions, each from 0 to 2.

- Model A total: 25/30
- Model B total: 22/30
- Generation-quality regression: no
- Merged model better: **no**

Unblinding: Model A is the base model; Model B is the merged model.

## Q1

Question: 请用不超过150字解释“为什么指令微调数据要做去重”，并给出一个重复样本可能带来的具体后果。

### Model A

> 指令微调数据做去重是为了确保模型训练时使用的数据集尽可能精炼，避免因重复样本导致的信息冗余和训练效率下降。重复样本可能导致模型过度拟合特定模式而忽视其他重要信息，影响泛化能力。
>
> 具体后果示例：假设在指令微调中，有多个样本要求模型生成天气预报，内容完全相同但来自不同时间点。这会导致模型在训练时过分关注这些重复的天气描述，而忽略了学习如何处理不同类型的天气变化或地理区域的差异性，从而降低模型的实际应用效果和适应性。

Score: 8/10 (instruction_completion=1, correctness=2, format_compliance=1, relevance_completeness=2, generation_quality=2)

Notes: Explains the reason and gives a concrete consequence, but the 208-character response exceeds the 150-character limit.

### Model B

> 指令微调数据需要去重，因为重复样本会导致模型在训练过程中出现过拟合的情况，从而降低模型的泛化能力。例如，如果在指令微调数据集中存在两个完全相同的样本，那么模型在训练过程中会将这两个样本都视为正例或反例，并试图通过调整权重来最大化这两个样本的分类准确率。这样，模型就会过度拟合这些样本，而忽略了其他样本的信息，导致模型在新的数据上的表现不佳。因此，在指令微调数据集中去除重复样本是非常重要的。

Score: 7/10 (instruction_completion=1, correctness=1, format_compliance=1, relevance_completeness=2, generation_quality=2)

Notes: States overfitting and weaker generalization, but the 195-character response exceeds the limit and frames causal-language-model SFT too narrowly as classification.

## Q2

Question: 某台24GB显存的服务器在训练7B模型时出现CUDA OOM。请严格按“现象—可能原因—排查顺序—优先解决方案”四个标题回答，每部分最多3点。

### Model A

> ### 现象
> - 服务器在训练7B模型时，出现CUDA OOM（Out of Memory）错误。
> - 训练过程中，程序突然中断，并显示内存不足的错误信息。
>
> ### 可能原因
> - **显存不足**：虽然服务器有24GB显存，但7B模型可能需要更多的显存来运行，尤其是在进行复杂的计算或批量处理时。
> - **数据加载问题**：数据加载器可能一次性加载了过多的数据到显存中，导致显存溢出。
> - **优化器状态占用**：优化器的状态（如梯度和动量）也可能占用大量显存，特别是在使用Adam等复杂优化器时。
>
> ### 排查顺序
> 1. **检查模型配置**：确认模型配置文件中的显存需求是否合理，是否有不必要的冗余参数。
> 2. **监控显存使用情况**：使用NVIDIA的NVIDIA-SMI工具监控显存使用情况，查看哪些操作占用了大量显存。
> 3. **调整数据加载策略**：检查数据加载器的batch size设置，尝试减小batch size以减少每次迭代时的数据量。
>
> ### 优先解决方案
> 1. **增加显存或优化模型**：如果可能，增加服务器的显存容量；或者通过模型剪枝、量化等方式优化模型，减少显存需求。
> 2. **优化数据加载**：将数据加载改为分批处理，避免一次性加载大量数据到显存中。
> 3. **减少优化器状态**：考虑使用更轻量级的优化器，或者在训练过程中定期清空优化器状态。

Score: 9/10 (instruction_completion=2, correctness=1, format_compliance=2, relevance_completeness=2, generation_quality=2)

Notes: Uses all four required headings and no more than three points per section. Some advice is not the best technical priority, especially clearing optimizer state during training.

### Model B

> 现象：
> 1. 在使用24GB显存的服务器训练7B模型时，出现CUDA OOM错误。
> 2. 训练过程中，GPU内存占用率持续升高，最终达到100%并触发OOM错误。
>
> 可能原因：
> 1. 模型参数过多：7B参数量较大的模型需要大量的显存来存储权重和偏置等参数。
> 2. 批处理大小过大：批处理大小过大会导致每次迭代所需的显存增加，从而导致OOM错误。
> 3. 数据预处理不当：数据预处理过程中可能会生成临时数据，占用额外的显存空间。
> 4. 代码实现问题：代码中可能存在内存泄漏或不必要的数据复制等问题，导致显存占用过高。
>
> 排查顺序：
> 1. 检查模型参数量是否过大，如果参数量过大可以尝试减小模型规模。
> 2. 调整批处理大小，减少每次迭代所需的显存。
> 3. 检查数据预处理过程，确保没有生成额外的数据。
> 4. 检查代码实现，寻找可能的内存泄漏或不必要的数据复制问题。
>
> 优先解决方案：
> 1. 尝试减小模型规模，降低参数量。
> 2. 调整批处理大小，减少每次迭代所需的显存。
> 3. 如果上述方法无效，可以考虑使用混合精度训练，通过降低部分变量的精度来节省显存。
> 4. 最后，如果仍然无法解决问题，可以考虑更换更高配置的服务器，以满足模型对显存的需求。

Score: 5/10 (instruction_completion=1, correctness=1, format_compliance=0, relevance_completeness=1, generation_quality=2)

Notes: Covers all four sections, but three sections contain four points despite the maximum of three, and it omits higher-priority methods such as gradient accumulation or checkpointing.

## Q3

Question: 把下面这句话改写成一条清晰、可执行且包含验收标准的任务指令：“帮我看看这些数据，太乱了，处理一下。”只输出改写后的指令。

### Model A

> 请整理并优化以下数据集，确保数据清晰有序。具体要求如下：
> 1. 检查并删除重复项。
> 2. 填补缺失值。
> 3. 格式化日期和数字字段。
> 4. 重新排序行以按时间顺序排列。
>
> 完成任务后，请将整理后的数据集发送给我，并附上简要说明每个步骤的操作结果。

Score: 8/10 (instruction_completion=1, correctness=2, format_compliance=2, relevance_completeness=1, generation_quality=2)

Notes: Produces only a task instruction and gives concrete operations, but the acceptance criteria are implicit and it assumes time-sortable fields without evidence.

### Model B

> 请整理并清理这些数据，确保它们清晰有序，并将结果呈现给我。验收标准：数据整洁有序，无重复或缺失项，格式统一。

Score: 10/10 (instruction_completion=2, correctness=2, format_compliance=2, relevance_completeness=2, generation_quality=2)

Notes: Concise, directly executable, contains an explicit acceptance-standard clause, and outputs only the rewritten instruction.
