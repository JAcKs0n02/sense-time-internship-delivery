# Day 26 模型归档说明

Day 26 的微调产物是 LoRA adapter，必须与固定基座 `Qwen/Qwen2-VL-7B-Instruct` 组合加载。Git 提交不包含 161.5 MB 的 `adapter_model.safetensors`，避免超过常规 GitHub 单文件限制；权重保留在 AutoDL 持久化数据盘，完整文件哈希见 `Model_Manifest.json`。

远端目录：

```text
/root/autodl-tmp/qwen2-vl-week5/day26/runs/formal_lora_v6_grounded/
```

最终评测使用 `checkpoint-84`，推理时 LoRA scale 为 `0.85`。根目录 adapter 是 LLaMA-Factory 在训练结束、加载最佳 checkpoint 后保存的正式归档版本。

加载关系：

```text
Qwen2-VL-7B-Instruct（revision eed13092…）
        + adapter_model.safetensors
        + adapter_config.json
        + LoRA scale 0.85
        = Day 26 最终候选 VLM
```

交付大文件时，应从上述远端目录复制 `adapter_model.safetensors` 与 `adapter_config.json`，复制后先重算 SHA-256，再与 `Model_Manifest.json` 比对。不要只交 adapter 而不说明基座 revision。
