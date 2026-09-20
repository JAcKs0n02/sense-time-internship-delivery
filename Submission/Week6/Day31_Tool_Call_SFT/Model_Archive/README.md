# Day31 Tool-call SFT 模型归档

正式训练输出位于 AutoDL 持久化目录：

`/root/autodl-tmp/qwen25-week6/formal-v2/runs/tool-sft-lora-v2`

最终部署候选为该目录下的 `checkpoint-38`：

- 基座：已核验的 Week4 DPO merged model；
- 权重文件：`checkpoint-38/adapter_model.safetensors`；
- 权重 SHA-256：`a5e348ad180f8183479d3138b40613ead7163f3b2089c9b271063b64be5cdc5c`；
- adapter config SHA-256：`3f9585b4844615acbddaf03ebc140f1e6b74694ea6bac801840e65392bf6222d`；
- LoRA target modules：q/k/v/o projection；rank 16，alpha 32，dropout 0.05。

训练目录 final adapter 为 40,400,200 bytes，并与 checkpoint-76 字节一致；它不是本次开发集排序选出的候选。部署与复现应以 `v2_frozen_release.json` 指向的 checkpoint-38 为准。

Git 与教师目录保存配置、manifest、大小和 SHA，不复制模型权重、checkpoint、optimizer state 或平台运维记录。
