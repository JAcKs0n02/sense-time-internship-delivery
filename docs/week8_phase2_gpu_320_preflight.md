# Week8：320机基础环境与原始基座预检

日期：2026-09-15。结论：**GPU_BASIC_AND_ORIGINAL_BASE_IDENTITY_PASS_NOT_TRAINING_READY**。

经用户确认，以最多1小时、费用上限¥1.48启动AutoDL 320机。已设置页面服务器时间12:50自动关机作为上限，预检完成后提前手动关机。实际生命周期状态见[会话收据](../reports/week8/phase2_gpu_320/session.json)。未操作344机。

## 实际结果

- 主机autodl-container-be044ebe99-be706b14；RTX 3090，24576 MiB；驱动595.71.05，PyTorch CUDA 12.1。预检前GPU无计算进程，显存仅1 MiB。
- 实际环境为/root/autodl-tmp/conda/envs/llm_exp/bin/python，Python 3.10.20、PyTorch 2.5.1+cu121、Transformers 4.50.0、LLaMA-Factory 0.9.3、datasets 3.2.0、bitsandbytes 0.43.3、OpenCompass 0.5.3。pip check通过。
- CUDA可用、BF16支持及16×16矩阵运算通过。4096元素NF4量化/反量化输出有限，平均绝对误差0.0140209049，小于探针阈值0.1。仅验证基础运算链路，不代表真实模型量化质量。
- 对原始Qwen2.5-7B-Instruct目录的15个文件重新计算SHA-256，均与此前原始基座身份记录一致；包括4个完整safetensors分片、索引、config和tokenizer。索引引用的分片齐全，没有adapter文件。未加载完整模型，也未执行模型forward或训练。
- 7份关键LLaMA-Factory源码与任务5 CPU验收的文件哈希全部一致。

## 保留的差异与处理

实际PEFT为0.15.1、Accelerate为1.2.1；CPU验收环境为0.15.2、1.7.0。未擅自升级现有环境；基础依赖及导入通过仍不能代替本周1580条在此环境的真实loader复验。

初次探针出现OMP_NUM_THREADS无效值提示，任务仍完成；补充探针仅对本次进程显式设为4，核心库导入和NF4计算通过。后续启动配置应显式固定有效线程数，不能忽略该提示。未改系统环境或历史项目文件。

现有SSH密钥认证失败，本次使用用户已登录的内嵌浏览器JupyterLab终端完成核验。没有读取密码、建立新密钥或更改认证配置。原始收据经编辑器读取后回收到本地，并与远端文件SHA-256核对一致；不是把历史结果当作新运行结果。

## 证据

- [原始远端收据](../reports/week8/phase2_gpu_320/remote_preflight.json)
- [量化与源码补充收据](../reports/week8/phase2_gpu_320/remote_supplement.json)
- [原始基座对照身份](../reports/week8/phase2_gpu_320/expected_base_identity.json)
- [综合验证结果](../reports/week8/phase2_gpu_320/verification.json)

## 下一步

仍需完成本周数据在目标环境的真实loader复验、OpenCompass显式配置接入及全题提示长度核验、裁判精确身份和完整评估运行锁。通过后才能分别进行3步SFT/DPO技术试跑。此次没有开始试跑、正式训练、评估推理或外部裁判调用。
