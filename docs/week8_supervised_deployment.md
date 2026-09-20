# Week8 监督部署器实测

本轮依据 Day41.2：自动启动量化 vLLM 和 Gradio，后台运行并检查健康状态。部署对象是第7周已核验的 Week4 DPO AWQ 4-bit 资产，不是此次蒸馏0.5B学生，也不是Week8新DPO模型；工程部署通过不能替代模型质量验收。

## 部署前检查

重新执行学生训练52份证据校验、前后比较243份证据核验和1346题重计分，结果通过。47道选项变化中8道变对、16道变错、23道错到另一错误选项。源需求DOCX/PDF哈希与快照一致。未发现阻塞本次部署的训练或评分技术错误，但没有通过新消融实验确定蒸馏降分原因。

## 实际问题及修复

首次真实启动后，vLLM API和Gradio客户端对话均成功，监督进程正常停止并释放GPU。随后立即重启时，端口预检报“Address already in use”。远端8000/7860均表现为普通bind失败、启用SO_REUSEADDR后成功；本地真实TCP连接回归稳定复现同样错误。

`scripts/deploy.py`的端口检查增加SO_REUSEADDR，使已关闭连接的TIME_WAIT状态不再阻止重启。正在监听的服务仍会被拒绝。两项真实套接字测试先失败后通过，另有9项Pipeline回归通过。原失败会话完整保留，不覆盖成成功。

## 验收范围

验收脚本直接调用`step4_deploy.sh`，使用Python监督进程管理两个后台子进程。不是另行安装supervisord，也不自动重试启动失败。

- vLLM `/health`返回200，`/v1/models`仅包含`week4-dpo-quantized`。
- 发出真实chat completions请求，要求非空完整回答及正确模型身份。
- Gradio首页、配置与`/chat`队列接口可用，Gradio Client请求经过真实UI后端调用vLLM。
- 向监督器发送SIGTERM，核对正常停止、两个端口关闭及GPU空闲。
- 立即重新启动；主动结束本轮Gradio子进程，核对监督器失败状态、vLLM清理及GPU释放。
- 前后核验AWQ完整文件清单和环境pip freeze，确保权重和依赖未被改动。

本轮没有重新做浏览器视觉/按钮交互验收；不将Gradio Client调用冒充浏览器点击。图片后端未启动。服务绑定127.0.0.1，未开放新的公网端口。

## 复现入口

在320机已有环境和模型目录中：

```bash
export WEEK7_SERVING_PYTHON=/root/autodl-tmp/qwen25-week7/envs/serving-vllm064-20260911/bin/python
export WEEK7_UI_PYTHON="$WEEK7_SERVING_PYTHON"
export PIPELINE_PYTHON="$WEEK7_SERVING_PYTHON"
export WEEK7_TEXT_MODEL_PATH=/root/autodl-tmp/qwen25-week7/models/week4-dpo-awq-4bit-g128-20260911
bash scripts/step4_deploy.sh --run-dir /root/autodl-tmp/week8-deploy-manual-01
```

每次使用不存在的新run目录。返回ready后监督器和两个服务在后台运行。停止前核对status.json中的supervisor_pid与实际进程命令，再发送SIGTERM；不要按端口批量杀死未知进程。

原会话：`/root/autodl-tmp/week8-supervised-deployment-20260918`；修复后会话：`/root/autodl-tmp/week8-supervised-deployment-v2-20260918`。本地证据位于`reports/week8/phase4_supervised_deployment/`。

修复后会话于2026-09-18 UTC21:41:47开始、21:43:12结束，正常停止和子进程故障回收均通过；48份证据（含首轮失败记录）已回收，并逐项哈希核对通过。AWQ全部文件与冻结清单一致，pip freeze前后不变；服务已停止，320已关机。

本地共11项测试通过，320另跑2项真实套接字测试通过。证据总包SHA256为`763fde2baf39309ae47928b2cae9c369a40a9126e3141ecb3769eee15e23d672`。复核命令：

```bash
python3 reports/week8/phase4_supervised_deployment/verify_evidence.py 763fde2baf39309ae47928b2cae9c369a40a9126e3141ecb3769eee15e23d672
```

本次完成的是Day41.2部署联调。Day41.3正式Pipeline与评分恢复入口整合、综合Word/PDF终稿和仓库发布仍需继续。
