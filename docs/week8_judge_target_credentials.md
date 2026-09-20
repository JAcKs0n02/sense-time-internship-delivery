# Week8 320机裁判凭据核验

2026-09-16（Europe/London），用户明确授权单独传输本机已保存的DeepSeek Key至指定AutoDL 320机。

通过已登录的内嵌浏览器单独上传文件，未将密钥混入代码包、提示词或命令参数。目标机保存位置为 `/root/.config/internship-week8/deepseek.key`，文件权限0600、父目录0700。上传暂存文件已移走。原始rename因数据盘与系统盘跨文件系统失败；随后改用支持跨文件系统的移动，最终检查通过。失败未输出密钥内容。

目标机重新验证了已部署的裁判配置与代码绑定，并成功调用官方 `/models`、`/user/balance` 两个只读接口。返回模型包含deepseek-flash和deepseek-v4-pro；余额显示9.83元。本轮付费评分调用0次，未加载模型或执行训练，未使用GPU。

原始非敏感回执已回收到 `reports/week8/phase2_judge_credentials/target_receipt.json`，SHA256为 `a7bd1e5ee06c8bafd235e56882ddee5e238b3a3a29a48316fe0fbb3698f58993`。回执不包含密钥值或密钥哈希。对相关脚本、配置及本轮报告目录的密钥明文检查通过。

本次解决目标机凭据和只读接口连通性问题，不等同于已完成目标机付费评分，更不代表训练已运行。全局索引已标记credential_verified=true；最终运行锁复核及独立3步SFT/DPO训练smoke仍待执行，training_allowed保持false。

完成全部需调用裁判的任务后，应在DeepSeek平台撤销该Key；仅删除本机或服务器上的文件不会使服务端Key失效。

控制台已确认320机“已关机”；本轮无卡会话结束，实际实例账单未查询。
