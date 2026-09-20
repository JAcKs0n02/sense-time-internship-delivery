# Week8：320机提示部署复验与最小forward

2026-09-15。本轮仅进行目标部署、完整提示检查、模型身份复核与最小forward，不运行训练或正式benchmark计分。**目标提示、正式配置及最小forward均通过**，收据见`reports/week8/phase2_target_runtime`。

## 会话及传输

320机`be044ebe99-be706b14`在约13:40 UTC启动，页面确认价格为¥1.48/小时，预算最多1小时。设置服务器15:35（UTC14:35）自动关机保护，计划工作完成即提前关机。开机时GPU计算进程列表为空。

传输包为`week8-target-runtime-20260915.tar.gz`，9,253,110字节，SHA-256为`885dd623e583b23d2c2b2bde78b1e0032150ecfda943f5055b3e3f336d16a46b`。远端先校验整包哈希，再解包到独立目录`/root/autodl-tmp/week8-target-runtime-20260915`，311文件逐个哈希一致。因终端显示延迟重复提交了解包指令，第二次被目录已存在保护拦截，没有覆盖第一次结果。

## 发现问题与修复

首次目标模板测试导入失败：目标环境存在其他`scripts`包，而本仓库原先只是namespace package。新增`scripts/__init__.py`明确包归属，并纳入配置依赖校验；配置构建器同步更新。初始失败日志`template_tests.log`与通过日志`template_tests_02.log`分别保留。本地更新后38项测试通过，远端4项模板回归测试通过（包括11条实际受影响题）。两处部署增量哈希见`deployment_delta.json`。

## 已观察通过的关卡

- Linux真实OpenCompass提示核验：119科、12928题、每题5示例，逐题token哈希与独立预期一致；最长1361 token，无截断变化。提示行输出SHA-256与本地完全相同：`1cb1c25c2a80e1b407c7947840e94d913cfe2193c179be41acde33058f310bc5`。
- 在目标目录重新生成正式配置，并通过实际CLI dry-run，退出码0。不会使用含本地Mac路径的候选配置。

## forward检查范围

`scripts/check_week8_target_forward.py`限定320主机且要求Linux完整提示通过收据。重新对照此前原始基座15个文件的SHA-256，包括4个权重分片；实际加载BF16模型并检查所有参数位于CUDA；用一条短提示检查logits形状与数值有限性，再通过真实OpenCompass包装器进行最多2 token的生成检查。

这一检查不代表完整benchmark通过，不验证训练反向传播、优化器或SFT/DPO质量，不产生可与历史分数比较的新模型指标。裁判身份未补齐，正式运行锁继续关闭。

## 最终结果与证据回收

最小forward退出码0。输入35 token，logits形状为[1,35,152064]，数值全部有限；全部参数在CUDA且为BF16，未量化。真实OpenCompass包装器最多2 token生成返回“你好！”。峰值PyTorch allocated显存15,278,057,984字节（约14.23 GiB）；这是短提示检查，不表示完整评测或训练的峰值显存。配置、实际dry-run配置与已审核提示的数据配置一致，304个配置依赖校验通过。

原始提示收据和forward收据已回收到本地，重新序列化后逐字节SHA-256分别与远端一致：

- `runtime-01/verification.json`：b7cf16519cca4526509414721accd7bd6ebab1670f67cc657ab7ec2b428ca025
- `forward.json`：78032a393246047aff32b43e6dc5f3f8242d425be4c4849044cf9fcb6576b59f
- 完整传输收据`receipt_transport.json`：5f8100ee61cf2910329ab29ee56f9341a92b0bdffa3f4614d5e1758e0273e84d

目标配置也已按目标路径在本地重建，字节哈希与远端一致，保存为`remote_opencompass_formal.py`，SHA-256为891ce38524a3050fc439e5e9294f367a0bbeed7da21f6a7b5646bfdf9cb88aa7。完整原始日志、提示行及配置另外打包在远端`/root/autodl-tmp/week8-target-runtime-20260915/runtime_evidence.tar.gz`；该完整压缩包尚未下载，本地收据记录了各原始日志的哈希，不将哈希记录冒称日志副本。

约13:50 UTC提前关闭320机，页面确认“已关机”；关机前GPU计算进程为空。本次约11分钟，未查询实际账单。监控保持暂停，不自动再次开机。先前正式配置候选因新增包标记而升版，新候选保存于本轮`local_candidate`，全局索引已更新。

下一步补齐裁判准确模型名称/版本及base URL，汇总运行锁并复核放行条件；之后才进入独立3步SFT/DPO smoke。现阶段正式计分、裁判调用和训练均未执行。
