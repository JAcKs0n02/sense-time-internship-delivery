# 第8周第一阶段任务4：分层内容抽查

完成日期：2026-09-15。状态：**TASK4_SAMPLE_REVIEW_PASS_NOT_DATA_READY**。本次由AI助手逐条阅读和复核，不是新增人工审核；未填写人工签名，未改动历史审核结论或原始正文。

## 结果与覆盖

50条样本、69条助手回答已全部阅读。41条KEEP，9条KEEP_WITH_NOTE，无新增阻断项、无改写、无隔离。数据仍为1422条训练、158条验证，另83条历史验证继续隔离。

| 来源 | 抽查train | 抽查validation | 合计 |
|---|---:|---:|---:|
| alpaca_gpt4_zh | 9 | 5 | 14 |
| coig_pc | 11 | 3 | 14 |
| sharegpt_zh | 19 | 3 | 22 |
| 合计 | 39 | 11 | 50 |

先固定全部13条多轮，再选剩余最长8条、最短6条、代码候选6条、数学候选6条，然后补齐来源/划分覆盖和确定性哈希抽样。样本编号与完整内容在选择后冻结，未因阅读结果而替换不理想样本。代码/数学关键词标签只用于筛选，存在交叉及误命中，不把标签数量当作独立能力验收数量。

这是一份偏重风险的分层样本，不是均匀随机样本，不能用41/50或50/50推算全量准确率。没有本轮截断、拒绝样本或真实system实例，对应抽查标为不适用；system处理此前只做过合成边界测试。

## 已识别并处置的9条提示

| 编号 | 发现 | 处置依据 |
|---|---|---|
| R08 | 中文姓名请求得到英文姓名 | 同一实体且明确包含名字；保留，记录本地化表达较弱 |
| R10 | 记忆化递归使用默认可变字典 | 实际运行前10项及重复调用正确；共享缓存不影响本例纯函数结果，记录工程写法局限 |
| R14 | 电影文章题干残尾 | 主题分类证据充分，Art正确；保留为分类样本，不当作完整文章 |
| R15 | 哲学文章有空括号、缺引用与残尾 | 主题仍明确；不认证题干中的全部史学论断 |
| R16 | 非中文演讲材料断句 | 不影响是否中文的二分类，JSON标签0正确 |
| R17 | 缅甸语材料有片段化语句 | 主要语言判断仍有文字及历史词形审核依据；不声称母语级逐句翻译通过 |
| R18 | 新闻引语在题干末尾中断 | 目标[BBC原文链接](https://www.bbc.com/ukchina/trad/vert-fut-40289371)可访问，主题与人物对应，完成链接定位；不认证全文统计数字 |
| R33 | “如下所示”后没有Node结构说明 | 回答明确限定为“一个可能”的链表节点示例，未声称复现缺失结构；实际构造/读取/连接检查通过 |
| R42 | region和zone均译成“区域” | 两条命令与[Google官方文档](https://docs.cloud.google.com/compute/docs/gcloud-compute)一致；记录应区分区域与可用区，读取的是配置值而非物理位置 |

以上均不是本周格式转换新增的问题。已对照历史逐条审核记录，其中Node示例假设、region/zone译法和长题干片段的可接受边界已有明确说明。本次没有新增证据推翻这些任务级判断，也不为润色文字而混入新修订正文。9条提示都已给出处置理由，无悬而未决的格式或上下文错配。

## 证据与验证

- 50条样本的ShareGPT与Alpaca逐条对齐：system、history、最后问答及原文字面内容一致。全部13条多轮没有丢轮或错接上下文。
- 13项定向代码/数值检查通过：C++计数1至5；两版Python斐波那契；SQLite筛选John；Python排序、求和、集合、Node、表达式、质心和列表连接；JavaScript反转。只执行已逐字检查的本地小例子，结果见[example_checks.json](../reports/week8/phase1_task4/example_checks.json)。这些不等于训练框架测试，也不宣称覆盖全部代码输入边界。
- 链接和外部事实只作定向核验：Spot容量回收机制参照[Azure官方文档](https://learn.microsoft.com/en-us/azure/virtual-machines/spot-vms)；三种官方语言参照[比利时政府](https://www.belgium.be/en/about_belgium/government/federale_staat)；奥巴马任期与身份参照[总统图书馆](https://obamalibrary.archives.gov/obamas/president-barack-obama)。其余摘要、翻译、分类以给定材料及逐条记录说明的证据边界为准。
- 重新核验当前15份JSON、原输入文件及协议哈希。所有字节与任务3验收绑定一致；处理规则和样本归属未变化，因此无受影响的数据产物需要重跑。任务3的全量污染检查和确定性复跑收据继续适用于同一版本。

## 交付文件

- [完整可展开复核包](../reports/week8/phase1_task4/review_pack.html)：50条编号、判断、完整原文，已转义为静态HTML，无远端脚本。
- [逐条机器记录](../reports/week8/phase1_task4/review_records.json)：sample_id、候选行号、来源、划分位置、审核范围、依据和处置。
- [9条提示清单](../reports/week8/phase1_task4/issues.json)：全部已给出保留理由，blocking_issue均为false。
- [抽样清单](../reports/week8/phase1_task4/selection_manifest.json)与[完整样本快照](../reports/week8/phase1_task4/selected_samples.json)。
- [验收收据](../reports/week8/phase1_task4/verification.json)：状态、覆盖、哈希及尚未完成的门禁。

`select_samples.py`只负责确定性抽样，`check_examples.py`只运行已检查的特定例子，`build_review.py`只打包显式写好的逐条判断并检查格式对齐；这些脚本不会自动做出内容审核结论。重跑打包不等于新一轮人工或模型阅读。

## 下一步

任务5：在目标环境用真实LLaMA-Factory加载当前版本，核对样本数、input_ids、attention_mask、labels、assistant监督区间及EOS，重点看13条多轮和长度边界。任务6发布及benchmark/裁判运行版本锁定仍待完成。此次没有启动GPU、训练或更改模型。
