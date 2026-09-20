# Day 22：Qwen2-VL 模型准备与图片素材包

## 状态

**已完成（2026-08-18）。** RTX 3090 24GB 环境、锁定 revision 的 Qwen2-VL-7B、17 个模型文件、5 个权重分片、完全离线图文 smoke，以及五类图片素材均已验证。联合验证结果为 `PASS`。

关键结果：

- 模型：`Qwen/Qwen2-VL-7B-Instruct@eed13092ef92e448dd6875b2a00151bd3f7db0ac`；
- 模型目录：17 个普通文件、5 个 safetensors 分片、16,594,403,704 bytes；
- 环境：RTX 3090 24GB、PyTorch `2.5.1+cu121`、Transformers `4.50.0`、`qwen-vl-utils 0.0.14`；
- 离线 smoke：`local_files_only=true`，BF16，输出“湖面上倒映着一座雪山，周围是茂密的森林。”；
- 图片：表格、自然风景、Logo、手写公式、UI 各 1 张，来源、许可、Ground Truth 与 SHA-256 齐全；
- 验证入口：[`day22_validation.json`](source/results/day22_validation.json)。

执行说明：AutoDL 到 `huggingface.co` 的直连超时，因此下载传输使用 `https://hf-mirror.com`；模型身份、不可变 revision、官方仓库 URL 和逐文件哈希仍按 Hugging Face 官方仓库固定。首次 smoke 的 OOM 并非权重损坏，而是 Transformers 4.50 仅修改 `min_pixels/max_pixels` 属性时没有改变生效的 `size`；改为显式设置 `size.shortest_edge/longest_edge` 后，输入从 16,258 token 降至 418 token，并在 BF16 下成功完成。

执行依据：[第 5 周完整执行计划](../../../docs/week5_execution_plan.md) · [Week 5 工程索引](../README.md)

## 老师要求

1. 根据显存选择 Qwen2-VL：8GB 使用 `Qwen/Qwen2-VL-2B-Instruct`，16GB 及以上使用 `Qwen/Qwen2-VL-7B-Instruct`；
2. 下载模型并安装 `qwen-vl-utils`；
3. 准备 5 张图片：表格截图、自然风景、Logo、手写公式和 UI 界面；
4. 交付模型下载确认与图片素材包。

既有 AutoDL GPU 为 RTX 3090 24GB，因此正式路线固定为：

```text
Qwen/Qwen2-VL-7B-Instruct
```

实际已通过 `nvidia-smi` 验证 GPU 为 NVIDIA GeForce RTX 3090，显存 24576 MiB。

## 当日交付

- 固定不可变 revision 的模型下载确认；
- Python、CUDA、PyTorch、Transformers、Accelerate、`qwen-vl-utils` 等环境清单；
- 模型普通文件清单、文件数、总字节和逐文件 SHA-256；
- `local_files_only=True` 的离线加载记录与一条图文 smoke；
- 五张图片及 `image_manifest.csv`；
- 每张图片的来源、许可、尺寸、SHA-256 和人工 ground truth；
- 数量、格式、哈希和敏感信息检查结果。

## 专业术语

| 术语 | 解释 |
|---|---|
| VLM | Vision-Language Model，同时理解图像与文本并生成文本回答的视觉语言模型 |
| Model Hub | 托管模型配置、权重、processor 和版本历史的仓库；本项目使用 Hugging Face 官方模型仓库 |
| Revision | 模型仓库的版本标识；最终记录使用不可变 commit，而不是会变化的 `main` |
| Processor | 把对话、图片和视频转换为模型张量的预处理组件 |
| 动态分辨率 | Qwen2-VL 根据图像尺寸产生不同数量的视觉 token；会影响显存、速度和注意力网格 |
| `min_pixels` / `max_pixels` | 限制图像预处理像素范围的参数；必须在正式实验前固定 |
| 离线加载 | 不访问网络，只从本地目录加载模型；可证明下载文件足以复现 |
| Smoke test | 用一条最小输入验证模型、processor、设备和生成链路能工作，不代表正式实验完成 |
| Manifest | 文件清单及其路径、大小、来源、版本和哈希，用来审计数据血缘 |
| SHA-256 | 根据文件内容计算的哈希；文件内容变化时哈希也会变化 |
| Ground truth | 人工核验的图片真实内容，供后续 OCR、推理和幻觉判断使用 |

## 固定输入

### 模型

| 字段 | 固定规则 |
|---|---|
| repository | `Qwen/Qwen2-VL-7B-Instruct` |
| revision | `eed13092ef92e448dd6875b2a00151bd3f7db0ac` |
| trust_remote_code | `false`；使用 Transformers 原生 `Qwen2VLForConditionalGeneration` |
| dtype | `bfloat16` |
| image limits | smoke 使用 `min_pixels=200704`、`max_pixels=301056`；有效网格为 `1×32×48` |
| random seed | `42` |

### 五张图片

| image_id | 类型 | 内容与来源约束 | Ground truth 要点 |
|---|---|---|---|
| `w5-table-01` | 表格截图 | Wikimedia Commons，Loz.ross，CC BY-SA 4.0 | 单元格文字、行列关系和占位内容 |
| `w5-scene-01` | 自然风景 | Wikimedia Commons，Bonnie Moreland，CC0 1.0 | 山、森林、湖面、倒影和可见天气 |
| `w5-logo-01` | Logo | Wikimedia Community Logo，Artur Jan Fijalkowski，Public domain | 颜色、几何形状和无文字事实 |
| `w5-formula-01` | 手写公式 | Wikimedia Commons，Public domain（PD-text） | 公式精确转写、符号和未知上下文 |
| `w5-ui-01` | UI 界面 | LibreOffice Writer UI，vmiklos / TDF，CC0 1.0 | 控件、文字、评论框、状态栏和层级 |

图片进入 Day 23 正式评测后即冻结。任何像素变化都必须重新计算 SHA，并使用新的 `image_id` 或明确版本号。

## 详细实施步骤

### 1. 预检 GPU、磁盘和环境

记录 `nvidia-smi`、持久化磁盘余量、Python、pip/conda、PyTorch 与 CUDA。验证实际显存至少 16GB 才继续 7B 路线；同时确认模型目录位于 `/root/autodl-tmp/` 的 Week 5 专用持久化路径，而不是易丢失的系统盘临时目录。

### 2. 锁定模型 revision

从 `Qwen/Qwen2-VL-7B-Instruct` 官方仓库解析当前 commit，将仓库名和 commit 同时保存到模型 manifest。下载和后续加载都显式使用该 revision，不把 `main` 作为唯一血缘。

### 3. 建立隔离环境并安装依赖

创建 Week 5 专用或明确复用的 Python 环境，安装与 Qwen2-VL 兼容的 PyTorch、Transformers、Accelerate、`qwen-vl-utils`、Pillow 等包。保存 `python --version`、`pip freeze` 和关键包的 `__version__`；不在文档中预写未经验证的版本号。

### 4. 下载并核对模型

下载 processor、tokenizer、配置与全部权重分片。遍历普通文件，保存相对路径、字节数和 SHA-256；另存文件总数与总字节。下载中断时从官方 revision 恢复，不混入其他 revision 的缓存文件。

### 5. 离线加载与图文 smoke

使用本地路径和 `local_files_only=True` 加载 processor/model。选一张非敏感图片运行一条简短描述题，记录输入、原始输出、return code、加载时长、生成时长和峰值显存。Smoke 只证明链路可用，不代替 Day 23 的 25 条正式推理。

### 6. 创建五张素材

按固定 ID 下载五张许可明确的 Wikimedia Commons 素材，保留版本化来源页、作者、许可、下载地址和 attribution。逐张检查像素内容与元数据；公式 JPEG 含普通 EXIF，但未发现 GPS 字段，其余图片无可识别 EXIF 隐私信息。

### 7. 建立 ground truth

逐张人工记录可见对象、精确文字、数值、关系、布局和不能从图中确定的内容。Ground truth 在 Day 23–Day 25 运行前冻结，避免根据模型回答倒推正确答案。

### 8. 生成并验证素材清单

`image_manifest.csv` 字段固定为：

```text
image_id,relative_path,image_type,source,license,width,height,file_bytes,sha256
```

检查五个 `image_id` 唯一、五种 `image_type` 各出现一次、文件可由 Pillow 解码、宽高与字节数一致、SHA 可重算、路径均在素材包内部。

### 9. 记录结果并更新状态

真实模型 revision、环境、文件清单、离线 smoke 和素材验证已写入工程归档，并从通过门禁的内容生成 `Submission/Week5/Day22_Model_and_Image_Preparation/`。

### 10. 结束 AutoDL 会话

必要的小型证据同步并校验后已关闭 AutoDL 实例。关机状态只用于内部运维确认，不进入老师的 `Submission/Week5/`。

## 实际文件结构

```text
deliverables/week5/day22/
├── README.md
└── source/
    ├── data/images/
    │   ├── w5-table-01.*
    │   ├── w5-scene-01.*
    │   ├── w5-logo-01.*
    │   ├── w5-formula-01.*
    │   └── w5-ui-01.*
    ├── manifests/
    │   ├── image_manifest.csv
    │   ├── image_ground_truth.json
    │   ├── image_sources.json
    │   └── model_selection.json
    ├── results/
    │   ├── remote_evidence/
    │   │   ├── environment.json
    │   │   ├── model_manifest.json
    │   │   ├── offline_load_smoke.json
    │   │   └── remote_status.json
    │   └── day22_validation.json
    ├── scripts/
    │   ├── build_image_manifest.py
    │   ├── prepare_day22_remote.py
    │   ├── run_day22_remote.sh
    │   ├── run_vlm_smoke.py
    │   └── validate_day22.py
    └── tests/
        ├── test_day22_assets.py
        ├── test_day22_model_tools.py
        └── test_run_vlm_smoke.py
```

模型大权重、缓存和 checkpoint 只保存在远端持久化目录，不复制到 Git 或教师 ZIP。

## 完成门禁

- [x] 实际 GPU 型号和显存已重新验证，满足 7B 路线；
- [x] 官方模型仓库和不可变 revision 已记录；
- [x] 环境版本清单来自实际运行环境；
- [x] 模型文件清单、总大小和 SHA-256 可重算；
- [x] `local_files_only=True` 离线加载 return code 为 0；
- [x] 一条图文 smoke 原始输入/输出与峰值显存已保存；
- [x] 五种图片类型各 1 张，ID 唯一且全部可解码；
- [x] 图片来源、许可、尺寸、字节、SHA 和 ground truth 齐全；
- [x] 敏感信息与许可检查通过；
- [x] AutoDL 使用结束后已关机，且教师材料未包含关机证据。

任一项没有证据时，Day 22 保持未完成。

## 故障处理

- **下载失败**：保留错误日志，确认官方仓库与 revision 后续传；不从来源不明的网盘替换权重。
- **模型文件哈希变化**：检查是否混入多个 revision；清理目标模型的具体残缺分片后按锁定 revision 恢复，不删除其他项目数据。
- **CUDA OOM**：先降低 `max_pixels`、使用 bf16/4-bit 与更短 smoke 输出；所有变更写入配置并重新生成 SHA。
- **包版本冲突**：保存 `pip check` 和导入错误，建立干净环境后按官方兼容组合安装；不覆盖原失败日志。
- **图片许可不明确**：在运行模型前替换为自建/自拍素材，并重新冻结 ID、来源和 ground truth。
- **图片含敏感信息**：重建素材，不仅依赖模糊化；检查像素内容、文件名和 EXIF。

## 下一日交接

Day 23 只能接收以下已经冻结的输入：模型仓库与 revision、processor/生成配置 SHA、五张图片及其 SHA、`image_manifest.csv` 和 `image_ground_truth.json`。任一项变化都必须使 Day 23 的旧结果失效并完整重跑。
