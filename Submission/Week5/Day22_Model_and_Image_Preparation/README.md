# Day 22：Qwen2-VL 模型下载确认与图片素材包

## 完成情况

Day 22 已完成并通过验证：根据 RTX 3090 24GB 显存选择 `Qwen/Qwen2-VL-7B-Instruct`，完成模型下载、依赖安装、离线图文推理验证，并准备表格截图、自然风景、Logo、手写公式和 UI 界面五类图片各 1 张。

## 模型下载确认

| 项目 | 实际结果 |
|---|---|
| 模型仓库 | `Qwen/Qwen2-VL-7B-Instruct` |
| 不可变 revision | `eed13092ef92e448dd6875b2a00151bd3f7db0ac` |
| GPU | NVIDIA GeForce RTX 3090，24576 MiB |
| 模型普通文件 | 17 个 |
| 权重分片 | 5 个 safetensors |
| 模型目录总大小 | 16,594,403,704 bytes |
| 模型文件清单摘要 SHA-256 | `575d62b8abe01fda09b28a02ff8e6cd04e8df0d15fd03e1642d3a22bbe7eb812` |
| PyTorch / Transformers | `2.5.1+cu121` / `4.50.0` |
| `qwen-vl-utils` | `0.0.14` |
| 离线加载 | PASS，`local_files_only=true` |
| 离线 smoke 输出 | 湖面上倒映着一座雪山，周围是茂密的森林。 |

AutoDL 网络传输使用 Hugging Face 镜像端点，但模型身份、不可变 revision、官方仓库地址和每个文件的 SHA-256 均记录在下载确认表中。模型权重体积约 16.6GB，因此本提交提供完整文件清单与哈希，不重复打包大权重。

## 五张图片

| 图片 ID | 类型 | 作者 / 来源 | 许可 |
|---|---|---|---|
| `w5-table-01` | 表格截图 | Loz.ross / Wikimedia Commons | CC BY-SA 4.0 |
| `w5-scene-01` | 自然风景 | Bonnie Moreland / Wikimedia Commons | CC0 1.0 |
| `w5-logo-01` | Logo | Artur Jan Fijalkowski (WarX) / Wikimedia Commons | Public domain |
| `w5-formula-01` | 手写公式 | Unknown author，Penarc 上传 / Wikimedia Commons | Public domain (PD-text) |
| `w5-ui-01` | UI 界面 | vmiklos / The Document Foundation / Wikimedia Commons | CC0 1.0 |

详细来源页、下载地址、作者、许可链接、尺寸、字节数和 SHA-256 分别见 `Manifests/Image_Sources.json` 与 `Manifests/Image_Manifest.csv`。五张图均完成隐私和许可复核，结果为 PASS。

## 文件说明

```text
Day22_Model_and_Image_Preparation/
├── README.md
├── Images/                         # 五类图片素材
├── Manifests/
│   ├── Image_Manifest.csv          # 类型、来源、许可、尺寸、字节与 SHA-256
│   ├── Image_Ground_Truth.json     # 人工核验的可见内容与不可推断项
│   └── Image_Sources.json          # 版本化来源、作者、许可及 attribution
└── Results/
    ├── Environment.json            # 实际 GPU 与关键依赖版本
    ├── Model_Download_Confirmation.json
    ├── Offline_Load_Smoke.json
    └── Day22_Validation.json       # 图片与远端证据联合验证，PASS
```

## 验证结论

`Day22_Validation.json` 中 9 项检查全部为 PASS：清单 schema、五类数量、图片完整性、Ground Truth、许可与隐私、远端环境、模型清单、离线 smoke 和远端证据一致性。
