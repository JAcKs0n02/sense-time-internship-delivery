#!/usr/bin/env python3
"""Build the manually re-audited Day26 v7 image fact cards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TRAIN_TRANSCRIPTIONS = {
    "ocr-01": ("票面顶部的英文标题", "BOARDING PASS"),
    "ocr-02": ("上排中间招牌的蓝色大字", "BLUE LAGOON"),
    "ocr-03": ("五个手写汉字，从左到右", "放、澳、吧、把、板"),
    "ocr-04": ("左图下方蓝绿色标牌上的三字机构名", "司法所"),
    "ocr-05": ("左图底部 GT 标注后的店名", "依佳人网咖"),
    "ocr-06": ("图中可确认反复出现的一个小写拉丁字母", "d"),
    "ocr-07": ("红色 GT 标注后的车牌文本", "皖A533Y3"),
    "ocr-08": ("文本行中清晰连续的六个汉字", "旅游表演形式"),
    "ocr-09": ("整行可见文本", "4年工作报告》中指出"),
    "ocr-10": ("招牌上的全部大字", "魅派集成吊顶"),
    "formula-01": (
        "整条公式",
        r"\mathcal{L}_{eyelid}=\sum_{t=1}^{T}\sum_{v=1}^{V}\mathcal{M}_{v}^{(eyelid)}\left(\lVert\hat{h}_{t,v}-x_{t,v}\rVert^2\right)",
    ),
    "formula-02": (
        "整条有理式分解等式",
        r"\frac{7x^2-x+1}{x^3+1}=\frac{7x^2-x+1}{(x+1)(x^2-x+1)}=\frac{A}{x+1}+\frac{Bx+C}{x^2-x+1}",
    ),
    "formula-03": (
        "整条复变函数展开式",
        r"F(z)=F(0)+\sum_{m=1}^{\infty}\left\{\frac{b_m}{z-a_m}+\frac{b_m}{a_m}\right\}+\sum_{m=1}^{\infty}\left\{\frac{b_{-m}}{z-a_{-m}}+\frac{b_{-m}}{a_{-m}}\right\}",
    ),
    "formula-04": (
        "整条红色损失函数",
        r"\mathcal{L}_t=\mathbb{E}_{\mathbf{x}_0,\epsilon}\left[\frac{(1-\alpha_t)^2}{2\alpha_t(1-\bar{\alpha}_t)\lVert\Sigma_\theta\rVert_2^2}\left\lVert\epsilon_t-\epsilon_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0+\sqrt{1-\bar{\alpha}_t}\epsilon_t,t)\right\rVert^2\right]",
    ),
    "formula-05": (
        "三行方程组",
        r"\begin{cases}25a-5b+c=1\\a-b+c=1\\9a-3b+c=0\end{cases}",
    ),
    "formula-06": (
        "四行四列矩阵",
        r"\begin{bmatrix}0&1&0&0\\0&0&\frac{m^2gl^2}{Mml^2}&0\\0&0&0&1\\0&0&-\frac{mgl(M+m)}{Mml^2}&0\end{bmatrix}",
    ),
    "formula-07": ("整条手写积分式", r"s_2-s_1=\int_{t_1}^{t_2}v\,dt"),
    "formula-08": (
        "等号右侧的清晰积分",
        r"\int_{-\infty}^{+\infty}e^{-ts^2}\,ds",
    ),
    "formula-09": (
        "两行常微分方程组",
        r"\begin{cases}\frac{dy}{dx}+y=3\\\frac{d^2y}{dx^2}+3\left(\frac{dy}{dx}\right)^3+y^3=0\end{cases}",
    ),
    "formula-10": (
        "整条极限式",
        r"\lim_{x\to\frac14}\frac{1-4^{x-\frac14}}{1-4x}",
    ),
}


HELDOUT_REVIEW = {
    "natural_scene-v7-dev-03": {
        "subject": "雪山、针叶林与湖泊构成的自然风景",
        "facts": [
            "远处雪山位于画面左侧",
            "湖面映出雪山和树木的倒影",
            "右侧近岸可见岩石、枯木和茂密针叶树",
        ],
        "claim": "湖面可以看到雪山的倒影。",
        "supported": True,
        "evidence": "雪山轮廓在湖面左下区域形成清晰倒影。",
    },
    "natural_scene-v7-dev-04": {
        "subject": "秋季林木间的一条狭窄土路",
        "facts": [
            "土路从画面底部向树林深处延伸",
            "左侧一棵树的叶片呈明显黄色",
            "周围分布多棵高大树木和绿色灌木",
        ],
        "claim": "林间小路铺有平整的黑色柏油。",
        "supported": False,
        "evidence": "路面是覆有落叶的土路，没有可见的黑色柏油铺装。",
    },
    "natural_scene-v7-final-03": {
        "subject": "有多棵高大棕榈树的沙滩海景",
        "facts": [
            "多棵棕榈树集中在画面右半部",
            "前景是浅色沙滩，左后方可见海面",
            "远处有低矮山体轮廓，天空整体明亮",
        ],
        "claim": "沙滩右侧可以看到多棵高大的棕榈树。",
        "supported": True,
        "evidence": "画面右侧有数棵树干细长、树冠呈扇形的棕榈树。",
    },
    "natural_scene-v7-final-04": {
        "subject": "两只在雪地上朝镜头跑来的犬",
        "facts": [
            "前后各有一只犬，均面向镜头",
            "两只犬的四肢处于奔跑姿态",
            "地面覆盖白雪，背景可见树木和灌丛",
        ],
        "claim": "两只犬都坐在雪地上。",
        "supported": False,
        "evidence": "两只犬的腿部均处于运动姿态，正在朝镜头跑来，并未坐下。",
    },
    "ocr-v7-dev-01": {
        "subject": "摊开在地图上的英文书籍第 94 和 95 页",
        "facts": [
            "左右两页均排有多段英文正文",
            "左页中部有一幅黑白漫画插图",
            "右页包含多行分式与乘法算式",
        ],
        "scope": "左页正文的粗体小标题",
        "transcription": "The disappearing sum",
        "claim": "左页粗体小标题写着“The disappearing sum”。",
        "supported": True,
        "evidence": "左页正文上方可直接读到粗体标题“The disappearing sum”。",
    },
    "ocr-v7-dev-02": {
        "subject": "带有五角星和镰刀锤头图案的红色圆形印章",
        "facts": [
            "印章外圈排列红色环形文字",
            "中央是红色镰刀锤头图案",
            "图案下方有四个较大的红色汉字",
        ],
        "scope": "印章中央图案下方的四个大字",
        "transcription": "发票专用章",
        "claim": "印章中央图案下方写着“合同专用章”。",
        "supported": False,
        "evidence": "该位置清楚写的是“发票专用章”，不是“合同专用章”。",
    },
    "ocr-v7-final-01": {
        "subject": "深色底上的一行浅色中文招牌文字",
        "facts": [
            "文字横向排列且占据几乎整个画面宽度",
            "背景为深绿黑色，字形为灰金色",
            "末尾两个字为“公寓”",
        ],
        "scope": "整行招牌文字",
        "transcription": "绿洲仕格维花园公寓",
        "claim": "招牌末尾两个字是“公寓”。",
        "supported": True,
        "evidence": "整行文字可转写为“绿洲仕格维花园公寓”，末尾确为“公寓”。",
    },
    "ocr-v7-final-02": {
        "subject": "深色终端界面的三行等宽字体命令与日志文本",
        "facts": [
            "第一行以 python 开头并包含 scripts 路径",
            "第二行包含 use model 和用户目录路径",
            "第三行包含 senet_lite_136 与 epoch=039",
        ],
        "scope": "第一行清晰可见的命令",
        "transcription": "python scripts/screenshot_daemon_with_server",
        "claim": "第一行命令以“pip install”开头。",
        "supported": False,
        "evidence": "第一行清楚以“python”开头，后接 scripts/screenshot_daemon_with_server。",
    },
    "chart_table-v7-dev-01": {
        "subject": "2023 年预期寿命与人均卫生支出的散点图",
        "facts": [
            "横轴是人均卫生支出，纵轴是预期寿命",
            "美国数据点位于最右侧，支出超过 12000 美元",
            "日本数据点位于约 84 年附近，颜色表示洲别",
        ],
        "claim": "美国的数据点位于所有标注国家的最右侧。",
        "supported": True,
        "evidence": "标注为 United States 的点在横轴最右端，明显超过其他国家。",
    },
    "chart_table-v7-dev-02": {
        "subject": "多国人均二氧化碳排放的历史折线图",
        "facts": [
            "横轴覆盖约 1750 年至 2024 年",
            "纵轴单位为每人吨数，图中有多条国家或地区曲线",
            "美国紫色曲线在二十世纪后期曾高于 20 吨",
        ],
        "claim": "中国曲线在 2024 年末低于每人 1 吨。",
        "supported": False,
        "evidence": "2024 年末中国曲线约在 8 至 9 吨附近，明显高于 1 吨。",
    },
    "chart_table-v7-final-01": {
        "subject": "2025 年各国一次能源中可再生能源占比的世界地图",
        "facts": [
            "地图以由浅至深的绿色表示占比区间",
            "图例范围从 0% 延伸至 70% 以上",
            "斜线填充区域在图例中标注为 No data",
        ],
        "claim": "斜线填充区域代表没有数据。",
        "supported": True,
        "evidence": "地图下方图例把斜线填充样式明确标为“No data”。",
    },
    "chart_table-v7-final-02": {
        "subject": "1950 至 2023 年世界人口年龄组堆叠面积图",
        "facts": [
            "横轴从 1950 年延伸到 2023 年",
            "右侧依次标注五个年龄组",
            "蓝色的 25–64 岁区域在末端最宽",
        ],
        "claim": "2023 年时 25–64 岁年龄组是图中最小的区域。",
        "supported": False,
        "evidence": "2023 年末蓝色 25–64 岁区域最宽，并非最小。",
    },
    "ui-v7-dev-01": {
        "subject": "PowerToys Advanced Paste 的弹出操作面板",
        "facts": [
            "顶部输入框提示可格式化为 C# 字符串定义",
            "面板列出 Paste as plain text、Paste as markdown 和 Paste as JSON",
            "底部还有 Clipboard history 入口",
        ],
        "claim": "面板提供“Paste as plain text”选项。",
        "supported": True,
        "evidence": "选项列表第一项清楚显示“Paste as plain text”。",
    },
    "ui-v7-dev-02": {
        "subject": "PowerToys FancyZones 的布局模板选择窗口",
        "facts": [
            "Templates 区域展示 Focus、Columns、Rows 等多个模板",
            "Custom 区域展示三个自定义布局缩略图",
            "右下角有 Create new layout 按钮",
        ],
        "claim": "窗口中只显示一个可选布局模板。",
        "supported": False,
        "evidence": "Templates 与 Custom 区域合计显示多种布局，远不止一个。",
    },
    "ui-v7-final-01": {
        "subject": "PowerToys Settings 中的 Color Picker 设置页",
        "facts": [
            "Enable Color Picker 开关显示为 On",
            "激活快捷键显示为 Windows、Shift 和 C",
            "下方列出 HEX、RGB、HSL、HSV 和 CMYK 格式",
        ],
        "claim": "Color Picker 当前处于启用状态。",
        "supported": True,
        "evidence": "Enable Color Picker 一行右侧显示 On，开关呈蓝色启用状态。",
    },
    "ui-v7-final-02": {
        "subject": "PowerToys Settings 的 Home 总览页",
        "facts": [
            "左侧导航含 Home、General 和多个工具分类",
            "中间 Quick access 区域提供多个快捷入口",
            "右侧 Modules 列表中多项开关处于开启状态",
        ],
        "claim": "Modules 列表中的所有模块开关都处于关闭状态。",
        "supported": False,
        "evidence": "Advanced Paste、Always On Top、Awake、Color Picker 等多项开关明显处于开启状态。",
    },
    "formula-v7-dev-01": {
        "subject": "含多条编号公式的双栏英文学术论文页面",
        "facts": [
            "页面分为左右两栏，公式编号从 (4) 延伸到 (14)",
            "左栏上部的式 (4) 含有 phi、delta 和 psi",
            "右栏中部的式 (11) 含 log10 M 与 log10 X",
        ],
        "scope": "左栏编号 (4) 的公式",
        "transcription": r"\phi(L,t)d\log_{10}L=\delta(M,t)\psi(M,t)dM",
        "claim": "编号 (4) 的公式左侧含有 phi(L,t) 与 d log10 L。",
        "supported": True,
        "evidence": "式 (4) 左侧清楚写有 phi(L,t)d log10 L。",
    },
    "formula-v7-dev-02": {
        "subject": "介绍 dVAE 训练损失的中文说明与居中公式",
        "facts": [
            "顶部中文说明提到 dVAE、VQ-VAE 和 KL 距离",
            "居中公式由负对数似然项与 KL 项相加",
            "底部文字提到 Gumbel-Softmax 与 p(z)",
        ],
        "scope": "居中的完整损失公式",
        "transcription": r"-\mathbb{E}_{z\sim q(z\mid x)}[\log(p(x\mid z))]+KL(q(z\mid x)\Vert p(z))",
        "claim": "居中的损失公式完全不包含 KL 项。",
        "supported": False,
        "evidence": "公式右半部分明确包含 KL(q(z|x)||p(z))。",
    },
    "formula-v7-final-02": {
        "subject": "介绍普通卷积与空洞卷积的双栏论文页面",
        "facts": [
            "左上角有输入图、密度图和预测图三幅示例",
            "右栏上部列出编号 (1) 和 (2) 的两个求和公式",
            "编号 (1) 的求和条件写在求和号下方",
        ],
        "scope": "右栏编号 (1) 的公式",
        "transcription": r"(F*k)(\mathbf{p})=\sum_{\mathbf{s}+\mathbf{t}=\mathbf{p}}F(\mathbf{s})k(\mathbf{t})",
        "claim": "编号 (1) 的求和条件写作 s+t=p。",
        "supported": True,
        "evidence": "式 (1) 的求和号下方清楚标出 s+t=p。",
    },
    "formula-v7-final-03": {
        "subject": "用彩色面积块展示配方法的二次方程示意图",
        "facts": [
            "左侧蓝色正方形标记为 x²",
            "两个橙色长方形均标有 b/(2a) 与 x",
            "右侧红色区域标记为 -c/a，绿色小正方形标记为 (b/(2a))²",
        ],
        "scope": "图中四类数学标签",
        "transcription": r"x^2;\ \frac{b}{2a}x;\ -\frac{c}{a};\ \left(\frac{b}{2a}\right)^2",
        "claim": "右侧红色区域中的标签是正的 c/a。",
        "supported": False,
        "evidence": "红色区域标签前有清晰负号，写作 -c/a。",
    },
}


def _gold(category: str, scope: str | None, value: str | None) -> dict:
    if category in {"ocr", "formula"}:
        if not scope or not value:
            raise ValueError(f"missing scoped transcription for {category}")
        return {
            "applicable": True,
            "scope": scope,
            "value": value,
            "review_status": "double_checked",
            "ambiguous_tokens": [],
        }
    return {
        "applicable": False,
        "scope": "not_applicable",
        "value": "",
        "review_status": "not_applicable",
        "ambiguous_tokens": [],
    }


def _prompts(split: str, category: str, scope: str | None) -> dict[str, str]:
    direct_prefix = {
        "train": "请严格依据当前图片作答",
        "dev": "请客观读取这张开发集图片",
        "final": "请独立分析这张最终测试图片",
    }[split]
    if category == "ocr":
        direct = f"{direct_prefix}：只转写{scope}，不要补写模糊或被裁切的其他文字。"
    elif category == "formula":
        direct = f"{direct_prefix}：用 LaTeX 转写{scope}，保留可见的符号、上下标和正负号。"
    elif category == "chart_table":
        direct = f"{direct_prefix}：概括图表主题、编码方式和两项可直接核对的数据关系。"
    elif category == "ui":
        direct = f"{direct_prefix}：说明界面名称、主要区域以及当前可见状态。"
    else:
        direct = f"{direct_prefix}：描述主体、背景与相对位置，不猜测地点、身份或拍摄信息。"

    verification = {
        "train": "请核验下列训练陈述；仅给出“符合”或“不符合”，并引用直接可见证据。",
        "dev": "请核对下列开发集陈述；给出结论并说明图中依据，不使用外部知识。",
        "final": "请审查下列最终测试陈述；先判断是否符合，再给出最短充分视觉依据。",
    }[split]
    prompts = {"direct": direct, "verification": verification}
    if split == "train":
        prompts.update(
            {
                "evidence": "请列出三项彼此不同、能从图片直接观察到的证据，不加入常识推断。",
                "grounding": "请区分图片中能够确认的信息与没有显示、因而不能确认的信息。",
            }
        )
    return prompts


def _answers(
    split: str,
    category: str,
    subject: str,
    facts: list[str],
    uncertainty: list[str],
    transcription: str | None,
    supported: bool,
    verification_evidence: str,
) -> dict:
    if category in {"ocr", "formula"}:
        direct = f"指定区域的转写为：{transcription}"
    else:
        direct = f"{subject}。{facts[0]}；{facts[1]}；{facts[2]}。"
    answers = {
        "direct": direct,
        "verification": {
            "verdict": "supported" if supported else "unsupported",
            "evidence": verification_evidence,
        },
    }
    if split == "train":
        answers.update(
            {
                "evidence": "；".join(
                    f"证据 {index}：{fact}" for index, fact in enumerate(facts, start=1)
                )
                + "。",
                "grounding": (
                    f"能够确认：{subject}，并且{facts[1]}。"
                    f"不能确认：{'；'.join(uncertainty)}"
                ),
            }
        )
    return answers


def _base_from_manifest(row: dict) -> dict:
    keys = (
        "image_id",
        "split",
        "category",
        "image_relative_path",
        "source_page_url",
        "download_url",
        "source_revision",
        "source_revision_kind",
        "author",
        "license",
        "license_url",
        "sha256",
    )
    return {key: row[key] for key in keys}


def build_curated_facts(manifest: list[dict], legacy_facts: list[dict]) -> list[dict]:
    """Join frozen provenance with the second-pass human annotations."""

    legacy_by_id = {
        row["image_id"]: row for row in legacy_facts if row.get("split") == "train"
    }
    manifest_heldout = {
        row["image_id"] for row in manifest if row["split"] in {"dev", "final"}
    }
    if manifest_heldout != set(HELDOUT_REVIEW):
        missing = sorted(manifest_heldout - set(HELDOUT_REVIEW))
        extra = sorted(set(HELDOUT_REVIEW) - manifest_heldout)
        raise ValueError(f"heldout review inventory mismatch; missing={missing}, extra={extra}")

    facts = []
    for row in manifest:
        base = _base_from_manifest(row)
        image_id = row["image_id"]
        if row["split"] == "train":
            old = legacy_by_id.get(image_id)
            if old is None:
                raise ValueError(f"missing legacy train review for {image_id}")
            subject = old["subject"]
            observable = [f"画面主体或版面为：{subject}", *old["details"]]
            scope, transcription = TRAIN_TRANSCRIPTIONS.get(image_id, (None, None))
            supported = int(image_id.rsplit("-", 1)[1]) % 2 == 1
            claim = (
                f"图中显示的是{subject}。" if supported else old["false_claim"]
            )
            verification_evidence = (
                f"画面主体或版面确为{subject}，且{old['details'][0]}。"
                if supported
                else f"图中实际显示的是{subject}，与待核验陈述不一致。"
            )
            uncertainty = [
                "未显示的地点、时间、作者或画外信息不作推断。"
            ]
            if row["category"] in {"ocr", "formula"}:
                uncertainty.append("精确金标只覆盖指定的清晰区域，其他区域不补写。")
        else:
            review = HELDOUT_REVIEW[image_id]
            subject = review["subject"]
            observable = review["facts"]
            scope = review.get("scope")
            transcription = review.get("transcription")
            supported = review["supported"]
            claim = review["claim"]
            verification_evidence = review["evidence"]
            uncertainty = [
                "只依据当前分辨率下可见的内容；未显示的信息不作推断。"
            ]
            if row["category"] in {"ocr", "formula"}:
                uncertainty.append("金标仅覆盖题目指定且经两次核对的清晰区域。")

        fact = {
            **base,
            "subject": subject,
            "observable_facts": observable,
            "gold_transcription": _gold(row["category"], scope, transcription),
            "uncertainty_notes": uncertainty,
            "verification_claim": claim,
            "claim_supported": supported,
            "prompts": _prompts(row["split"], row["category"], scope),
            "answers": _answers(
                row["split"],
                row["category"],
                subject,
                observable,
                uncertainty,
                transcription,
                supported,
                verification_evidence,
            ),
            "annotation_status": "double_checked",
            "annotation_method": "manual visual review plus second-pass scoped transcription audit",
        }
        facts.append(fact)
    return sorted(facts, key=lambda item: (item["split"], item["category"], item["image_id"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--legacy-facts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    legacy_facts = json.loads(args.legacy_facts.read_text(encoding="utf-8"))
    facts = build_curated_facts(manifest, legacy_facts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(facts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(facts)} double-checked facts to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
