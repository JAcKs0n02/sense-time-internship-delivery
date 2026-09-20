#!/usr/bin/env python3
"""Merge the reviewed visual annotations with the immutable source manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def ann(subject: str, details: list[str], visible_text: str) -> dict:
    return {"subject": subject, "details": details, "visible_text": visible_text}


ANNOTATIONS = {
    # General visual understanding / natural and product scenes.
    "natural_scene-01": ann(
        "一名穿橙色宇航服的女宇航员半身肖像",
        ["人物身后有美国国旗", "右后方可见白色火箭模型，前景有黑色头盔"],
        "宇航服胸前可见 NASA 风格任务徽章",
    ),
    "natural_scene-02": ann(
        "一名在户外操作三脚架相机的男子",
        ["画面为黑白照片", "男子弯腰贴近取景器，远处有建筑和树木"],
        "无需要转写的主体文字",
    ),
    "natural_scene-03": ann(
        "一只棕色虎斑猫的面部特写",
        ["猫有黄绿色眼睛", "鼻子位于画面中央，背景明显虚化"],
        "无需要转写的主体文字",
    ),
    "natural_scene-04": ann(
        "木桌上的一杯咖啡",
        ["白色咖啡杯放在红色杯碟上", "杯碟右侧放有一把金属小勺"],
        "无需要转写的主体文字",
    ),
    "natural_scene-05": ann(
        "深色背景上按行摆放的多枚硬币",
        ["硬币大小和图案各不相同", "图像为灰度照片，共有多行硬币"],
        "无需要转写的主体文字",
    ),
    "natural_scene-06": ann(
        "由长方形砖块组成的规则表面纹理",
        ["砖块呈灰色", "砖缝形成错位的横纵线条"],
        "无需要转写的主体文字",
    ),
    "natural_scene-07": ann(
        "近距离拍摄的草地纹理",
        ["画面由交错的草叶覆盖", "图像为灰度或低饱和度"],
        "无需要转写的主体文字",
    ),
    "natural_scene-08": ann(
        "铺满画面的碎石纹理",
        ["碎石大小接近但形状不规则", "画面没有明显前景主体或文字"],
        "无需要转写的主体文字",
    ),
    "natural_scene-09": ann(
        "白色背景上的黑色马匹侧面剪影",
        ["马头朝向右侧", "四条腿、尾巴和耳朵轮廓清晰"],
        "无需要转写的主体文字",
    ),
    "natural_scene-10": ann(
        "布满星系和恒星的深空图像",
        ["背景接近黑色", "可见大量不同颜色和亮度的点状、团状天体"],
        "无需要转写的主体文字",
    ),
    "natural_scene-11": ann(
        "布满撞击坑和阴影的月球表面",
        ["图像为灰度", "不同大小的圆形凹坑分布在崎岖地表"],
        "无需要转写的主体文字",
    ),
    "natural_scene-12": ann(
        "车库内停放的一辆红色摩托车",
        ["摩托车侧面朝向右侧", "背景可见木架、纸箱和工具等杂物"],
        "无需要转写的主体文字",
    ),
    "natural_scene-13": ann(
        "眼底视网膜的圆形医学图像",
        ["整体为橙红色", "左侧可见较亮的视盘，并有血管向四周分支"],
        "无需要转写的主体文字",
    ),
    "natural_scene-14": ann(
        "夜间发射台上的白色运载火箭",
        ["火箭竖立在画面中央偏右", "两侧有发射塔架，底部灯光形成星芒"],
        "无需要转写的主体文字",
    ),

    # OCR and text transcription.
    "ocr-01": ann(
        "一张中英文登机牌",
        ["票面包含航班、日期、座位、出发地和目的地字段", "右侧有红色印章，底部有条形码"],
        "BOARDING PASS；MU 2379；03DEC；FUZHOU；TAIYUAN；ZHANGQIWEI；G11",
    ),
    "ocr-02": ann(
        "由多张英文招牌和体育照片组成的拼图",
        ["上排可见椭圆或矩形商店招牌", "中下部可见球员背影和体育场入口"],
        "BLUE LAGOON；TROPICAL BAR & GRILL；SEA YARD COFFEE；Hy-Vee；WAL-MART",
    ),
    "ocr-03": ann(
        "五个分隔开的手写汉字样本",
        ["每个字位于独立浅色小框内", "笔迹为黑灰色且粗细不一"],
        "放；澳；吧；把；板（个别字形可能受手写体影响）",
    ),
    "ocr-04": ann(
        "两张街景招牌检测示例的并排拼图",
        ["左图是商铺入口及多块绿色标牌", "右图包含黄色横向招牌和红色竖向招牌"],
        "可见多处中文店名与服务文字；小字号内容不宜强行补全",
    ),
    "ocr-05": ann(
        "两张中文店面招牌及其标注文字",
        ["左图为深色网咖招牌", "右图为红色艺术教育机构招牌"],
        "GT: 依佳人网咖；GT: 赫伦芭蕾国际艺术教育",
    ),
    "ocr-06": ann(
        "一组多行彩色手写数字与字母",
        ["字符按行排列，颜色包括灰、绿、蓝、紫等", "部分笔画重叠或不清晰"],
        "可辨认若干数字串与字母 d、m；由于手写重叠，不应声称完整精确转写",
    ),
    "ocr-07": ann(
        "车辆牌照的倾斜近景",
        ["蓝色牌照位于画面中央", "白色字符受透视和反光影响"],
        "可见皖A开头的车牌，后续字符近似 533Y3，需保留不确定性",
    ),
    "ocr-08": ann(
        "灰底黑字的中文文本行裁片",
        ["文字横向排列", "左侧开头为“如，”"],
        "如，和对旋游表演形式（中部字形较模糊）",
    ),
    "ocr-09": ann(
        "灰底黑字的中文报告文本行裁片",
        ["文字为印刷体", "右端仍有被裁切的后续内容"],
        "4年工作报告》中指出",
    ),
    "ocr-10": ann(
        "蓝白描边的中文门店招牌",
        ["大字横向排列", "背景为深色"],
        "魅派集成吊顶",
    ),
    "ocr-11": ann(
        "白色发光中文店招",
        ["文字位于灰色背景上", "字间距较大"],
        "母婴用品连锁",
    ),
    "ocr-12": ann(
        "放在地面上的招商银行联名银行卡",
        ["卡片为橙黄色并带卡通图案", "右下角可见银联标识"],
        "招商银行；MICHAEL；UnionPay 银联",
    ),
    "ocr-13": ann(
        "包含街景、店招、票据和手机截图的多图拼图",
        ["小图数量较多", "可见中文商户名、数字价格和交通标识等多种文字"],
        "可辨认数字 2.33 等局部内容；不应把所有小图文字拼成一段连续文本",
    ),
    "ocr-14": ann(
        "一张英文购物收据的照片",
        ["收据中央有较大的号码", "下方按行列出项目、数量与金额"],
        "YOUR NUMBER TOTAL IS；T31232；TOTAL；底部包含 Please See Your Receipt 等提示",
    ),

    # Tables, charts and structured document understanding.
    "chart_table-01": ann(
        "一张比较不同组别指标的科研数据表",
        ["左侧列出 Group 和 N", "表头分为 fHjDA uptake 与 DAT density，并各有 Vmax、Km、SD 等子列"],
        "Group；N；Vmax；Km；SD；fHjDA uptake；DAT density",
    ),
    "chart_table-02": ann(
        "一张行为量表相关系数表",
        ["行列均列出多种量表或行为维度", "部分数值带星号表示统计标记"],
        "BEHAVE-AD Scales；M-NCAS Scales；Total BEHAVE-AD；Hallucination；Aggression",
    ),
    "chart_table-03": ann(
        "一道带手写计算痕迹的分组频数表题目",
        ["表格按数值区间分组", "右侧是频数或百分比，表格旁有手写算式"],
        "可见区间如 600≤x<800、800≤x<1000、1000≤x<1200，以及手写百分数",
    ),
    "chart_table-04": ann(
        "一页带彩色版面框标注的双栏学术文档",
        ["页面上方有标题与作者信息", "正文为双栏排版，并包含被框出的表格或文本区域"],
        "页内为英文论文文字；小字号不适合逐字完整转写",
    ),
    "chart_table-05": ann(
        "一页包含大量人脸缩略图和对照表的学术文章",
        ["上方是多行图像样本", "中部及下方有表格和双栏正文"],
        "可见 Figure/Table 风格标题及多组数值，但字号较小",
    ),
    "chart_table-06": ann(
        "一张按紫、绿、红色区块分组的结构化表格",
        ["不同颜色区块包含多行记录", "右侧列中有数值或短文本"],
        "表格含多个分组标题与行项目；缩略图不足以可靠补全全部单元格",
    ),
    "chart_table-07": ann(
        "包含商品招牌图片和多个比较表的报告页",
        ["上半部分是多张圆形或矩形店招图片", "下半部分是密集的数值表与说明文字"],
        "可见多列指标、百分比和英文说明；应按图表区域分别读取",
    ),
    "chart_table-08": ann(
        "两张并排的中文年度数据表",
        ["左表为灰色浅网格", "右表有更明显的黑色边框，列头包含多个年份"],
        "可见年份 2020、2021、2022、2023、2024 及多行百分比数值",
    ),
    "chart_table-09": ann(
        "蓝色表头的中文业务表与右侧明细表拼图",
        ["左侧是多列记录表", "右侧按行列展示较长中文说明"],
        "包含多列中文字段、编号和说明文字；单元格内容较密集",
    ),
    "chart_table-10": ann(
        "两份并排展示的检验结果表",
        ["左侧表格含检验名称、结果、单位和参考范围", "右侧表格对同类项目给出结果与参考区间"],
        "可见项目缩写与数值，以及 U/L、参考范围等字段",
    ),
    "chart_table-11": ann(
        "表格 OCR 处理流程图",
        ["左侧输入表格经过文本检测和表格结构预测", "右侧通过 Cell 坐标聚合与文本聚合生成表格 Excel"],
        "文本检测算法；表格结构预测；Cell坐标聚合；Cell文本聚合；表格Excel",
    ),
    "chart_table-12": ann(
        "按国家或地区统计奖牌数的排名表",
        ["列包含名次、国家/地区、金牌、银牌、铜牌和奖牌总数", "前几行可见中国、美国、俄罗斯等"],
        "名次；国家/地区；金牌；银牌；铜牌；奖牌总数；中国(CHN) 48 22 30 100",
    ),
    "chart_table-13": ann(
        "一张名为 CRUncover 的小型结果表",
        ["表格有多列文本与数值", "底部一行包含 2.72、Ingcubic 和 $744.78"],
        "CRUncover；Dres；Abstr；2.72；Ingcubic；$744.78",
    ),
    "chart_table-14": ann(
        "一页含三幅小图和双栏正文的学术文档",
        ["页面上部并排展示图表", "下方是英文双栏正文，并有彩色区域框"],
        "包含坐标图、图注和英文论文正文；小字号正文不宜逐字猜测",
    ),

    # User interface understanding.
    "ui-01": ann(
        "Windows 上打开的 Microsoft Excel 空白工作簿",
        ["顶部是功能区和公式栏", "中央是带行号列标的空白单元格网格"],
        "Excel 功能区可见 Home 等标签，工作表区域尚未输入数据",
    ),
    "ui-02": ann(
        "浏览器中的 Google 搜索主页",
        ["中央有 Google 标志和搜索框", "页面顶部和底部有导航链接"],
        "Google；Google Search；I'm Feeling Lucky",
    ),
    "ui-03": ann(
        "一部 iPhone 的主屏幕应用网格",
        ["多行应用图标排列在蓝色背景上", "底部 Dock 栏含电话、信息等常用应用"],
        "图标中可见 X、TikTok 等应用标识",
    ),
    "ui-04": ann(
        "横向显示的 iPadOS 主屏幕",
        ["左侧有天气、时钟和日历等小组件", "底部为应用 Dock，背景是卡通羊图案"],
        "小组件中可见温度 3° 及日历信息",
    ),
    "ui-05": ann(
        "浏览器内打开的 Microsoft OneNote 网页界面",
        ["左侧是笔记本和页面导航栏", "中间为笔记编辑区域，顶部有紫色工具栏"],
        "OneNote 界面包含 Home、Insert、Draw 等功能区域",
    ),
    "ui-06": ann(
        "Microsoft Word 中的空白文档编辑界面",
        ["顶部功能区展开", "中央白色页面顶部出现 Copilot 写作提示"],
        "Draft with Copilot；Home；Insert；Draw；Design；Layout",
    ),
    "ui-07": ann(
        "浏览器中的 OmniParser 相关演示或说明页面",
        ["页面左侧为文字说明", "中部包含一张桌面界面截图，右侧有橙色操作按钮"],
        "页面包含 Setup/Run 类说明与截图；按钮文字较小",
    ),
    "ui-08": ann(
        "深色主题的 Microsoft Teams 聊天界面",
        ["左侧是团队或聊天列表", "中间为消息会话，右侧展示代码或内容面板"],
        "Teams 界面可见多条会话消息和顶部搜索/导航区域",
    ),
    "ui-09": ann(
        "Windows 11 桌面",
        ["背景为蓝色 Windows 11 波纹图案", "左侧有多个桌面快捷方式，底部任务栏居中"],
        "桌面可见 Google Chrome、Microsoft Edge、VLC 等快捷方式",
    ),
    "ui-10": ann(
        "以山地风景为壁纸的 Windows 桌面",
        ["桌面本身没有打开应用窗口", "底部可见任务栏，壁纸显示山脊与沙地"],
        "未见打开的对话框或文档窗口",
    ),
    "ui-11": ann(
        "Windows 上并排打开的任务管理器和 Google 浏览器窗口",
        ["左侧任务管理器显示进程与资源占用", "右侧浏览器停留在 Google 搜索主页"],
        "Task Manager；Processes；Google；CPU；Memory",
    ),
    "ui-12": ann(
        "Windows 11 桌面及左侧快捷方式",
        ["背景为蓝色 Windows 11 图案", "底部任务栏可见，桌面左侧排列应用图标"],
        "可见浏览器、回收站等快捷方式；没有打开应用窗口",
    ),
    "ui-13": ann(
        "Microsoft Word 的文档编辑窗口",
        ["顶部为 Word 功能区", "中央白色页面基本为空，并显示 Copilot 写作提示"],
        "Draft with Copilot；Home；Insert；Draw；Design；Layout",
    ),
    "ui-14": ann(
        "浏览器中的 Microsoft Corporation 股票信息页",
        ["页面中央有价格折线图", "右侧显示市场摘要和绿色涨幅数据"],
        "Microsoft Corp；MSFT；373.04 USD；Market summary",
    ),

    # Formula transcription and explanation.
    "formula-01": ann(
        "一条白字黑底的眼睑损失函数",
        ["含对 t 和 v 的双重求和", "末项是预测量与目标量之差的平方范数"],
        r"\mathcal{L}_{eyelid}=-\sum_{t=1}^{T}\sum_{v=1}^{V}\mathcal{M}_{v}^{(eyelid)}\left(\lVert\hat{h}_{t,v}-x_{t,v}\rVert^2\right)",
    ),
    "formula-02": ann(
        "一条蓝灰色的有理式分解等式",
        ["左端分子近似 7x^2-x+1，分母是三次多项式", "右端把表达式拆为三个分式项"],
        r"\frac{7x^2-x+1}{x^3+1}=\frac{7x^2-x+1}{(x+1)(x^2-x+1)}=\frac{1}{x+1}+\frac{6x+6}{x^2-x+1}",
    ),
    "formula-03": ann(
        "一条包含两个无穷级数的复变函数展开式",
        ["等式以 F(z)=F(0) 开头", "后续两组求和的分母含 z-a_m 或 z-a_{-m}"],
        r"F(z)=F(0)+\sum_{m=1}^{\infty}\left\{\frac{b_m}{z-a_m}+\frac{b_m}{a_m}\right\}+\sum_{m=1}^{\infty}\left\{\frac{b_{-m}}{z-a_{-m}}+\frac{b_{-m}}{a_{-m}}\right\}",
    ),
    "formula-04": ann(
        "一条红字黑底的组合损失函数",
        ["左端为 L_t", "右端包含 L_{rec}、分式项以及带根号的 e_t 项"],
        r"L_t=L_{rec}+\left[\frac{(1-\alpha_t)^2}{2\alpha_t(1-\bar{\alpha}_t)\lVert\Sigma_\theta\rVert_2^2}\right]e_t- e_t(\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon_t,t)",
    ),
    "formula-05": ann(
        "一个包含三条线性方程的方程组",
        ["左侧用大括号连接三行", "未知量为 a、b、c"],
        r"\begin{cases}25a-5b+c=1\\a-b+c=1\\9a-3b+c=0\end{cases}",
    ),
    "formula-06": ann(
        "一个四行四列的稀疏矩阵",
        ["矩阵多数元素为 0", "非零项包含 1、m^2gl^2/(Mml^2) 和 -mgl(M+m)/(Mml^2)"],
        r"\begin{bmatrix}0&1&0&0\\0&0&\frac{m^2gl^2}{Mml^2}&0\\0&0&0&1\\0&0&-\frac{mgl(M+m)}{Mml^2}&0\end{bmatrix}",
    ),
    "formula-07": ann(
        "一条手写的位移与速度积分关系式",
        ["左端为 s_2-s_1", "积分上下限为 t_1 和 t_2，被积函数为 v"],
        r"s_2-s_1=\int_{t_1}^{t_2}v\,dt",
    ),
    "formula-08": ann(
        "一条手写的广义积分等式",
        ["左右两边积分上下限均为负无穷到正无穷", "被积函数含指数项，左侧还含 ds"],
        r"\int_{-\infty}^{+\infty} e^{-t(s+\frac{2ij}{s+t})^2}\,ds=\int_{-\infty}^{+\infty}e^{-ts^2}\,ds",
    ),
    "formula-09": ann(
        "由两条常微分方程组成的手写方程组",
        ["第一行是一阶导数 dy/dx 与 y 的关系", "第二行包含二阶导数和一阶导数的三次方"],
        r"\begin{cases}\frac{dy}{dx}+y=3\\\frac{d^2y}{dx^2}+3\left(\frac{dy}{dx}\right)^3+y^3=0\end{cases}",
    ),
    "formula-10": ann(
        "一个手写极限表达式",
        ["自变量 x 趋近 1/4", "分子为 1-4^{x-1/4}，分母为 1-4x"],
        r"\lim_{x\to\frac14}\frac{1-4^{x-\frac14}}{1-4x}",
    ),
    "formula-11": ann(
        "一条印刷体的复积分表达式",
        [r"左端以 \zeta_0(\nu) 开头", "右端含无穷积分、C_+ 路径积分、指数项和波函数符号"],
        r"\zeta_0(\nu)=-\frac{\nu e^{-2\nu}}{\pi}\int_{\mu}^{\infty}d\omega\int_{C_+}dz\,\frac{2z^2}{(z^2+\omega^2)^{\nu+1}}\widetilde{\Psi}(\omega,z)e^{i\epsilon z}",
    ),
    "formula-12": ann(
        "一条白色手写字的代数项",
        ["黑色背景上有两个乘积项", "两个乘积之间是加号"],
        r"x_k x_{2k}+y_k y_{2k}",
    ),
    "formula-13": ann(
        "一条白色手写字的级数表达式",
        ["左端为 f(x) 或相近函数记号", "右端是带 a_n 和 x^n 的求和"],
        r"f(x)=\sum_{n=-\infty}^{\infty}a_n x^n（左端与下限手写字形略模糊）",
    ),
    "formula-14": ann(
        "一条白色手写字的复数模平方等式",
        ["左端为 |a+ib| 的平方形式", "右端是 a^2+b^2"],
        r"|a+ib|^2=a^2+b^2",
    ),
}


FALSE_CLAIMS = {
    "natural_scene": [
        "图中主体是一辆蓝色公交车",
        "画面右上角有醒目的红色文字",
        "图中至少有三个人正在交谈",
        "画面是室内会议室",
    ],
    "ocr": [
        "图中清楚写着 WELCOME HOME",
        "所有文字都是英文印刷体",
        "图中的数字只有 2024",
        "图片完全不含可读文字",
    ],
    "chart_table": [
        "图中是一张只有红色折线的折线图",
        "表格只有两行两列",
        "所有单元格都为空白",
        "图中没有任何结构化数据区域",
    ],
    "ui": [
        "界面正在显示打印对话框",
        "屏幕中央弹出了系统错误提示",
        "用户正在视频通话且摄像头已打开",
        "当前应用是一个全屏游戏",
    ],
    "formula": [
        "图中没有等号或数学符号",
        "表达式只包含普通中文段落",
        "公式的结果明确等于零",
        "图片是一张统计数据表而不是公式",
    ],
}


def _summary(category: str, annotation: dict) -> str:
    details = "；".join(annotation["details"])
    if category == "ocr":
        return f"图中是{annotation['subject']}。{details}。可辨认文字包括：{annotation['visible_text']}。"
    if category == "formula":
        return f"图中是{annotation['subject']}。可转写为：{annotation['visible_text']}。{details}。"
    return f"图中显示{annotation['subject']}。{details}。"


def build_facts(manifest: list[dict]) -> list[dict]:
    manifest_ids = {row["image_id"] for row in manifest}
    if manifest_ids != set(ANNOTATIONS):
        raise ValueError("annotation IDs do not exactly match the source manifest")
    facts = []
    category_positions = {category: 0 for category in FALSE_CLAIMS}
    for row in manifest:
        annotation = ANNOTATIONS[row["image_id"]]
        position = category_positions[row["category"]]
        category_positions[row["category"]] += 1
        false_claim = FALSE_CLAIMS[row["category"]][position % 4]
        summary = _summary(row["category"], annotation)
        fact = dict(row)
        fact.update(annotation)
        fact.update(
            {
                "summary": summary,
                "explanation": (
                    f"判断只依据图中可见的主体、布局、颜色和文字："
                    f"{'；'.join(annotation['details'])}。其中可见文字或符号为："
                    f"{annotation['visible_text']}。"
                ),
                "false_claim": false_claim,
                "correction": f"该说法不符合图像。{summary}",
                "annotation_status": "HUMAN_REVIEWED_FROM_CONTACT_SHEET",
            }
        )
        facts.append(fact)
    return facts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    facts = build_facts(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(facts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
