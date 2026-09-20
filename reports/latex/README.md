# 报告排版与源文件

唯一内容主稿为 `reports/technical_report.md`。`reports/technical_report.tex`由脚本生成，模板为本目录的`report-template.tex`。请修改Markdown后重新生成，不要分别维护三份正文。

当前报告采用LaTeX源文件和PDF。旧`technical_report.docx`保留为9月18日内容的历史快照，不作为新版报告。

在仓库根目录准备Python及Pandoc：

```bash
python -m pip install pypandoc_binary
python scripts/render_technical_report_latex.py
```

已生成的LaTeX无需Python即可编译。安装Tectonic后，在`reports/`目录运行：

```bash
tectonic technical_report.tex
```

也可以在具备ctex、XeLaTeX、Fandol及TeX Gyre字体的TeX Live环境中运行两次`xelatex technical_report.tex`，使目录页码收敛。当前实测编译器为Tectonic 0.17.0；其他工具链未单独验证。首次编译需要下载LaTeX宏包，后续可以复用缓存。

源文件交付包括Markdown、生成的`.tex`、本目录模板、渲染脚本及`reports/figures/`图片。PDF的中文正文采用Fandol宋体常规字重，标题采用Fandol黑体；西文与数字使用配套TeX Gyre字体。正文11pt，标题按层级设置，表格和图注统一使用小字号；不逐段设置字体，也不依赖Word样式转换。

旧Word/PDF/Markdown备份位于`reports/week8/latex_report_20260919/previous_version/`。LaTeX初版检查位于同级验证目录。9月20日文字修订的备份与检查记录位于`reports/week8/report_editorial_20260920/`；各验证记录仅对应其注明的文件版本。
