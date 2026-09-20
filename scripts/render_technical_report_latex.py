#!/usr/bin/env python3
"""Generate LaTeX from the canonical Markdown; compile with Tectonic or XeLaTeX.

Requires pandoc on PATH or the pypandoc_binary Python package.
This renderer never reassembles or overwrites the Markdown report.
"""
from pathlib import Path
import json
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/'reports'


def pandoc_command():
    executable = shutil.which('pandoc')
    if not executable:
        import pypandoc
        executable = pypandoc.get_pandoc_path()
    return executable


def plain(inlines):
    return ''.join(x.get('c', '') if x['t'] == 'Str' else ' ' if x['t'] in ('Space', 'SoftBreak') else '' for x in inlines)


def render():
    pandoc = pandoc_command()
    text = (REPORTS/'technical_report.md').read_text()
    # Title and edition are supplied by the fixed template.
    text = re.sub(r'^# [^\n]+\n+[^\n]+\n+', '', text, count=1)
    document = json.loads(subprocess.check_output(
        [pandoc, '--from=markdown+pipe_tables', '--to=json'], input=text, text=True))
    heading = '实验结果'
    chapter = 0
    subsection = 0
    for block in document['blocks']:
        if block['t'] == 'Header':
            level, _, inlines = block['c']
            heading = plain(inlines)
            if level == 2:
                match = re.match(r'第(\d+)章', heading)
                chapter = int(match[1]) if match else 0
                subsection = 0
            elif level == 3 and chapter:
                subsection += 1
                heading = re.sub(r'^\d+(?:\.\d+)*(?:[.、]\s*|\s+)', '', heading)
                block['c'][2] = [{'t':'Str','c':f'{chapter}.{subsection} {heading}'}]
            block['c'][0] = max(1, level-1)
        elif block['t'] == 'Table':
            c = block['c']
            title = re.sub(r'^\d+(?:\.\d+)*(?:[.、]\s*|\s+)', '', heading)
            c[1] = [None, [{'t':'Plain','c':[{'t':'Str','c':title}]}]]
            n = len(c[2])
            widths = {2:[.38,.62],3:[.32,.38,.30],4:[.31,.23,.23,.23],5:[.28,.18,.18,.18,.18]}.get(n, [1/n]*n)
            widths = [int(w*10000)/10000 for w in widths[:-1]]
            widths.append(round(1-sum(widths), 4))
            for spec, width in zip(c[2], widths):
                spec[1] = {'t':'ColWidth','c':width}
            for row in c[3][1]:
                for cell in row[1]:
                    for paragraph in cell[4]:
                        if paragraph['t'] in ('Plain','Para'):
                            paragraph['c'] = [{'t':'Strong','c':paragraph['c']}]
        elif block['t'] == 'Figure':
            caption = block['c'][1][1]
            if caption:
                title = re.sub(r'^图\s*\d+\s*', '', plain(caption[0]['c']))
                caption[0]['c'] = [{'t':'Str','c':title}]

    def walk(node):
        if isinstance(node, list):
            return [walk(x) for x in node]
        if not isinstance(node, dict):
            return node
        if node.get('t') == 'Str' and '/' in node['c']:
            pieces = re.split(r'([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+)', node['c'])
            if len(pieces) > 1:
                inlines = []
                for i, piece in enumerate(pieces):
                    if not piece:
                        continue
                    inlines.append({'t':'RawInline','c':['latex','\\nolinkurl{'+piece+'}']}
                                   if i % 2 else {'t':'Str','c':piece})
                return {'t':'Span','c':[['',[],[]],inlines]}
        if node.get('t') == 'CodeBlock':
            # Break long paths/commands; no syntax colors or fake bold.
            content = node['c'][1].replace('├─', '+-').replace('└─', '+-')
            return {'t':'RawBlock','c':['latex', '\\begin{Verbatim}[breaklines,breakanywhere,fontsize=\\small]\n'+content+'\n\\end{Verbatim}']}
        if node.get('t') == 'Code':
            content = node['c'][1]
            # URL-style wrapping is useful for identifiers and archive paths.
            if not any(c in content for c in '{}\\'):
                return {'t':'RawInline','c':['latex','\\nolinkurl{'+content+'}']}
        if node.get('t') == 'Link' and '://' not in node['c'][2][0]:
            # Repository-relative links are meaningful in Markdown, not in an
            # isolated downloaded PDF. Preserve their visible label as text.
            return {'t':'Span','c':[['',[],[]], walk(node['c'][1])]}
        return {k:walk(v) for k,v in node.items()}

    document = walk(document)
    blocks = []
    for block in document['blocks']:
        if block['t'] == 'Table':
            rows = len(block['c'][3][1]) + sum(len(body[3]) for body in block['c'][4])
            headings = []
            while blocks and blocks[-1]['t'] == 'Header':
                headings.insert(0, blocks.pop())
            space = min(rows*20+35+35*len(headings), 550)
            blocks.append({'t':'RawBlock','c':['latex',f'\\Needspace{{{space}pt}}']})
            blocks.extend(headings)
        blocks.append(block)
    document['blocks'] = blocks
    output = REPORTS/'technical_report.tex'
    subprocess.run([pandoc, '--from=json', '--to=latex', '--standalone',
        '--template='+str(REPORTS/'latex/report-template.tex'), '--output='+str(output)],
        input=json.dumps(document,ensure_ascii=False), text=True, check=True)
    print(output)


if __name__ == '__main__':
    render()
