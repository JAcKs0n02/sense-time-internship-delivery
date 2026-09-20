#!/usr/bin/env python3
"""Configure Gemini locally; no network access or credential output."""
import argparse
import getpass
import sys
from pathlib import Path
from setup_week8_judge_key import save_key, load_key

KEY_PATH = Path.home() / '.config' / 'internship-week8' / 'gemini.key'


def main(argv=None):
    parser = argparse.ArgumentParser(description='保存或本地检查 Week8 Gemini Key')
    parser.add_argument('--check', action='store_true', help='仅检查本地文件，不调用 API')
    args = parser.parse_args(argv)
    if args.check or KEY_PATH.exists() or KEY_PATH.is_symlink():
        try:
            load_key(KEY_PATH)
        except FileNotFoundError:
            print('尚未配置 Gemini Key；请在终端运行本脚本（不带 --check）。')
            return 1
        except (ValueError, OSError):
            print('本地 Key 文件校验失败；文件未修改，密钥未显示。')
            return 1
        print('Gemini Key 本地格式与权限检查通过；未验证 API 有效性，未覆盖文件。')
        return 0
    if not sys.stdin.isatty():
        print('请在交互式终端运行，以便隐藏 Key 输入；未读取或保存输入。')
        return 1
    try:
        value = getpass.getpass('粘贴 Gemini API Key 后回车（输入不显示，不要加引号）：')
        save_key(value, KEY_PATH)
        load_key(KEY_PATH)
    except (ValueError, OSError, EOFError):
        print('配置未完成；请检查输入和本地文件权限。密钥未显示。')
        return 1
    except KeyboardInterrupt:
        print('\n已取消。')
        return 1
    print('Gemini Key 已保存到仓库外的私有文件；未调用 API、未打印密钥。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
