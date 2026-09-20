#!/usr/bin/env python3
"""Store the Week8 API credential privately, outside the repository."""
import getpass
import os
import stat
from pathlib import Path

KEY_PATH = Path.home() / '.config' / 'internship-week8' / 'deepseek.key'


def save_key(value, path=KEY_PATH):
    if not value or any(c.isspace() for c in value):
        raise ValueError('Key must be nonempty and contain no whitespace')
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        stream.write(value)


def load_key(path=KEY_PATH):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Key must not be a symlink')
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
        raise ValueError('Key file permissions must be private (0600)')
    key = path.read_text()
    if not key or any(c.isspace() for c in key):
        raise ValueError('Key file content is invalid')
    return key


def main():
    if KEY_PATH.exists() or KEY_PATH.is_symlink():
        print('本地密钥文件已存在，未覆盖。')
        return
    value = getpass.getpass('请粘贴 DeepSeek API Key（输入不会显示），然后回车：')
    save_key(value)
    print('密钥已保存到仓库外的私有文件；未调用API、未打印密钥。')


if __name__ == '__main__':
    main()
