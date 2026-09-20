#!/usr/bin/env python3
"""Supervise final Week7 text service and Gradio; stop only owned children."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from common import ROOT, write_json


def require_free_port(port):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', port))


def wait_http(url, child, timeout, expected_model=None):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError(f'service exited with {child.returncode}: {url}')
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    if expected_model:
                        ids = [x['id'] for x in json.load(response)['data']]
                        if ids != [expected_model]:
                            raise RuntimeError(f'unexpected served model: {ids}')
                    return
        except (OSError, TimeoutError):
            pass
        time.sleep(1)
    raise RuntimeError(f'health timeout: {url}')


def supervise(run_dir):
    children = []; logs = []
    run_dir.mkdir(parents=True, exist_ok=False)
    def interrupted(signum, frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        require_free_port(8000); require_free_port(7860)
        env = dict(os.environ, WEEK7_TEXT_BASE_URL='http://127.0.0.1:8000/v1', GRADIO_ANALYTICS_ENABLED='False')
        if not env.get('WEEK7_SERVING_PYTHON') or not env.get('WEEK7_TEXT_MODEL_PATH'):
            raise ValueError('set WEEK7_SERVING_PYTHON and WEEK7_TEXT_MODEL_PATH to the verified AWQ deployment')
        commands = [
            ['bash', str(ROOT/'deliverables/week7/day36/source/scripts/start_text_server.sh')],
            [env.get('WEEK7_UI_PYTHON', sys.executable), str(ROOT/'deliverables/week7/day38/app.py')],
        ]
        for i, command in enumerate(commands):
            log = (run_dir/f'{i}.log').open('w'); logs.append(log)
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            children.append(child)
            if i == 0:
                wait_http('http://127.0.0.1:8000/health', child, 300)
                wait_http('http://127.0.0.1:8000/v1/models', child, 10, 'week4-dpo-quantized')
            else:
                wait_http('http://127.0.0.1:7860/', child, 90)
        write_json(run_dir/'status.json', {'status': 'ready', 'supervisor_pid': os.getpid(), 'child_pids': [p.pid for p in children], 'url': 'http://127.0.0.1:7860', 'vision_backend': 'not_started_single_gpu'})
        while all(p.poll() is None for p in children):
            time.sleep(1)
        raise RuntimeError('one deployment child exited')
    except KeyboardInterrupt:
        write_json(run_dir/'status.json', {'status': 'stopped'})
    except Exception as exc:
        write_json(run_dir/'status.json', {'status': 'failed', 'error': str(exc)})
        raise
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()
        for log in logs:
            log.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--foreground', action='store_true', help='internal supervisor mode')
    args = p.parse_args()
    args.run_dir = args.run_dir.resolve()
    if args.foreground:
        supervise(args.run_dir)
        return
    if args.run_dir.exists():
        raise ValueError('deployment directory already exists; use a new run directory')
    args.run_dir.parent.mkdir(parents=True, exist_ok=True)
    with (args.run_dir.parent/(args.run_dir.name+'_supervisor.log')).open('w') as log:
        process = subprocess.Popen([sys.executable, __file__, '--foreground', '--run-dir', str(args.run_dir)], start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic()+420
        while time.monotonic() < deadline:
            status_path = args.run_dir/'status.json'
            if status_path.exists():
                status = json.loads(status_path.read_text())
                if status['status']=='ready':
                    print(json.dumps(status)); return
                raise RuntimeError(str(status))
            if process.poll() is not None:
                raise RuntimeError('deployment supervisor exited before readiness')
            time.sleep(1)
        raise RuntimeError('supervisor startup timed out')
    except BaseException:
        if process.poll() is None:
            process.terminate(); process.wait(timeout=45)
        raise


if __name__ == '__main__':
    main()
