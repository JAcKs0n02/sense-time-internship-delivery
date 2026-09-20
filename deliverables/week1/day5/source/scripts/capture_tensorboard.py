#!/usr/bin/env python3
"""Capture the live TensorBoard Time Series page from the AutoDL host."""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:6006/#timeseries")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-output", type=Path, required=True)
    parser.add_argument("--wait-ms", type=int, default=12000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.text_output.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        page = browser.new_page(viewport={"width": 1600, "height": 1200})
        page.goto(args.url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(args.wait_ms)
        body_text = page.locator("body").inner_text()
        args.text_output.write_text(body_text + "\n", encoding="utf-8")
        page.screenshot(path=str(args.output), full_page=True)
        print(f"title={page.title()}")
        print(f"url={page.url}")
        print(f"body_text_length={len(body_text)}")
        print(f"contains_train_loss={'train/loss' in body_text}")
        print(f"contains_run1={'identity-qlora-run1' in body_text}")
        print(f"contains_run2={'identity-qlora-run2' in body_text}")
        print(f"screenshot={args.output}")
        browser.close()


if __name__ == "__main__":
    main()
