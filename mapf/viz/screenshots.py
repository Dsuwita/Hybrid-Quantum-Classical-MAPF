#!/usr/bin/env python3
"""Capture one screenshot of each studio tab for the README.

Starts serve.py on a private port, drives the three tabs with a headless
Chromium (Playwright), runs a small solve on each so the canvases have real
content, and writes PNGs into mapf/viz/. This is a dev/CI convenience, not a
runtime dependency: the studio itself needs no browser automation.

    pip install playwright && python3 -m playwright install chromium
    python3 mapf/viz/screenshots.py
"""

import pathlib
import subprocess
import sys
import time

from playwright.sync_api import sync_playwright

VIZ = pathlib.Path(__file__).resolve().parent
ROOT = VIZ.parent.parent
PORT = 8912


def main():
    srv = subprocess.Popen([sys.executable, str(VIZ / "serve.py"), "--port", str(PORT),
                            "--no-browser"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.5)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1400, "height": 820})
            base = f"http://localhost:{PORT}"

            # --- Tab 1: MAPF studio ---
            page.goto(base)
            page.wait_for_timeout(600)
            # pick the room map + run hybrid and CBS side by side at 8 agents
            page.select_option("#m-map", value="data/maps/room-32-32.map")
            page.wait_for_timeout(400)
            page.check("#m-cbs")
            page.fill("#m-agents", "8") if False else page.eval_on_selector(
                "#m-agents", "el => { el.value = 8; el.dispatchEvent(new Event('input')); }")
            page.click("#m-run")
            page.wait_for_timeout(6000)  # let the live solve animate and finish
            page.screenshot(path=str(VIZ / "studio_mapf.png"))
            print("wrote studio_mapf.png")

            # --- Tab 2: Annealer lab ---
            page.click("text=Annealer lab")
            page.wait_for_timeout(300)
            for _ in range(3):
                page.click("#l-run")
                page.wait_for_timeout(1800)
            page.screenshot(path=str(VIZ / "studio_lab.png"))
            print("wrote studio_lab.png")

            # --- Tab 3: Max-Cut ---
            page.click("text=Max-Cut")
            page.wait_for_timeout(300)
            page.fill("#x-sweeps", "5000")
            page.click("#x-run")
            page.wait_for_timeout(4000)
            page.screenshot(path=str(VIZ / "studio_maxcut.png"))
            print("wrote studio_maxcut.png")

            browser.close()
    finally:
        srv.terminate()


if __name__ == "__main__":
    main()
