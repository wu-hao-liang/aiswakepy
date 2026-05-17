"""Playwright script: load pre-computed data via /api/dev_load and capture the map.

Usage (server must already be running):
    uv run python scripts/capture_map.py

Or start server and capture in one go:
    uv run python scripts/capture_map.py --start-server

Output: screenshots/map_tracks.png  (and optionally map_waves.png)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).parent.parent
BASE_URL = 'http://localhost:8050'
# Default output directory to load — adjust as needed
DEFAULT_DATA_DIR = 'data/compare_JI/output'
SCREENSHOT_DIR = REPO / 'screenshots'


def wait_for_server(timeout_s: int = 30) -> None:
    import urllib.request
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            urllib.request.urlopen(BASE_URL, timeout=2)
            return
        except Exception:
            time.sleep(1)
    raise TimeoutError(f'Server not reachable at {BASE_URL} after {timeout_s}s')


def capture(data_dir: str = DEFAULT_DATA_DIR, headless: bool = True) -> None:
    from playwright.sync_api import sync_playwright

    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=headless)
        page = browser.new_page(viewport={'width': 1600, 'height': 900})

        print(f'Navigating to {BASE_URL} …')
        page.goto(BASE_URL)
        page.wait_for_load_state('networkidle', timeout=15_000)

        # POST to dev_load endpoint
        print(f'Loading data from: {data_dir}')
        resp = page.request.post(
            f'{BASE_URL}/api/dev_load',
            data={'directory': data_dir},
            headers={'Content-Type': 'application/json'},
        )
        if resp.status != 200:
            print(f'ERROR: /api/dev_load returned {resp.status}: {resp.text()}')
        else:
            result = resp.json()
            print(f'  segments: {result.get("n_segs", "?")}  waves: {result.get("n_waves", "?")}')

        # Navigate to app and wait for deck.gl to re-render after version bump
        page.goto(BASE_URL)
        page.wait_for_load_state('networkidle', timeout=15_000)

        # Give deck.gl time to fetch Arrow data and render
        page.wait_for_timeout(4000)

        out = SCREENSHOT_DIR / 'map_tracks.png'
        page.screenshot(path=str(out), full_page=False)
        print(f'Saved: {out}')

        # Zoom in if waves are present (optional second screenshot)
        waves_path = SCREENSHOT_DIR / 'map_waves.png'
        page.evaluate("""() => {
            if (window.deckInstance) {
                window.deckInstance.setProps({
                    initialViewState: { longitude: 103.72, latitude: 1.26,
                                        zoom: 13, pitch: 0, bearing: 0 }
                });
            }
        }""")
        page.wait_for_timeout(2000)
        page.screenshot(path=str(waves_path), full_page=False)
        print(f'Saved: {waves_path}')

        browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description='Capture aiswakepy Dash map screenshot')
    parser.add_argument('--data-dir', default=DEFAULT_DATA_DIR,
                        help='Output directory to load (relative to repo root)')
    parser.add_argument('--start-server', action='store_true',
                        help='Start the Dash server before capturing')
    parser.add_argument('--no-headless', action='store_true',
                        help='Show browser window (useful for debugging)')
    args = parser.parse_args()

    server_proc = None
    if args.start_server:
        print('Starting server…')
        server_proc = subprocess.Popen(
            [sys.executable, '-m', 'uv', 'run', 'python', 'dash_app.py'],
            cwd=str(REPO),
        )
        try:
            wait_for_server(timeout_s=30)
            print('Server ready.')
        except TimeoutError as e:
            print(e)
            server_proc.terminate()
            sys.exit(1)

    try:
        capture(data_dir=args.data_dir, headless=not args.no_headless)
    finally:
        if server_proc:
            server_proc.terminate()


if __name__ == '__main__':
    main()
