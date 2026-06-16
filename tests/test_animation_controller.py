from pathlib import Path

import pytest
from playwright.sync_api import Error as PlaywrightError


SCRIPT = (
    Path(__file__).parents[1] / "assets" / "animation_controller.js"
).read_text()


@pytest.fixture
def animation_page(playwright):
    try:
        browser = playwright.chromium.launch()
    except PlaywrightError as exc:
        pytest.skip(f"Chromium system dependencies unavailable: {exc}")
    page = browser.new_page()
    page.set_content("<html><body></body></html>")
    page.add_script_tag(content=SCRIPT)
    page.evaluate(
        """
        () => {
            window.testNow = 0;
            window.pendingFrame = null;
            window.changes = [];
            window.controller = new VesselWaveAnimationController({
                durationMs: 12000,
                trackFraction: 0.7,
                now: () => window.testNow,
                requestFrame: fn => { window.pendingFrame = fn; return 1; },
                cancelFrame: () => { window.pendingFrame = null; },
                onChange: state => window.changes.push({
                    playing: state.playing,
                    progress: state.progress,
                    selected: !!state.selection,
                }),
            });
        }
        """
    )
    yield page
    browser.close()


def test_selection_enables_play_pause_and_resets(animation_page):
    page = animation_page
    assert page.evaluate("controller.toggle()") is False
    page.evaluate("controller.select({segIdx: 3, mmsi: 1, segmentId: 9})")
    assert page.evaluate("controller.getState().progress") == 0
    assert page.evaluate("controller.toggle()") is True
    page.evaluate("testNow = 6000; pendingFrame(6000)")
    assert page.evaluate("controller.getState().progress") == pytest.approx(0.5)

    page.evaluate("controller.select({segIdx: 4, mmsi: 2, segmentId: 10})")
    state = page.evaluate("controller.getState()")
    assert state["playing"] is False
    assert state["progress"] == 0
    assert state["selection"]["segIdx"] == 4

    page.evaluate("controller.clear()")
    assert page.evaluate("controller.getState().selection") is None


def test_recursive_loop_and_source_timed_ray_progress(animation_page):
    page = animation_page
    page.evaluate("controller.select({segIdx: 1}); controller.play()")
    page.evaluate("testNow = 12500; pendingFrame(12500)")
    assert page.evaluate("controller.getState().progress") == pytest.approx(500 / 12000)

    page.evaluate("controller.pause(); controller.setProgress(0.5)")
    assert page.evaluate("controller.getState().trackProgress") == pytest.approx(5 / 7)
    assert page.evaluate("controller.rayProgress(0.5)") == pytest.approx(
        (0.5 - 0.35) / 0.65
    )
    assert page.evaluate("controller.rayProgress(0.9)") == 0
