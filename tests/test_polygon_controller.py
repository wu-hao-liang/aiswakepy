from pathlib import Path

import pytest
from playwright.sync_api import Error as PlaywrightError


CONTROLLER_JS = (
    Path(__file__).parents[1] / "assets" / "polygon_controller.js"
)


@pytest.fixture
def polygon_page(playwright):
    try:
        browser = playwright.chromium.launch()
    except PlaywrightError as exc:
        pytest.skip(f"Chromium system dependencies unavailable: {exc}")
    page = browser.new_page()
    page.set_content(
        """
        <button id="draw">Draw polygon on the map</button>
        <div id="map" style="width:800px;height:600px"></div>
        <canvas id="overlay"></canvas>
        """
    )
    page.add_script_tag(path=str(CONTROLLER_JS))
    page.evaluate(
        """
        window.completedPolygons = [];
        window.controller = new window.AiswakePolygonController({
            container: document.getElementById('map'),
            canvas: document.getElementById('overlay'),
            button: document.getElementById('draw'),
            getViewport: () => ({
                project: point => point.slice(),
                unproject: point => point.slice(),
            }),
            onComplete: polygon => window.completedPolygons.push(polygon),
        });
        """
    )
    yield page
    browser.close()


def _click_map(page, x, y, *, ctrl=True):
    page.evaluate(
        """([x, y, ctrl]) => {
            const map = document.getElementById('map');
            const rect = map.getBoundingClientRect();
            const clientX = rect.left + x;
            const clientY = rect.top + y;
            map.dispatchEvent(new PointerEvent('pointerdown', {
                bubbles: true, button: 0, clientX, clientY, ctrlKey: ctrl,
            }));
            map.dispatchEvent(new PointerEvent('pointerup', {
                bubbles: true, button: 0, clientX, clientY, ctrlKey: ctrl,
            }));
        }""",
        [x, y, ctrl],
    )


def test_first_arm_and_ctrl_release_preserve_draft(polygon_page):
    page = polygon_page
    assert page.evaluate("controller.arm()") is True
    assert page.evaluate("controller.getState().armed") is True

    page.evaluate("controller.setCtrlHeld(true)")
    _click_map(page, 100, 100)
    _click_map(page, 200, 100)
    page.evaluate("controller.setCtrlHeld(false)")

    state = page.evaluate("controller.getState()")
    assert state["drawing"] is True
    assert state["ctrlHeld"] is False
    assert state["vertices"] == [[100, 100], [200, 100]]

    _click_map(page, 200, 200, ctrl=False)
    assert page.evaluate("controller.getState().vertices.length") == 2

    page.evaluate("controller.setCtrlHeld(true)")
    _click_map(page, 200, 200)
    assert page.evaluate("controller.getState().vertices.length") == 3


def test_backspace_undo_and_close_near_start(polygon_page):
    page = polygon_page
    page.evaluate("controller.setCtrlHeld(true); controller.arm()")
    for point in [(100, 100), (220, 100), (220, 220), (100, 220)]:
        _click_map(page, *point)

    page.keyboard.press("Backspace")
    assert page.evaluate("controller.getState().vertices.length") == 3

    _click_map(page, 108, 106)
    assert page.evaluate("completedPolygons.length") == 1
    assert page.evaluate("completedPolygons[0].length") == 3
    state = page.evaluate("controller.getState()")
    assert state["armed"] is False
    assert state["drawing"] is False


@pytest.mark.parametrize("cancel_action", ["escape", "right_click", "button_toggle"])
def test_cancel_controls_cleanup_draft(polygon_page, cancel_action):
    page = polygon_page
    page.evaluate("controller.setCtrlHeld(true); controller.arm()")
    _click_map(page, 100, 100)
    _click_map(page, 200, 100)

    if cancel_action == "escape":
        page.keyboard.press("Escape")
    elif cancel_action == "right_click":
        page.evaluate(
            """document.getElementById('map').dispatchEvent(
                new MouseEvent('contextmenu', {bubbles: true, button: 2})
            )"""
        )
    else:
        page.evaluate("controller.arm()")

    state = page.evaluate("controller.getState()")
    assert state["armed"] is False
    assert state["drawing"] is False
    assert state["vertices"] == []
    assert page.locator("#overlay").evaluate("el => el.style.display") == "none"


def test_pan_does_not_add_vertex_and_repeated_sessions_work(polygon_page):
    page = polygon_page
    page.evaluate("controller.setCtrlHeld(true); controller.arm()")
    page.evaluate(
        """() => {
            const map = document.getElementById('map');
            const rect = map.getBoundingClientRect();
            map.dispatchEvent(new PointerEvent('pointerdown', {
                bubbles: true, button: 0,
                clientX: rect.left + 100, clientY: rect.top + 100, ctrlKey: true,
            }));
            map.dispatchEvent(new PointerEvent('pointerup', {
                bubbles: true, button: 0,
                clientX: rect.left + 180, clientY: rect.top + 170, ctrlKey: true,
            }));
        }"""
    )
    assert page.evaluate("controller.getState().vertices.length") == 0

    page.evaluate("controller.cancel(); controller.arm()")
    _click_map(page, 120, 120)
    assert page.evaluate("controller.getState().vertices") == [[120, 120]]


def test_touch_toggle_allows_vertices_without_pointer_ctrl(polygon_page):
    page = polygon_page
    page.evaluate("controller.arm()")

    page.evaluate("controller.setCtrlHeld(true)")
    _click_map(page, 100, 100, ctrl=False)
    _click_map(page, 180, 100, ctrl=False)
    assert page.evaluate("controller.getState().vertices") == [[100, 100], [180, 100]]

    page.evaluate("controller.setCtrlHeld(false)")
    _click_map(page, 180, 180, ctrl=False)
    assert page.evaluate("controller.getState().vertices.length") == 2
