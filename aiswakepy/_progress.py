"""Simple progress utilities for terminal and Jupyter notebook environments."""
from __future__ import annotations

import time

_SPIN = ["-", "/", "|", "\\"]


def _in_jupyter() -> bool:
    """Return True when running inside a Jupyter (ZMQ) kernel."""
    try:
        from IPython import get_ipython
        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


class Spinner:
    """Spinning progress indicator that works in both terminal and Jupyter.

    In terminal : updates the current line in place using ``\\r``.
    In Jupyter  : uses ``display(display_id=True)`` to update a single output
                  line without clearing any surrounding cell output.

    Two usage patterns:

    **Loop** (shows [n/total])::

        spinner = Spinner(total=n, desc="Shore impact")
        for i, item in enumerate(items):
            spinner.update(i + 1)
        spinner.done()

    **Blocking call** (no count, just elapsed + row count on completion)::

        spinner = Spinner(desc="load_ais")
        result = do_work()
        spinner.done(rows=len(result))
    """

    def __init__(
        self,
        total: int | None = None,
        desc: str = "",
        interval: float = 0.3,
    ) -> None:
        self._total = total
        self._desc = desc
        self._interval = interval
        self._t0 = time.perf_counter()
        self._last = -999.0
        self._idx = 0
        self._has_loop = False   # True once update() is called
        self._jupyter = _in_jupyter()
        self._handle = None

        msg = self._fmt(0, done=False)
        if self._jupyter:
            from IPython.display import HTML, display
            self._handle = display(
                HTML(f"<pre style='margin:0;font-family:monospace'>{msg}</pre>"),
                display_id=True,
            )
        else:
            print(f"\r{msg}", end="", flush=True)

    def _fmt(self, n: int, *, done: bool, rows: int | None = None) -> str:
        elapsed = time.perf_counter() - self._t0
        char = "\u2713" if done else _SPIN[self._idx % 4]  # ✓
        # Only show [n/total] if update() was called (i.e. it's a real loop)
        if self._has_loop and self._total is not None:
            count_s = f"  [{n}/{self._total}]"
        else:
            count_s = ""
        rows_s = f"  ({rows:,} rows)" if rows is not None else ""
        return f"  {char} {self._desc}{count_s}  {elapsed:.1f}s{rows_s}"

    def _render(self, msg: str, *, finish: bool) -> None:
        if self._jupyter:
            from IPython.display import HTML
            self._handle.update(
                HTML(f"<pre style='margin:0;font-family:monospace'>{msg}</pre>")
            )
        elif finish:
            print(f"\r{msg}")
        else:
            print(f"\r{msg}", end="", flush=True)

    def update(self, n: int) -> None:
        """Update the spinner. Throttled to at most once per ``interval`` seconds."""
        self._has_loop = True
        now = time.perf_counter()
        if now - self._last < self._interval:
            return
        self._last = now
        self._idx += 1
        self._render(self._fmt(n, done=False), finish=False)

    def done(self, n: int | None = None, rows: int | None = None) -> None:
        """Mark the spinner as complete and print the final ✓ line."""
        if n is None:
            n = self._total or 0
        self._render(self._fmt(n, done=True, rows=rows), finish=True)
