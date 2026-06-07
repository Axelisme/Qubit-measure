"""Launcher for the autofluxdep workflow GUI (Phase C prototype).

Opens the node-list / node-detail window. Prototype: Setup builds fake resources
and Run drives the orchestrator on synthetic data — no hardware. Use it to
exercise the interaction (add/remove/reorder/rename Nodes, Setup→Run enable,
edit↔run lock, auto-follow, Run/Stop toggle, global flux progress).

DEBUG output is written to ``autofluxdep_gui_debug.log`` at the repo root (the
whole run loop: each flux point, each Node entered / skipped / fitted). Disable
with ``--no-log`` or redirect with ``--log-file``.

Example:
    uv run python script/run_autofluxdep_gui.py
    uv run python script/run_autofluxdep_gui.py --no-log
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

LOG_FILE = Path(__file__).parent.parent / "autofluxdep_gui_debug.log"
LOG_FORMAT = "%(asctime)s.%(msecs)03d [%(levelname)-7s] %(name)s: %(message)s"
LOG_DATE = "%H:%M:%S"


def _setup_logging(to_file: bool = True, log_file: Path = LOG_FILE) -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE))
    root.addHandler(stderr_handler)

    if to_file:
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE))
        log = logging.getLogger("zcu_tools.gui.app.autofluxdep")
        log.addHandler(file_handler)
        log.setLevel(logging.DEBUG)
        print(
            f"[run_autofluxdep_gui] Logging DEBUG output to: {log_file}",
            file=sys.stderr,
        )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_autofluxdep_gui",
        description="Launch the autofluxdep workflow GUI",
    )
    parser.add_argument("--no-log", action="store_true", help="Disable file logging")
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Override the DEBUG log file path",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    log_file = Path(args.log_file) if args.log_file else LOG_FILE
    _setup_logging(to_file=not args.no_log, log_file=log_file)

    # Force a NON-interactive backend BEFORE any pyplot import. Unlike
    # measure-gui / fluxdep, autofluxdep does NOT use the custom
    # ``module://zcu_tools.gui.plotting.backend`` (that backend is for the
    # worker-draws-then-marshals model — ADR-0017). autofluxdep's worker never
    # touches matplotlib; the main thread embeds each Node's figure with a bare
    # ``FigureCanvasQTAgg`` (ADR-0017). With ``Agg`` selected, pyplot can never
    # pop a detached window, while the directly-constructed embed canvases work
    # as normal. (Selecting the custom pyplot backend here would route stray
    # pyplot figures to detached windows — the bug this avoids.)
    import matplotlib

    matplotlib.use("Agg")

    from zcu_tools.gui.app.autofluxdep.app import run_app

    run_app()
