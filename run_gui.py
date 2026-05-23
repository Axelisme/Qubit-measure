"""Quick launcher for the v2 GUI (Phase 8 manual test).

Usage:
    uv run python run_gui.py           # log → gui_debug.log (+ stderr)
    uv run python run_gui.py --no-log  # no file log
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Make sure lib/ is on the path when running from the repo root
sys.path.insert(0, str(Path(__file__).parent / "lib"))

LOG_FILE = Path(__file__).parent / "gui_debug.log"
LOG_FORMAT = "%(asctime)s.%(msecs)03d [%(levelname)-7s] %(name)s: %(message)s"
LOG_DATE = "%H:%M:%S"


def _setup_logging(to_file: bool = True) -> None:
    """Configure root logger: DEBUG to file, WARNING to stderr."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # stderr — WARNING and above only (keeps terminal clean)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE))
    root.addHandler(stderr_handler)

    if to_file:
        # File — full DEBUG trace for all zcu_tools.gui modules
        file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE))
        # Narrow scope to gui modules only to avoid noise from other subsystems
        gui_logger = logging.getLogger("zcu_tools.gui")
        gui_logger.addHandler(file_handler)
        gui_logger.setLevel(logging.DEBUG)
        # Also capture v2_gui app-level logs
        app_logger = logging.getLogger("zcu_tools.experiment.v2_gui")
        app_logger.addHandler(file_handler)
        app_logger.setLevel(logging.DEBUG)

        print(f"[run_gui] Logging DEBUG output to: {LOG_FILE}", file=sys.stderr)


if __name__ == "__main__":
    no_log = "--no-log" in sys.argv
    _setup_logging(to_file=not no_log)

    from zcu_tools.gui.mpl_backend_setup import configure_gui_matplotlib_backend

    configure_gui_matplotlib_backend()

    from zcu_tools.experiment.v2_gui.app import run_app

    run_app()
