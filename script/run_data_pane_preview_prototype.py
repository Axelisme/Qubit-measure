#!/usr/bin/env python
# THROWAWAY LAUNCHER — measure-save-preview-gallery / 001-prototype-data-pane-preview-layouts
# NOT FOR PRODUCTION. Launches the synthetic Data-pane preview prototype only.
# No hardware, result files, persistence, or network effects. All Save/Load
# actions inside the prototype are inert.
"""
Launcher for the Data-pane preview throwaway Qt prototype.

One documented command from the repository checkout:

    .venv/bin/python script/run_data_pane_preview_prototype.py
    .venv/bin/python script/run_data_pane_preview_prototype.py --variant B
    QT_QPA_PLATFORM=offscreen .venv/bin/python script/run_data_pane_preview_prototype.py --smoke

Variants:
  A — Stacked Right Rail (save left, three previews stacked right)
  B — Inline Cards (save+preview paired per artifact, right pane idle)
  C — Focus Tabs (save left, single large tabbed preview right)

The prototype is throwaway and lives beside the Data-pane UI it prototypes
(lib/zcu_tools/gui/app/main/ui/prototype_data_pane_preview.py). Production
Artifacts and figure ownership are untouched.

Known limitations: synthetic matplotlib figures only, read-only previews,
no figure editing, no real analysis controls, no persistence — the status
and paths shown are local to the prototype session.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
# Worktree-aware import: ensure this checkout's lib/ is on sys.path ahead of
# the editable install (which points at the main checkout's lib/).
_lib = str(PROJECT_ROOT / "lib")
if _lib not in sys.path:
    sys.path.insert(0, _lib)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_data_pane_preview_prototype",
        description="Launch the throwaway Data-pane preview prototype (A/B/C).",
        epilog=(
            "All previews are synthetic and all save actions are inert. "
            "Use the pill or ←/→ to cycle variants; path/comment edits are local only."
        ),
    )
    p.add_argument(
        "--variant",
        choices=["A", "B", "C", "a", "b", "c", "0", "1", "2"],
        default="A",
        help="Initial variant to show (default: A). Also switchable inside the window.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Bounded startup smoke: show the window offscreen and auto-quit (for CI/gates).",
    )
    return p.parse_args(argv)


def _variant_to_index(value: str) -> int:
    v = str(value).strip().upper()
    if v in ("0", "A"):
        return 0
    if v in ("1", "B"):
        return 1
    if v in ("2", "C"):
        return 2
    raise ValueError(f"unknown variant {value!r}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    initial = _variant_to_index(args.variant)

    # Qt must be imported after the environment is set; the caller controls
    # QT_QPA_PLATFORM (offscreen for smoke). No other environment required.
    from qtpy.QtCore import QTimer  # type: ignore[attr-defined]
    from qtpy.QtWidgets import QApplication  # type: ignore[attr-defined]

    app = QApplication.instance()
    created_app = False
    if app is None:
        app = QApplication(sys.argv)
        created_app = True

    # Import prototype after QApplication exists (safe for FigureCanvas).
    from zcu_tools.gui.app.main.ui.prototype_data_pane_preview import (
        PrototypeMainWindow,
    )

    win = PrototypeMainWindow(initial_variant=initial)
    win.show()

    if args.smoke:
        # Auto-quit quickly so the smoke gate terminates without user input.
        QTimer.singleShot(900, app.quit)  # type: ignore[attr-defined]

    if created_app:
        return int(app.exec())  # type: ignore[attr-defined]
    # If an app already existed (e.g. embedded), just ensure the window is visible.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
