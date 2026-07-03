#!/usr/bin/env python3
from __future__ import annotations

import runpy
from pathlib import Path

MAIN_SCRIPT = (
    Path(__file__).resolve().parents[4]
    / ".agents"
    / "skills"
    / "orchestrate"
    / "scripts"
    / "merge_queue.py"
)

runpy.run_path(str(MAIN_SCRIPT), run_name="__main__")
