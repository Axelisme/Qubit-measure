from __future__ import annotations

import argparse
from pathlib import Path

from zcu_tools.experiment.v2_gui import launch_mock_gui

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start v2 GUI in fake-only mode")
    parser.add_argument(
        "--backend",
        choices=["mock"],
        default="mock",
        help="Backend mode. Fake-only phase supports mock only.",
    )
    args = parser.parse_args()
    launch_mock_gui(str(Path(__file__).resolve().parents[1]), backend=args.backend)
