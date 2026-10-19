"""Transfer result folders: configure once, then upload or download via CLI."""

from pathlib import Path

from zcu_tools.utils.result_transfer.cli import main

if __name__ == "__main__":
    raise SystemExit(main(Path(__file__).resolve().parent.parent / "results"))
