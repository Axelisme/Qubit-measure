"""Transfer result folders: configure once, then upload or download via CLI."""

from pathlib import Path

from zcu_tools.utils.result_transfer.cli import main

if __name__ == "__main__":
    checkout = Path(__file__).resolve().parent.parent
    raise SystemExit(main(checkout / "result", checkout / ".result-transfer"))
