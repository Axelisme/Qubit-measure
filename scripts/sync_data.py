"""Synchronize this checkout's result and Database trees."""

from pathlib import Path

from zcu_tools.utils.data_sync.cli import main

if __name__ == "__main__":
    raise SystemExit(main(checkout=Path(__file__).resolve().parents[1]))
