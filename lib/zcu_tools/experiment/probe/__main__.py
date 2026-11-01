"""Run the existing offline probe CLI."""

from . import main

raise SystemExit(0 if main().exception is None else 1)
