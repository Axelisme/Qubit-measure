"""Execute the existing offline probe command-line entry."""

from . import main

raise SystemExit(0 if main().exception is None else 1)
