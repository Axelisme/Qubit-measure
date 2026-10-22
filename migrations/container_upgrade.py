"""Explicit batch conversion using datafile's long-lived native reader."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from zcu_tools.datafile import load, save


def upgrade_containers(
    sources: Sequence[Path], *, output_dir: Path, replace: bool = False
) -> tuple[Path, ...]:
    """Read sources and write major-3 containers under output_dir by basename.

    Source files remain unchanged unless caller explicitly chooses their parent
    as output_dir and replace=True. Reject duplicate basenames before writing.
    Create output_dir if necessary. Existing outputs raise FileExistsError unless
    replace=True. All conversions reuse datafile.load/save, including detached
    opaque groups and normalized legacy attrs. Each output is atomic; a later
    failure does not roll back earlier outputs. Decode and I/O errors propagate.
    """
    destinations = tuple(output_dir / source.name for source in sources)
    if len(set(destinations)) != len(destinations):
        raise ValueError("duplicate source basenames cause destination collision")
    if not replace:
        for destination in destinations:
            if destination.exists():
                raise FileExistsError(destination)
    output_dir.mkdir(parents=True, exist_ok=True)
    for source, destination in zip(sources, destinations, strict=True):
        save(destination, load(source), replace=replace)
    return destinations


def main(argv: Sequence[str] | None = None) -> int:
    """Convert explicit source paths from argv; return 0 or propagate errors.

    Syntax: sources... --output-dir DIR [--replace]. No directory scanning,
    hardware access or implicit source cleanup occurs. argparse handles usage.
    """
    parser = argparse.ArgumentParser(
        description="Convert native 2.x files to containers"
    )
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)
    upgrade_containers(args.sources, output_dir=args.output_dir, replace=args.replace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
