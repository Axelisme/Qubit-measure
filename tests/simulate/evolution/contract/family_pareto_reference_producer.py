"""Tests-owned independent reference preparation from frozen numeric inputs.

This is the only preparation entry point. Measurement tools consume cache-only
references through evolution and never import this module (ADR-0078).
"""

import argparse
import json
from collections.abc import Callable
from contextlib import closing
from pathlib import Path

from zcu_tools.simulate.evolution import (
    ModelSource,
    ReferenceResult,
    TimelineBatch,
    load_reference,
    read_frozen_manifest,
    read_physical_input,
    reference_key,
    write_reference,
)

from .interaction_reference import interaction_reference
from .reference import converged_reference


def prepare_references(
    inputs: Path,
    cache: Path,
    *,
    solver: Callable[[ModelSource, TimelineBatch], ReferenceResult],
) -> dict[str, str]:
    """Prepare one manifest with the caller's explicit independent solver.

    Read only record name and physical; metadata never selects a solver.
    Existing keys are loaded cache-only. Only a missing key triggers the given
    solver, without trying another solver after failure. Each decoded source
    is closed on success or failure; earlier writes remain after later failure.
    Container, numeric, solver and I/O errors propagate. Run separately under
    approved compute limits, never during timing.
    """
    prepared = {}
    for record in read_frozen_manifest(inputs):
        frozen = read_physical_input(record.physical)
        with closing(frozen.source):
            try:
                load_reference(cache, frozen.source, frozen.batch)
            except FileNotFoundError:
                result = solver(frozen.source, frozen.batch)
                write_reference(cache, frozen.source, frozen.batch, result)
            prepared[record.name] = reference_key(frozen.source, frozen.batch)
    return prepared


def main(argv: list[str] | None = None) -> int:
    """Prepare explicit --converged/--interaction manifests into --cache.

    Each solver option accepts one path and may repeat. At least one is required.
    argv omits the executable name. Print name/key JSON; argparse handles invalid
    options/help. Input, solver and I/O failures propagate without solver fallback.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--converged",
        type=Path,
        action="append",
        default=[],
        help="Frozen manifest for the independent lab-frame solver (repeatable)",
    )
    parser.add_argument(
        "--interaction",
        type=Path,
        action="append",
        default=[],
        help="Frozen manifest for the interaction-picture solver (repeatable)",
    )
    parser.add_argument(
        "--cache", type=Path, required=True, help="Reference NPZ cache destination"
    )
    args = parser.parse_args(argv)
    if not args.converged and not args.interaction:
        parser.error("at least one --converged or --interaction manifest is required")
    prepared = {}
    for paths, solver in (
        (args.converged, converged_reference),
        (args.interaction, interaction_reference),
    ):
        for path in paths:
            prepared.update(prepare_references(path, args.cache, solver=solver))
    print(json.dumps(prepared, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
