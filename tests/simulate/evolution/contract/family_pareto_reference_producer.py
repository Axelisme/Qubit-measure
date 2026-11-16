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
    FrozenRecord,
    ModelSource,
    ReferenceResult,
    TimelineBatch,
    load_reference,
    read_frozen_index,
    read_physical_input,
    read_scene_records,
    reference_key,
    write_reference,
)

from .interaction_reference import interaction_reference
from .reference import converged_reference


def prepare_references(
    inputs: Path,
    cache: Path,
    *,
    scene_cache: Path,
    solver: Callable[[ModelSource, TimelineBatch], ReferenceResult],
) -> dict[str, str]:
    """Prepare one index with explicit scene/cache paths and independent solver.

    Read only record name and physical; metadata never selects a solver.
    Existing keys are loaded cache-only. Only a missing key triggers the given
    solver, without trying another solver after failure. Each decoded source
    is closed on success or failure; earlier writes remain after later failure.
    Container, numeric, solver and I/O errors propagate. Run separately under
    approved compute limits, never during timing.
    """
    return _prepare_records(_read_scenes(inputs, scene_cache), cache, solver)


def _read_scenes(index: Path, scene_cache: Path) -> tuple[FrozenRecord, ...]:
    try:
        return read_scene_records(scene_cache, read_frozen_index(index))
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"{error}\nPrepare scenes with: python -m tools.family_pareto_scenarios "
            f"--scene-cache {scene_cache} (use --scenario NAME for one indexed scene)"
        ) from error


def _prepare_records(
    records: tuple[FrozenRecord, ...],
    cache: Path,
    solver: Callable[[ModelSource, TimelineBatch], ReferenceResult],
) -> dict[str, str]:
    prepared = {}
    for record in records:
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
    """Prepare explicit --converged/--interaction indexes into --cache.

    --scene-cache is required and explicit; no Git discovery or scene rebuilding.
    Load all indexed numeric files before any solving. A missing scene lists its
    name, hashes and producer command, then fails without running a solver.

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
        help="Frozen index for the independent lab-frame solver (repeatable)",
    )
    parser.add_argument(
        "--interaction",
        type=Path,
        action="append",
        default=[],
        help="Frozen index for the interaction-picture solver (repeatable)",
    )
    parser.add_argument(
        "--cache", type=Path, required=True, help="Reference NPZ cache destination"
    )
    parser.add_argument("--scene-cache", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.converged and not args.interaction:
        parser.error("at least one --converged or --interaction index is required")
    selected = [
        (_read_scenes(path, args.scene_cache), solver)
        for paths, solver in (
            (args.converged, converged_reference),
            (args.interaction, interaction_reference),
        )
        for path in paths
    ]
    prepared = {}
    for records, solver in selected:
        prepared.update(_prepare_records(records, args.cache, solver))
    print(json.dumps(prepared, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
