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
    FrozenIndexEntry,
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

Solver = Callable[[ModelSource, TimelineBatch], ReferenceResult]


def prepare_references(
    inputs: Path,
    cache: Path,
    *,
    scene_cache: Path,
    solver: Solver,
    scenario: str | None = None,
) -> dict[str, str]:
    """Prepare one index with explicit scene/cache paths and independent solver.

    scenario requires exactly one matching indexed row before reading payloads;
    None selects the entire index. Zero/multiple matches raise ValueError.
    Read only selected record names and physical; metadata never selects a solver.
    Existing keys are loaded cache-only. Only a missing key triggers the given
    solver for secular-bath inputs, without trying another solver after failure.
    A miss for explicit collapse operators raises ValueError. Each decoded source
    is closed on success or failure; earlier writes remain after later failure.
    Container, numeric, solver and I/O errors propagate. Run separately under
    approved compute limits, never during timing.
    """
    [(entries, _)] = _select_indexes([(inputs, solver)], scenario)
    return _prepare_records(_read_scenes(entries, scene_cache), cache, solver)


def _select_indexes(
    sources: list[tuple[Path, Solver]], scenario: str | None
) -> list[tuple[tuple[FrozenIndexEntry, ...], Solver]]:
    indexed = [(path, read_frozen_index(path), solver) for path, solver in sources]
    if scenario is None:
        return [(entries, solver) for _, entries, solver in indexed]
    matches = [
        (path, entry, solver)
        for path, entries, solver in indexed
        for entry in entries
        if entry.name == scenario
    ]
    if not matches:
        legal = ", ".join(entry.name for _, entries, _ in indexed for entry in entries)
        raise ValueError(f"Unknown scenario: {scenario}. Legal names: {legal}")
    if len(matches) != 1:
        paths = ", ".join(str(path) for path, _, _ in matches)
        raise ValueError(f"Ambiguous scenario: {scenario}. Sources: {paths}")
    _, entry, solver = matches[0]
    return [((entry,), solver)]


def _read_scenes(
    entries: tuple[FrozenIndexEntry, ...], scene_cache: Path
) -> tuple[FrozenRecord, ...]:
    try:
        return read_scene_records(scene_cache, entries)
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
                levels = {tuple(int(v) for v in row) for row in frozen.batch.flux.level}
                levels.add((0,) * frozen.batch.flux.level.shape[1])
                if any(
                    frozen.source.model(level).collapse_operators for level in levels
                ):
                    raise ValueError(
                        f"{record.name}: the reference solvers do not support explicit collapse operators"
                    ) from None
                result = solver(frozen.source, frozen.batch)
                write_reference(cache, frozen.source, frozen.batch, result)
            prepared[record.name] = reference_key(frozen.source, frozen.batch)
    return prepared


def main(argv: list[str] | None = None) -> int:
    """Prepare explicit --converged/--interaction indexes into --cache.

    --scene-cache is required and explicit; no Git discovery or scene rebuilding.
    --scenario NAME requires one match across all supplied indexes before reading
    any payload; zero/multiple matches raise ValueError without solving. Omission
    selects every indexed row. Load selected numeric files before any solving.
    A missing scene lists its
    name, hashes and producer command, then fails without running a solver.

    Each solver option accepts one path and may repeat. At least one is required.
    The solvers support only secular baths; a full-L cache miss raises ValueError.
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
    parser.add_argument("--scenario")
    args = parser.parse_args(argv)
    if not args.converged and not args.interaction:
        parser.error("at least one --converged or --interaction index is required")
    sources = [
        (path, solver)
        for paths, solver in (
            (args.converged, converged_reference),
            (args.interaction, interaction_reference),
        )
        for path in paths
    ]
    selected = [
        (_read_scenes(entries, args.scene_cache), solver)
        for entries, solver in _select_indexes(sources, args.scenario)
    ]
    prepared = {}
    for records, solver in selected:
        prepared.update(_prepare_records(records, args.cache, solver))
    print(json.dumps(prepared, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
