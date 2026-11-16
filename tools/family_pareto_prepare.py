"""Pinned scene preparation; numeric file formats stay in simulate."""

import json
from pathlib import Path

from zcu_tools.simulate.evolution import (
    FrozenIndexEntry,
    read_frozen_index,
    read_scene_records,
    write_frozen_index,
    write_scene_record,
)

from tools.family_pareto_inputs import Input, freeze_records
from tools.family_pareto_storage import indexed_scenes


def _select_inventory(
    inputs: list[Input],
    pinned: dict[str, tuple[FrozenIndexEntry, ...]],
    scenario: str | None,
) -> list[Input]:
    """Validate category/name sets before any selected builder is invoked."""
    selected = [item for item in inputs if scenario is None or item.name == scenario]
    if scenario is not None:
        matches = [
            (category, entry)
            for category, entries in pinned.items()
            for entry in entries
            if entry.name == scenario
        ]
        if len(matches) != 1 or len(selected) != 1:
            raise ValueError(
                f"scene inventory missing or ambiguous scenario: {scenario}"
            )
        if selected[0].category != matches[0][0]:
            raise ValueError(f"scene inventory category mismatch: {scenario}")
    else:
        unexpected_categories = {item.category for item in selected} - pinned.keys()
        if unexpected_categories:
            raise ValueError(
                f"scene inventory unexpected category: {sorted(unexpected_categories)}"
            )
        for category, entries in pinned.items():
            expected = {entry.name for entry in entries}
            names = [item.name for item in selected if item.category == category]
            actual = set(names)
            if actual != expected or len(names) != len(actual):
                raise ValueError(
                    f"scene inventory {category}: missing={sorted(expected - actual)}; "
                    f"unexpected={sorted(actual - expected)}; duplicate={len(names) != len(actual)}"
                )
    return selected


def prepare_scenes(
    inputs: list[Input],
    indexes: dict[str, Path],
    scene_cache: Path,
    *,
    scenario: str | None = None,
    update_index: bool = False,
) -> tuple[FrozenIndexEntry, ...]:
    """Rebuild selected declarations, checking pins before publishing numeric data.

    indexes explicitly maps each declaration category to its index. scene_cache
    is an explicit path. scenario selects exactly one declaration and index row;
    otherwise the inventory must match each complete index name set. Only an
    explicit update_index changes selected identities, preserving unselected
    rows/order. No rows are implicitly added, deleted or moved between indexes.
    All selected builders must succeed before any index is published. Numeric
    files already published remain after a later failure; existing pinned files
    are never replaced by mismatching output. Each built source is closed.
    ValueError diagnoses inventory/identity disagreement. Codec, builder and I/O
    failures propagate. No solver or Engine runs. Return selected identities.
    """
    indexed_scenes(tuple(indexes.values()))
    pinned = {category: read_frozen_index(path) for category, path in indexes.items()}
    selected = _select_inventory(inputs, pinned, scenario)
    prepared: list[FrozenIndexEntry] = []
    replacements: dict[str, FrozenIndexEntry] = {}
    for item in selected:
        original = next(
            entry for entry in pinned[item.category] if entry.name == item.name
        )
        try:
            read_scene_records(scene_cache, (original,))
        except FileNotFoundError:
            status = "missing"
        else:
            status = "hit"
        print(
            json.dumps({"name": item.name, "cache": status, "action": "rebuild-check"}),
            flush=True,
        )
        (record,) = freeze_records([item])
        rebuilt = FrozenIndexEntry.from_record(record)
        if rebuilt != original and not update_index:
            raise ValueError(
                f"scene hash mismatch: {item.name}; pinned={original}; rebuilt={rebuilt}"
            )
        prepared.append(write_scene_record(scene_cache, record))
        replacements[item.name] = rebuilt
        print(
            json.dumps(
                {
                    "name": item.name,
                    "action": "updated" if rebuilt != original else "verified",
                    "cache": "filled" if status == "missing" else "hit",
                }
            ),
            flush=True,
        )
    if update_index:
        for category, path in indexes.items():
            entries = pinned[category]
            if any(entry.name in replacements for entry in entries):
                write_frozen_index(
                    path, (replacements.get(entry.name, entry) for entry in entries)
                )
    return tuple(prepared)
