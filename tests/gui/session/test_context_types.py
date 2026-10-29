"""Structural library commands operate on isolated resource snapshots."""

from dataclasses import replace
from typing import Literal

import pytest
from zcu_tools.cfg_model import ConstWaveformCfg, PulseSpec
from zcu_tools.gui.expected_error import InvalidInputError
from zcu_tools.gui.session.context_types import LibraryChange
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument


@pytest.mark.parametrize(
    ("change", "updated", "field", "initial", "replacement"),
    [
        (
            LibraryChange(
                kind="module",
                action="put",
                path="Q1.pulses.pi",
                module=PulseSpec(
                    ch=0, nqz=1, freq=100.0, gain=0.2, waveform="Q1.waveforms.base"
                ),
            ),
            PulseSpec(ch=0, nqz=1, freq=100.0, gain=0.5, waveform="Q1.waveforms.base"),
            "gain",
            0.2,
            0.5,
        ),
        (
            LibraryChange(
                kind="waveform",
                action="put",
                path="Q1.waveforms.base",
                waveform=ConstWaveformCfg(length=1.0),
            ),
            ConstWaveformCfg(length=2.0),
            "length",
            1.0,
            2.0,
        ),
    ],
    ids=["module", "waveform"],
)
def test_library_change_applies_isolated_put_replace_rename_and_delete(
    change: LibraryChange,
    updated: PulseSpec | ConstWaveformCfg,
    field: str,
    initial: float,
    replacement: float,
) -> None:
    library = LibrarySnapshot(
        ModuleLibraryDocument(format="zcu.module-library", format_version="2.0")
    )
    original = change.apply(library)
    assert original.entry_value(change.path, kind=change.kind)[field] == initial
    with pytest.raises(KeyError):
        library.entry_value(change.path, kind=change.kind)

    update = (
        replace(change, module=updated)
        if isinstance(updated, PulseSpec)
        else replace(change, waveform=updated)
    )
    overwritten = update.apply(original)
    assert overwritten.entry_value(change.path, kind=change.kind)[field] == replacement
    same_path = replace(update, action="replace", new_path=change.path).apply(original)
    assert same_path.entry_value(change.path, kind=change.kind)[field] == replacement

    moved_path = change.path + "_moved"
    moved = replace(update, action="replace", new_path=moved_path).apply(original)
    assert moved.entry_value(moved_path, kind=change.kind)[field] == replacement
    with pytest.raises(KeyError):
        moved.entry_value(change.path, kind=change.kind)
    assert original.entry_value(change.path, kind=change.kind)[field] == initial

    renamed = LibraryChange(
        kind=change.kind, action="rename", path=moved_path, new_path=change.path
    ).apply(moved)
    assert renamed.entry_value(change.path, kind=change.kind)[field] == replacement
    with pytest.raises(KeyError):
        renamed.entry_value(moved_path, kind=change.kind)
    if change.kind == "module":
        assert (
            renamed.entry_value(change.path, kind="module")["waveform"]
            == "Q1.waveforms.base"
        )

    deleted = LibraryChange(kind=change.kind, action="delete", path=change.path).apply(
        renamed
    )
    with pytest.raises(KeyError):
        deleted.entry_value(change.path, kind=change.kind)
    assert renamed.entry_value(change.path, kind=change.kind)[field] == replacement

    with pytest.raises(KeyError):
        replace(update, action="replace", new_path=moved_path).apply(library)
    with pytest.raises(ValueError, match="destination already exists"):
        LibraryChange(
            kind=change.kind, action="rename", path=change.path, new_path=moved_path
        ).apply(change.apply(moved))


@pytest.mark.parametrize("kind", ["module", "waveform"])
@pytest.mark.parametrize("action", ["put", "replace"])
def test_library_change_rejects_missing_spec(
    kind: Literal["module", "waveform"], action: Literal["put", "replace"]
) -> None:
    with pytest.raises(InvalidInputError, match="require a .* Spec"):
        LibraryChange(
            kind=kind,
            action=action,
            path="Q1.pulses.pi" if kind == "module" else "Q1.waveforms.base",
            new_path="Q1.pulses.pi" if action == "replace" else None,
        )
