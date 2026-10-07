"""Entry-scoped memory candidates retain field and local source validation."""

from pathlib import Path
from uuid import uuid4

import pytest
from zcu_tools.cfg_model import ConstWaveformCfg
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument
from zcu_tools.resources.entry import (
    ResultEntry,
    document_meta,
    document_value,
    snapshot,
)
from zcu_tools.resources.entry.ledger_models import (
    AcceptedPayload,
    AcceptedWrite,
    LedgerEvent,
)
from zcu_tools.resources.entry.provenance import Provenance

AT = "2026-10-07T00:00:00Z"


def _source(entry: ResultEntry) -> Provenance:
    event_id = str(uuid4())
    entry.ledger.append(
        LedgerEvent(
            id=event_id,
            kind="accepted",
            at=AT,
            entry_id=entry.entry_id,
            origin="gui",
            payload=AcceptedPayload(
                writes=(AcceptedWrite(path="general.flux_value", value=0.2),),
                analysis_summary={},
            ),
        )
    )
    return Provenance(event_id, "fit", None, AT, 0.01)


@pytest.mark.parametrize("explicit_source", [False, True], ids=["manual", "accepted"])
def test_entry_prepares_isolated_point_candidate_with_local_source(
    entry: ResultEntry, explicit_source: bool
) -> None:
    point = entry.new_point("a")
    original = snapshot(point)
    source = _source(entry) if explicit_source else None
    updated = entry.updated_point(original, "general.ext", {"gain": 0.2}, source=source)
    assert document_value(updated, "general.ext.gain") == 0.2
    provenance = document_meta(updated, "general.ext.gain")
    assert provenance is not None
    if source is None:
        assert provenance.source == "manual"
    else:
        assert provenance == source
    assert updated.general.created_at == original.general.created_at
    assert document_value(original, "general.ext") == {}
    updated.general.ext["gain"] = 99.0
    assert document_value(original, "general.ext") == {}
    assert snapshot(entry.use_point("a")).model_dump() == original.model_dump()


def test_entry_point_candidate_rejects_unknown_local_source_without_mutation(
    entry: ResultEntry,
) -> None:
    original = snapshot(entry.new_point("a"))
    before = original.model_copy(deep=True)
    bad = Provenance(str(uuid4()), "fit", None, AT, 0.01)
    with pytest.raises(ValueError, match="source.*does not exist") as error:
        entry.updated_point(original, "general.flux_value", 0.2, source=bad)
    assert error.value.__cause__ is not None
    assert original == before
    assert snapshot(entry.use_point("a")).model_dump() == before.model_dump()


@pytest.mark.parametrize("known", [False, True], ids=["unknown", "local"])
def test_memory_library_validation_uses_this_entry_ledger(
    entry: ResultEntry, known: bool
) -> None:
    source = (
        _source(entry) if known else Provenance(str(uuid4()), "fit", None, AT, 0.01)
    )
    library = LibrarySnapshot(
        ModuleLibraryDocument(format="zcu.module-library", format_version="1.0")
    )
    candidate = library.replaced_waveform(
        "Q1.waveforms.base", ConstWaveformCfg(length=0.2), provenance=source
    ).document()
    before = candidate.model_copy(deep=True)
    if known:
        assert entry.validate_library(candidate) is None
    else:
        with pytest.raises(ValueError, match="source.*does not exist") as error:
            entry.validate_library(candidate)
        assert error.value.__cause__ is not None
    assert candidate == before
    assert library.waveform_keys() == ()


def test_memory_library_rejects_source_from_another_entry(
    entry: ResultEntry, entry_roots: tuple[Path, Path]
) -> None:
    results, database = entry_roots
    other = ResultEntry.create("other", result_root=results, database_root=database)
    source = _source(other)
    library = LibrarySnapshot(
        ModuleLibraryDocument(format="zcu.module-library", format_version="1.0")
    )
    candidate = library.replaced_waveform(
        "Q1.waveforms.base", ConstWaveformCfg(length=0.2), provenance=source
    ).document()
    with pytest.raises(ValueError, match="source.*does not exist"):
        entry.validate_library(candidate)
