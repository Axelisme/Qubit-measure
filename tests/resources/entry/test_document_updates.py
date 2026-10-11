"""Memory-only document updates and local source acceptance without mutation leaks."""

from collections.abc import Generator
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ConfigDict, Field, ValidationError
from zcu_tools.format_version import YamlMap
from zcu_tools.resources.entry import (
    AcceptedPayload,
    AcceptedWrite,
    ComponentSchema,
    LedgerEvent,
    Provenance,
    accept_source,
    document_meta,
    document_value,
    manual_update,
    snapshot,
    source_path,
)
from zcu_tools.resources.entry.schema import PointDocument

from .view_support import create_entry, registered_model


@pytest.fixture
def memory_point(registry_state_guard: None) -> Generator[PointDocument]:
    class Measured(ComponentSchema):
        model_config = ConfigDict(extra="forbid")
        freq: float = Field(gt=0)
        wiring: dict[str, float]
        nullable: float | None = None

    with registered_model("notebook/memory-point", Measured) as kind:
        yield PointDocument.model_validate(
            {
                "format": "zcu.parameter-container",
                "format_version": "1.0",
                "general": {"created_at": "2026-10-06T00:00:00Z"},
                "components": {
                    "N1": {"kind": kind, "freq": 10.0, "wiring": {"delay": 0.2}}
                },
                "provenance": {
                    "N1.wiring.delay": {
                        "source": "manual",
                        "kind": None,
                        "run_id": None,
                        "at": "2026-10-05T00:00:00Z",
                        "stderr": None,
                        "cloned_from": {
                            "entry_id": "9d25d7f2-f8c2-4893-bd36-1c27f8f48b61",
                            "point": "old",
                        },
                    }
                },
            }
        )


@pytest.mark.parametrize("stderr", [None, 0.0, 0.125])
def test_manual_update_prepares_isolated_working_value_and_source(
    memory_point: PointDocument, stderr: float | None
) -> None:
    before = memory_point.model_copy(deep=True)
    started = datetime.now(timezone.utc)
    updated = manual_update(memory_point, "N1.freq", 12.0, stderr=stderr)
    finished = datetime.now(timezone.utc)
    assert document_value(updated, "N1.freq") == 12.0
    source = document_meta(updated, "N1.freq")
    assert source is not None
    assert source.source == "manual"
    assert source.kind is None and source.run_id is None
    assert source.stderr == stderr and source.cloned_from is None
    assert started <= datetime.fromisoformat(source.at) <= finished
    assert updated.general.created_at == before.general.created_at
    assert updated.provenance["N1.wiring.delay"] == before.provenance["N1.wiring.delay"]
    assert memory_point == before
    updated.components.clear()
    updated.provenance.clear()
    assert memory_point == before


def test_manual_update_accepts_container_leaves_and_preserves_input(
    memory_point: PointDocument,
) -> None:
    value: YamlMap = {"delay": 0.4, "offset": 0.1}
    updated = manual_update(memory_point, "N1.wiring", value, stderr=0.02)
    value["delay"] = 99.0
    assert document_value(updated, "N1.wiring") == {"delay": 0.4, "offset": 0.1}
    for path in ("N1.wiring", "N1.wiring.delay", "N1.wiring.offset"):
        source = document_meta(updated, path)
        assert source is not None
        assert source.source == "manual" and source.stderr == 0.02
        assert source.cloned_from is None
    assert memory_point.provenance["N1.wiring.delay"]["cloned_from"] is not None
    general = manual_update(updated, "general.ext.temperature", 0.03)
    assert document_value(general, "general.ext.temperature") == 0.03
    assert document_meta(general, "general.ext.temperature") is not None
    assert document_value(updated, "general.ext") == {}
    assert general.general.created_at == memory_point.general.created_at


@pytest.mark.parametrize(
    ("path", "value", "error"),
    [
        ("", 1.0, ValueError),
        ("N1", 1.0, ValueError),
        ("N1..freq", 1.0, ValueError),
        ("missing.freq", 1.0, AttributeError),
        ("N1.freq.child", 1.0, AttributeError),
        ("N1.freq", -1.0, ValidationError),
        ("N1.unknown", 1.0, ValidationError),
        ("general.created_at", "not-a-time", ValidationError),
    ],
)
def test_manual_update_rejects_invalid_write_without_mutating_input(
    memory_point: PointDocument, path: str, value: float | str, error: type[Exception]
) -> None:
    before = memory_point.model_copy(deep=True)
    with pytest.raises(error):
        manual_update(memory_point, path, value)
    assert memory_point == before


@pytest.mark.parametrize("stderr", [-0.1, float("nan"), float("inf"), -float("inf")])
def test_manual_update_rejects_invalid_stderr_without_mutating_input(
    memory_point: PointDocument, stderr: float
) -> None:
    before = memory_point.model_copy(deep=True)
    with pytest.raises(ValueError, match="stderr"):
        manual_update(memory_point, "N1.freq", 12.0, stderr=stderr)
    assert memory_point == before


@pytest.mark.parametrize("owner", ["setup", "point"])
def test_accept_source_attaches_staged_value_without_rerunning_model_hooks(
    tmp_path: Path, registry_state_guard: None, owner: str
) -> None:
    hooks: list[str] = []

    class Measured(ComponentSchema):
        freq: float
        wiring: dict[str, float]

        def model_post_init(self, context: object) -> None:
            hooks.append("validated")

    with registered_model("notebook/source-stage", Measured) as kind:
        entry, _, _ = create_entry(tmp_path)
        entry.setup.add_component("N1", kind=kind, freq=10.0, wiring={"delay": 0.2})
        view = entry.setup if owner == "setup" else entry.new_point("working")
        before_disk = source_path(view).read_bytes()
        with view.edit() as draft:
            draft.set("N1.wiring", {"delay": 0.4, "offset": 0.1})
            before_source = snapshot(draft)
            event = LedgerEvent(
                id="92f1a2ca-71bf-4fa5-98eb-ff77d29f091a",
                kind="accepted",
                at="2026-10-06T00:00:00Z",
                entry_id=entry.entry_id,
                origin="notebook",
                payload=AcceptedPayload(
                    writes=(AcceptedWrite(path="N1.wiring.delay", value=0.4),),
                ),
            )
            entry.ledger.append(event)
            provenance = Provenance(event.id, "fit", None, event.at, 0.03)
            before_hooks = hooks.copy()
            accept_source(draft, "N1.wiring", provenance=provenance)
            captured = snapshot(draft)
            assert captured.components == before_source.components
            assert captured.general == before_source.general
            assert hooks == before_hooks
            for path in ("N1.wiring", "N1.wiring.delay", "N1.wiring.offset"):
                assert document_meta(captured, path) == provenance
            assert source_path(view).read_bytes() == before_disk
        assert view.meta("N1.wiring.delay") == provenance
        assert view.meta("N1.wiring.offset") == provenance
        view.refresh()
        assert view.meta("N1.wiring.delay") == provenance


@pytest.mark.parametrize("owner", ["setup", "point"])
@pytest.mark.parametrize(
    ("path", "source", "error"),
    [
        ("", "manual", ValueError),
        ("N1..rate", "manual", ValueError),
        ("N1.missing", "manual", AttributeError),
        ("N1.rate", "absent-event", ValueError),
    ],
)
def test_accept_source_rejects_invalid_path_or_source_without_changing_draft(
    tmp_path: Path, owner: str, path: str, source: str, error: type[Exception]
) -> None:
    entry, _, _ = create_entry(tmp_path)
    entry.setup.add_component("N1", kind="fake/drive/a", rate=10.0)
    view = entry.setup if owner == "setup" else entry.new_point("working")
    with view.edit() as draft:
        before = snapshot(draft)
        with pytest.raises(error) as caught:
            accept_source(
                draft,
                path,
                provenance=Provenance(source, None, None, "2026-10-06T00:00:00Z", None),
            )
        if source == "absent-event":
            assert isinstance(caught.value.__cause__, KeyError)
        assert snapshot(draft) == before


@pytest.mark.parametrize(
    "path", ["N1.ext.notes", "N1.ext.notes.level", "general.ext.nullable"]
)
def test_accept_source_clears_clone_origin_and_accepts_explicit_null(
    tmp_path: Path, path: str
) -> None:
    entry, _, _ = create_entry(tmp_path)
    entry.setup.add_component("N1", kind="fake/drive/a", rate=10.0)
    original = entry.new_point("original")
    with original.edit() as draft:
        draft.set("N1.ext.notes", {"level": 1.0})
        draft.set("general.ext.nullable", None)
    original_before = snapshot(original)
    clone = entry.new_point("clone", clone_from="original")
    before = snapshot(clone)
    assert before.provenance[path]["cloned_from"] is not None
    source = Provenance("manual", None, None, "2026-10-06T00:00:00Z", 0.0)
    with clone.edit() as draft:
        accept_source(draft, path, provenance=source)
        after = snapshot(draft)
        assert after.components == before.components
        assert after.general == before.general
        assert "cloned_from" not in after.provenance[path]
        assert after.provenance[path]["stderr"] == 0.0
        if path == "N1.ext.notes":
            assert "cloned_from" not in after.provenance["N1.ext.notes.level"]
        else:
            assert after.provenance["N1.ext.notes"] == before.provenance["N1.ext.notes"]
    clone.refresh()
    assert snapshot(clone).provenance[path] == after.provenance[path]
    assert document_value(snapshot(clone), "general.ext.nullable") is None
    assert snapshot(original) == original_before
