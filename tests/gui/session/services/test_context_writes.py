"""Direct target writes through the shipped facet and real resource owners."""

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

import pytest
from pydantic import ValidationError
from zcu_tools.cfg_model import ConstWaveformCfg, PulseSpec
from zcu_tools.format_version import YamlMap, YamlValue
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.context_types import (
    ContextTarget,
    PointUseRequest,
    SavedPoint,
)
from zcu_tools.gui.session.events import (
    ContextSwitchedPayload,
    MdChangedPayload,
    MlChangedPayload,
)
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.resources.document_store import ConflictError
from zcu_tools.resources.entry import (
    EditView,
    PointView,
    ResultEntry,
    document_meta,
    snapshot,
)
from zcu_tools.resources.entry.ledger_models import (
    AcceptedPayload,
    AcceptedWrite,
    LedgerEvent,
)
from zcu_tools.resources.entry.provenance import Provenance
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

Kind = Literal["parameter", "module", "waveform"]
KINDS: tuple[Kind, ...] = ("parameter", "module", "waveform")
AT = "2026-10-07T00:00:00Z"


@dataclass
class _Session:
    entry: ResultEntry
    control: ContextControlFacet
    service: ContextService
    state: SessionState
    bus: BaseEventBus


@pytest.fixture
def session(tmp_path: Path) -> _Session:
    entry = ResultEntry.create(
        "entry", result_root=tmp_path / "Result", database_root=tmp_path / "Database"
    )
    state = SessionState(SessionEnv(parameters=None, soc=None, soccfg=None))
    bus = BaseEventBus()
    service = ContextService(state, bus, bind_assets=ArbWaveformDatabase.info_at)
    service.install_entry(entry)
    return _Session(entry, ContextControlFacet(context=service), service, state, bus)


def _write(
    control: ContextControlFacet,
    target: ContextTarget,
    kind: Kind,
    number: float,
    *,
    source: Provenance | None = None,
    metadata: YamlMap | None = None,
    path: str | None = None,
) -> None:
    destination = (
        path
        if path is not None
        else {
            "parameter": "general.ext.flux_value",
            "module": "Q1.pulses.pi",
            "waveform": "Q1.waveforms.base",
        }[kind]
    )
    if kind == "parameter":
        control.set_parameter(target, destination, number, metadata=metadata)
    elif kind == "module":
        control.set_module(
            target,
            destination,
            PulseSpec(
                ch=0, nqz=1, freq=100.0, gain=number, waveform="Q1.waveforms.missing"
            ),
            source=source,
        )
    else:
        control.set_waveform(
            target, destination, ConstWaveformCfg(length=number), source=source
        )


def _value(capture: ParameterContext, kind: Kind) -> YamlValue:
    if kind == "parameter":
        return capture.value("general.ext.flux_value")
    if kind == "module":
        return capture.library.entry_value("Q1.pulses.pi", kind="module")["gain"]
    return capture.library.entry_value("Q1.waveforms.base", kind="waveform")["length"]


def _library_meta(capture: ParameterContext, kind: Kind) -> Provenance | None:
    path = {"module": "Q1.pulses.pi.gain", "waveform": "Q1.waveforms.base.length"}[kind]
    return capture.library.meta(path)


def _select(session: _Session, label: str) -> SavedPoint:
    session.entry.new_point(label)
    return session.control.use_point(PointUseRequest(label))


@pytest.mark.parametrize("kind", KINDS)
def test_read_target_observes_detached_writes_without_selecting_or_publishing(
    session: _Session, kind: Kind
) -> None:
    first = _select(session, "a")
    _write(session.control, first, kind, 0.2)
    capture = session.control.read_context()
    second = _select(session, "b")
    _write(session.control, first, kind, 0.3)
    selected = session.control.read_context()
    version = session.state.version.get("context")
    events: list[object] = []
    session.bus.subscribe(MdChangedPayload, events.append)
    session.bus.subscribe(MlChangedPayload, events.append)
    current = session.control.read_target(first)
    assert current.target == first
    assert _value(current, kind) == pytest.approx(0.3)
    assert _value(capture, kind) == pytest.approx(0.2)
    assert session.control.read_context().target == second
    assert session.control.read_context().document() == selected.document()
    assert session.state.version.get("context") == version
    assert events == []


def test_read_target_rejects_other_entries(session: _Session) -> None:
    target = SavedPoint(entry_id="other", point="a")
    before = session.state.session_env
    with pytest.raises(FailedPreconditionError) as error:
        session.control.read_target(target)
    assert error.value.reason_code == "entry_not_open"
    assert session.state.session_env == before


def test_read_target_missing_point_does_not_create_it(session: _Session) -> None:
    target = SavedPoint(entry_id=session.entry.entry_id, point="missing")
    with pytest.raises(FileNotFoundError):
        session.control.read_target(target)
    assert session.entry.list_points() == []


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
                writes=(AcceptedWrite(path="general.ext.flux_value", value=0.2),),
            ),
        )
    )
    return Provenance(event_id, "fit", None, AT, 0.01)


def _persisted(entry: ResultEntry, target: SavedPoint, kind: Kind) -> YamlValue:
    if kind == "parameter":
        return snapshot(entry.use_point(target.point)).general.ext["flux_value"]
    library = entry.module_library(
        target.point,
        assets=ArbWaveformDatabase.info_at(entry.database_path / "arb_waveforms"),
    ).snapshot()
    if kind == "module":
        return library.entry_value("Q1.pulses.pi", kind="module")["gain"]
    return library.entry_value("Q1.waveforms.base", kind="waveform")["length"]


@pytest.mark.parametrize("metadata", [None, {}, {"stderr": 0.01, "source": "caller"}])
def test_parameter_setter_commits_caller_metadata_and_keeps_old_capture(
    session: _Session, metadata: YamlMap | None
) -> None:
    target = _select(session, "a")
    with session.entry.use_point("a").edit() as draft:
        draft.set_meta("general.ext.flux_value", {"comment": "old"})
    capture = session.control.read_context()
    old_document = capture.document()
    _write(session.control, target, "parameter", 0.2, metadata=metadata)
    current = session.control.read_context()
    assert _value(current, "parameter") == pytest.approx(0.2)
    expected = {"comment": "old"} if metadata is None else metadata
    assert document_meta(current.document(), "general.ext.flux_value") == expected
    stored = snapshot(session.entry.use_point("a"))
    assert document_meta(stored, "general.ext.flux_value") == expected
    assert capture.document() == old_document


@pytest.mark.parametrize("kind", ["module", "waveform"])
@pytest.mark.parametrize("explicit_source", [False, True], ids=["manual", "accepted"])
def test_library_setters_preserve_source_and_old_capture(
    session: _Session, kind: Kind, explicit_source: bool
) -> None:
    target = _select(session, "a")
    capture = session.control.read_context()
    old_document = capture.document()
    old_library = capture.library.document()
    created_at = old_document.general.created_at
    source = _source(session.entry) if explicit_source else None
    _write(session.control, target, kind, 0.2, source=source)
    current = session.control.read_context()
    assert _value(current, kind) == pytest.approx(0.2)
    provenance = _library_meta(current, kind)
    assert provenance is not None
    if source is None:
        assert provenance.source == "manual" and provenance.kind is None
    else:
        assert provenance == source
    assert provenance.cloned_from is None
    assert current.document().general.created_at == created_at
    assert capture.document() == old_document
    assert capture.library.document() == old_library
    assert _persisted(session.entry, target, kind) == pytest.approx(0.2)


@pytest.mark.parametrize("kind", KINDS)
def test_detached_saved_point_write_keeps_current_point_and_publication(
    session: _Session, kind: Kind
) -> None:
    first = _select(session, "a")
    second = _select(session, "b")
    capture = session.control.read_context()
    before_version = session.state.version.get("context")
    events: list[object] = []
    session.bus.subscribe(MdChangedPayload, events.append)
    session.bus.subscribe(MlChangedPayload, events.append)
    _write(session.control, first, kind, 0.2)
    assert _persisted(session.entry, first, kind) == pytest.approx(0.2)
    assert session.control.read_context().target == second
    assert session.control.read_context().document() == capture.document()
    assert (
        session.control.read_context().library.document() == capture.library.document()
    )
    assert session.state.version.get("context") == before_version
    assert events == []
    session.control.use_point(PointUseRequest("a"))
    assert _value(session.control.read_context(), kind) == pytest.approx(0.2)
    _write(session.control, first, kind, 0.3)
    assert _value(session.control.read_context(), kind) == pytest.approx(0.3)


@pytest.mark.parametrize("kind", KINDS)
def test_cross_entry_target_rejected_before_any_write(
    session: _Session, tmp_path: Path, kind: Kind
) -> None:
    old_target = _select(session, "a")
    other = ResultEntry.create(
        "other", result_root=tmp_path / "Result", database_root=tmp_path / "Database"
    )
    session.service.install_entry(other)
    current = session.state.session_env
    with pytest.raises(FailedPreconditionError) as error:
        _write(session.control, old_target, kind, 0.2)
    assert error.value.reason_code == "entry_not_open"
    assert session.state.session_env == current
    assert session.entry.list_points() == ["a"]
    assert other.list_points() == []


@pytest.mark.parametrize("kind", ["module", "waveform"])
def test_invalid_library_source_keeps_projection_and_version(
    session: _Session, kind: Kind
) -> None:
    target = _select(session, "a")
    capture = session.control.read_context()
    version = session.state.version.get("context")
    bad = Provenance(str(uuid4()), "fit", None, AT, 0.01)
    with pytest.raises(ValueError, match="source.*does not exist") as error:
        _write(session.control, target, kind, 0.2, source=bad)
    assert error.value.__cause__ is not None
    assert session.control.read_context().document() == capture.document()
    assert (
        session.control.read_context().library.document() == capture.library.document()
    )
    assert session.state.version.get("context") == version


def test_invalid_parameter_schema_is_not_published(session: _Session) -> None:
    target = _select(session, "a")
    capture = session.control.read_context()
    with pytest.raises(ValidationError, match="created_at"):
        session.control.set_parameter(target, "general.created_at", "bad-time")
    assert session.control.read_context().document() == capture.document()


@pytest.mark.parametrize("kind", KINDS)
def test_current_write_notifies_after_installation_without_switching_binding(
    session: _Session, kind: Kind
) -> None:
    target = _select(session, "a")
    binding = session.control.get_source_binding()
    version = session.state.version.get("context")
    observed: list[tuple[object, YamlValue, int]] = []

    def observe(payload: MdChangedPayload | MlChangedPayload) -> None:
        observed.append(
            (
                payload,
                _value(session.control.read_context(), kind),
                session.state.version.get("context"),
            )
        )

    session.bus.subscribe(MdChangedPayload, observe)
    session.bus.subscribe(MlChangedPayload, observe)
    _write(session.control, target, kind, 0.2)
    assert len(observed) == 1
    payload, value, installed_version = observed[0]
    expected = MdChangedPayload if kind == "parameter" else MlChangedPayload
    assert isinstance(payload, expected)
    assert binding is not None
    assert payload.binding_id == binding.binding_id
    assert payload.content_changed and payload.fault is None
    assert value == pytest.approx(0.2) and installed_version == version + 1
    assert session.control.get_source_binding() == binding


def test_detached_description_write_updates_cache_without_changing_selected_source(
    session: _Session,
) -> None:
    first = _select(session, "a")
    _select(session, "b")
    before = session.control.read_context()
    version = session.state.version.get("context")
    binding = session.control.get_source_binding()
    rows = session.control.get_alias_rows()
    descriptions = {
        info.label: info.description for info in session.control.list_points()
    }
    events: list[object] = []
    for event in (MdChangedPayload, MlChangedPayload, ContextSwitchedPayload):
        session.bus.subscribe(event, events.append)

    session.control.set_parameter(
        first, "general.description", "changed while detached"
    )

    after_descriptions = {
        info.label: info.description for info in session.control.list_points()
    }
    assert after_descriptions == {**descriptions, "a": "changed while detached"}
    assert (
        snapshot(session.entry.use_point("a")).general.description
        == "changed while detached"
    )
    after = session.control.read_context()
    assert after.target == before.target
    assert after.document() == before.document()
    assert after.library.document() == before.library.document()
    assert session.state.version.get("context") == version
    assert session.control.get_source_binding() == binding
    assert session.control.get_alias_rows() == rows
    assert events == []


def test_description_write_updates_cached_point_description(session: _Session) -> None:
    target = _select(session, "a")
    session.control.set_parameter(target, "general.description", "changed")
    assert session.control.list_points()[0].description == "changed"


@pytest.mark.parametrize("kind", KINDS)
def test_saved_setter_io_failure_keeps_publication(
    session: _Session, monkeypatch: pytest.MonkeyPatch, kind: Kind
) -> None:
    target = _select(session, "a")
    capture = session.control.read_context()
    version = session.state.version.get("context")

    def fail_replace(self: Path, target: str | Path) -> Path:
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="synthetic replace failure"):
        _write(session.control, target, kind, 0.2)
    assert session.control.read_context().document() == capture.document()
    assert (
        session.control.read_context().library.document() == capture.library.document()
    )
    assert session.state.version.get("context") == version


def test_parameter_conflict_propagates_without_publishing_candidate(
    session: _Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _select(session, "a")
    session.control.set_parameter(target, "general.ext.flux_value", 0.1)
    capture = session.control.read_context()
    version = session.state.version.get("context")
    other = session.entry.use_point("a")
    original = PointView.edit

    @contextmanager
    def interleaved(view: PointView) -> Generator[EditView]:
        with original(view) as draft:
            yield draft
            with original(other) as external:
                external.set("general.ext.flux_value", 0.3)

    monkeypatch.setattr(PointView, "edit", interleaved)
    with pytest.raises(ConflictError):
        session.control.set_parameter(target, "general.ext.flux_value", 0.2)
    assert session.control.read_context().document() == capture.document()
    assert session.state.version.get("context") == version
    assert snapshot(session.entry.use_point("a")).general.ext["flux_value"] == 0.3


@pytest.mark.parametrize("kind", KINDS)
def test_missing_saved_point_is_not_created_by_setter(
    session: _Session,
    kind: Kind,
) -> None:
    _select(session, "selected")
    target = SavedPoint(entry_id=session.entry.entry_id, point="missing")
    capture = session.control.read_context()
    with pytest.raises(FileNotFoundError):
        _write(session.control, target, kind, 0.2)
    assert session.entry.list_points() == ["selected"]
    assert session.control.read_context().document() == capture.document()
    assert (
        session.control.read_context().library.document() == capture.library.document()
    )


@pytest.mark.parametrize("kind", KINDS)
def test_wrong_path_or_library_partition_keeps_publication(
    session: _Session,
    kind: Kind,
) -> None:
    target = _select(session, "a")
    capture = session.control.read_context()
    path = {
        "parameter": "unknown.field",
        "module": "Q1.waveforms.pi",
        "waveform": "Q1.pulses.base",
    }[kind]
    with pytest.raises((ValueError, AttributeError)):
        _write(session.control, target, kind, 0.2, path=path)
    assert session.control.read_context().document() == capture.document()
    assert (
        session.control.read_context().library.document() == capture.library.document()
    )


def test_shared_writeback_entry_retains_order_metadata_and_failure_prefix(
    session: _Session,
) -> None:
    from zcu_tools.resources.entry import writeback

    target = _select(session, "a")
    events: list[object] = []
    session.bus.subscribe(MdChangedPayload, events.append)
    session.bus.subscribe(MlChangedPayload, events.append)
    before = session.state.version.get("context")
    with pytest.raises(writeback.WritebackApplyError) as caught:
        session.control.apply_writeback(
            target,
            (
                writeback.WaveformWriteback(
                    "Q1.waveforms.base", ConstWaveformCfg(length=0.1)
                ),
                writeback.ParameterWriteback(
                    "general.description", "retained", metadata={"stderr": 0.2}
                ),
                writeback.ParameterWriteback("general.description", 42),
                writeback.ParameterWriteback("general.ext.later", 3.0),
            ),
            evidence=writeback.WritebackEvidence(
                kind="test/ordered", origin="gui", run_id="run-1"
            ),
        )
    error = caught.value
    assert error.applied == 0
    assert isinstance(error.__cause__, ValueError)
    event = session.entry.ledger.get(error.event_id).event
    assert event.kind == "accepted"
    current = session.control.read_target(target)
    assert current.document().general.description is None
    assert not current.lookup("general.ext.later").present
    assert current.library.waveform_keys() == ()
    assert document_meta(current.document(), "general.description") is None
    assert events == []
    assert session.state.version.get("context") == before
    assert session.entry.use_point("a").description is None


def test_grouped_writeback_publishes_one_fact_per_committed_document(
    session: _Session,
) -> None:
    from zcu_tools.resources.entry import writeback

    target = _select(session, "a")
    events: list[object] = []
    session.bus.subscribe(MdChangedPayload, events.append)
    session.bus.subscribe(MlChangedPayload, events.append)
    before = session.state.version.get("context")
    result = session.service.apply_writeback(
        target,
        (
            writeback.WaveformWriteback(
                "Q1.waveforms.base", ConstWaveformCfg(length=0.1)
            ),
            writeback.ParameterWriteback("general.ext.fit", {"gain": 0.1}),
            writeback.ModuleWriteback(
                "Q1.pulses.pi",
                PulseSpec(
                    ch=0, nqz=1, freq=100.0, gain=0.5, waveform="Q1.waveforms.base"
                ),
            ),
            writeback.ParameterWriteback("general.ext.fit.gain", 0.2),
        ),
        evidence=writeback.WritebackEvidence(kind="test/grouped", origin="gui"),
    )
    assert [type(event) for event in events] == [MdChangedPayload, MlChangedPayload]
    assert session.state.version.get("context") == before + 2
    assert [item.value for item in result.committed[:2]] == [{"gain": 0.1}, 0.2]
    assert session.control.read_context().value("general.ext.fit.gain") == 0.2


@pytest.mark.parametrize("kind", ["module", "waveform"])
@pytest.mark.parametrize("stderr", [-0.1, float("inf")])
def test_library_writeback_rejects_invalid_uncertainty_before_acceptance(
    session: _Session, kind: str, stderr: float
) -> None:
    from zcu_tools.resources.entry import writeback

    target = _select(session, "a")
    capture = session.control.read_context()
    version = session.state.version.get("context")
    waveform = ConstWaveformCfg(length=0.1)
    item = (
        writeback.ModuleWriteback(
            "Q1.pulses.pi",
            PulseSpec(ch=0, nqz=1, freq=100.0, gain=0.2, waveform=waveform),
            stderr=stderr,
        )
        if kind == "module"
        else writeback.WaveformWriteback("Q1.waveforms.base", waveform, stderr=stderr)
    )
    with pytest.raises(ValueError, match="stderr"):
        session.control.apply_writeback(
            target,
            (item,),
            evidence=writeback.WritebackEvidence(kind="test/uncertainty", origin="gui"),
        )
    assert session.entry.ledger.events() == ()
    assert (
        session.control.read_context().library.document() == capture.library.document()
    )
    assert session.state.version.get("context") == version


def test_shared_writeback_entry_uses_detached_saved_target(
    session: _Session,
) -> None:
    from zcu_tools.resources.entry import writeback

    first = _select(session, "a")
    second = _select(session, "b")
    version = session.state.version.get("context")
    item = writeback.ParameterWriteback("general.description", "detached")
    evidence = writeback.WritebackEvidence(kind="test/fixed", origin="mcp")
    result = session.control.apply_writeback(first, (item,), evidence=evidence)
    event_id = result.event_id
    assert [(item.path, item.value) for item in result.committed] == [
        ("general.description", "detached")
    ]
    assert session.entry.use_point("a").description == "detached"
    assert session.control.read_context().target == second
    assert session.state.version.get("context") == version
    assert session.entry.ledger.get(event_id).event.origin == "mcp"


@pytest.mark.parametrize(
    ("label", "error_type"),
    [("missing", FileNotFoundError), ("../outside", ValueError)],
    ids=["missing", "invalid-label"],
)
def test_sample_only_writeback_rejects_invalid_saved_point_before_acceptance(
    session: _Session, label: str, error_type: type[Exception]
) -> None:
    from zcu_tools.resources.entry import writeback

    selected = _select(session, "a")
    capture = session.control.read_context()
    points = session.entry.list_points()
    point_before = snapshot(session.entry.use_point("a")).model_dump(mode="json")
    version = session.state.version.get("context")
    target = SavedPoint(entry_id=session.entry.entry_id, point=label)

    with pytest.raises(error_type):
        session.service.apply_writeback(
            target,
            (writeback.RecordWriteback("Q1", {"t1_us": 50}),),
            evidence=writeback.WritebackEvidence(
                kind="test/sample", origin="gui", run_id="run-1"
            ),
        )

    assert session.entry.ledger.events() == ()
    assert session.entry.records("Q1").read().empty
    assert session.entry.list_points() == points
    current = session.control.read_context()
    assert current.target == selected
    assert current.document() == capture.document()
    assert current.library.document() == capture.library.document()
    assert (
        snapshot(session.entry.use_point("a")).model_dump(mode="json") == point_before
    )
    assert session.state.version.get("context") == version


def test_saved_point_samples_persist_and_replace_latest_run_row(
    session: _Session,
) -> None:
    from zcu_tools.resources.entry import writeback

    target = _select(session, "a")
    result = session.control.apply_writeback(
        target,
        (
            writeback.RecordWriteback(
                "Q1", {"t1_us": 20.09, "dev_value": 0.125, "dev_unit": "native"}
            ),
            writeback.ParameterWriteback("general.description", "draft value"),
        ),
        evidence=writeback.WritebackEvidence(
            kind="test/sample", origin="gui", run_id="run-1"
        ),
    )
    event_id = result.event_id
    event = session.entry.ledger.get(event_id).event
    row = session.entry.records("Q1").read().iloc[0].to_dict()
    assert datetime.fromisoformat(row["at"]) == datetime.fromisoformat(event.at)
    expected = {
        "t1_us": 20.09,
        "dev_value": 0.125,
        "dev_unit": "native",
        "run_id": "run-1",
        "point": "a",
        "at": row["at"],
        "source": event_id,
    }
    assert session.entry.records("Q1").read().iloc[0].to_dict() == expected
    assert result.committed[1].value == expected
    assert session.control.read_record(target, "Q1", "run-1") == expected
    assert session.control.read_record(target, "Q1", "missing") is None
    second = session.control.apply_writeback(
        session.control.read_context().target,
        (writeback.RecordWriteback("Q1", {"t1_us": 31.7}),),
        evidence=writeback.WritebackEvidence(
            kind="test/sample", origin="gui", run_id="run-1"
        ),
    )
    second_id = second.event_id
    second_row = session.entry.records("Q1").read().iloc[0].to_dict()
    assert datetime.fromisoformat(second_row["at"]) == datetime.fromisoformat(
        session.entry.ledger.get(second_id).event.at
    )
    assert session.control.read_record(
        session.control.read_context().target, "Q1", "run-1"
    ) == {
        "t1_us": 31.7,
        "run_id": "run-1",
        "point": "a",
        "at": second_row["at"],
        "source": second_id,
    }
