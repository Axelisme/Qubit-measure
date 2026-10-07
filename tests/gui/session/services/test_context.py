"""Entry/point publication and fixed reads through the public session owner."""

from pathlib import Path

import pytest
from zcu_tools.cfg_model import WaveformAssetInfo
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    ComponentFocus,
    EntryDraft,
    PointBindingError,
    PointCreateRequest,
    PointUseRequest,
    SavedPoint,
)
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import ContextReadiness, SessionEnv
from zcu_tools.resources.entry import ResultEntry
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase


class Assets:
    """Explicit resource capability provider with an observable binding failure."""

    def __init__(self) -> None:
        self.paths: list[Path] = []
        self.fail = False

    def bind(self, path: Path) -> WaveformAssetInfo:
        self.paths.append(path)
        if self.fail:
            raise OSError("synthetic asset binding failure")
        return ArbWaveformDatabase.info_at(path)


@pytest.fixture
def context() -> ContextService:
    return ContextService(
        SessionState(SessionEnv(parameters=None, soc=None, soccfg=None)),
        BaseEventBus(),
        bind_assets=ArbWaveformDatabase.info_at,
    )


def _entry(tmp_path: Path, name: str = "entry") -> ResultEntry:
    return ResultEntry.create(
        name, result_root=tmp_path / "result", database_root=tmp_path / "Database"
    )


def _new_request(
    entry: ResultEntry, label: str, *, clone: str | None = None
) -> PointCreateRequest:
    return PointCreateRequest(
        entry.entry_id, label, "session description", CommandOrigin("gui", None), clone
    )


def _entry_command(context: ContextService, command: str, entry_id: str) -> object:
    if command == "use":
        return context.use_point(PointUseRequest(entry_id, "new"))
    if command == "new":
        return context.new_point(
            PointCreateRequest(entry_id, "new", "", CommandOrigin("gui", None))
        )
    return context.entry_ledger(entry_id)


def test_empty_queries_have_no_entry_point_or_component_selection(
    context: ContextService,
) -> None:
    assert not context.has_context()
    assert not context.has_draft_context()
    assert not context.is_active_context()
    assert context.get_entry_info() is None
    assert context.list_points() == ()
    assert context.get_component_focus() == ComponentFocus(None, None, ())


def test_empty_read_rejects_instead_of_returning_default_parameters(
    context: ContextService,
) -> None:
    with pytest.raises(FailedPreconditionError) as error:
        context.read_context()
    assert error.value.reason_code == "no_context"


@pytest.mark.parametrize("command", ["use", "new", "ledger"])
def test_entry_commands_without_entry_fail_explicitly(
    context: ContextService, command: str
) -> None:
    with pytest.raises(FailedPreconditionError) as error:
        _entry_command(context, command, "not-open")
    assert error.value.reason_code == "no_entry"


def test_install_entry_publishes_empty_draft_and_preserves_nonparameter_capabilities(
    context: ContextService, tmp_path: Path
) -> None:
    entry = _entry(tmp_path)
    old = context.get_session_env()
    info = context.install_entry(entry)
    current = context.get_session_env()
    assert info.entry_id == entry.entry_id
    assert info.result_path == entry.result_path
    assert info.database_path == entry.database_path
    assert current.readiness is ContextReadiness.DRAFT
    assert current.soc is old.soc and current.values is old.values
    assert current.parameters is not None
    assert current.parameters.roles is None
    assert context.read_context().target == EntryDraft(entry_id=entry.entry_id)
    assert context.read_context().document().components == {}
    assert context.list_points() == ()


def test_use_point_detaches_capture_and_uses_same_entry_library(
    context: ContextService, tmp_path: Path
) -> None:
    entry = _entry(tmp_path)
    first = entry.new_point("first", description="first point")
    entry.new_point("second", description="second point")
    context.install_entry(entry)
    target = context.use_point(PointUseRequest(entry.entry_id, "first"))
    capture = context.read_context()
    assert target == SavedPoint(entry_id=entry.entry_id, point="first")
    assert capture.document().general.description == "first point"
    with first.edit() as edit:
        edit.general.description = "changed outside publication"
    context.use_point(PointUseRequest(entry.entry_id, "second"))
    assert capture.document().general.description == "first point"
    assert capture.target == target
    assert context.read_context().document().general.description == "second point"
    assert context.get_session_env().readiness is ContextReadiness.ACTIVE
    assert {point.label: point.description for point in context.list_points()} == {
        "first": "first point",
        "second": "second point",
    }


@pytest.mark.parametrize("command", ["use", "new", "ledger"])
def test_cross_entry_commands_reject_before_changing_binding(
    context: ContextService, tmp_path: Path, command: str
) -> None:
    entry = _entry(tmp_path)
    other = _entry(tmp_path, "other")
    context.install_entry(entry)
    before = context.read_context().target
    with pytest.raises(FailedPreconditionError) as error:
        _entry_command(context, command, other.entry_id)
    assert error.value.reason_code == "entry_not_open"
    assert context.read_context().target == before
    assert entry.list_points() == [] and other.list_points() == []


def test_failed_point_read_keeps_previous_capture(
    context: ContextService, tmp_path: Path
) -> None:
    entry = _entry(tmp_path)
    entry.new_point("good", description="retained")
    context.install_entry(entry)
    before = context.use_point(PointUseRequest(entry.entry_id, "good"))
    with pytest.raises(FileNotFoundError):
        context.use_point(PointUseRequest(entry.entry_id, "missing"))
    assert context.read_context().target == before
    assert context.read_context().document().general.description == "retained"


def test_failed_entry_asset_preparation_keeps_previous_binding(tmp_path: Path) -> None:
    assets = Assets()
    context = ContextService(
        SessionState(SessionEnv(parameters=None, soc=None, soccfg=None)),
        BaseEventBus(),
        bind_assets=assets.bind,
    )
    first = _entry(tmp_path)
    other = _entry(tmp_path, "other")
    context.install_entry(first)
    assets.fail = True
    with pytest.raises(OSError, match="asset binding"):
        context.install_entry(other)
    info = context.get_entry_info()
    assert info is not None and info.entry_id == first.entry_id
    assert context.read_context().target == EntryDraft(entry_id=first.entry_id)
    assert assets.paths[-1] == other.database_path / "arb_waveforms"


@pytest.mark.parametrize("clone", [None, "source"])
def test_new_point_uses_seed_or_clone_and_request_description(
    context: ContextService, tmp_path: Path, clone: str | None
) -> None:
    entry = _entry(tmp_path)
    source = entry.new_point("source", description="source")
    with source.edit() as edit:
        edit.set("general.ext.marker", 7.0)
    context.install_entry(entry)
    target = context.new_point(_new_request(entry, "destination", clone=clone))
    assert target == SavedPoint(entry_id=entry.entry_id, point="destination")
    document = context.read_context().document()
    assert document.general.description == "session description"
    assert document.general.ext.get("marker") == (None if clone is None else 7.0)
    assert {point.label for point in context.list_points()} == {"source", "destination"}


def test_created_point_binding_failure_retains_point_and_previous_binding(
    tmp_path: Path,
) -> None:
    assets = Assets()
    context = ContextService(
        SessionState(SessionEnv(parameters=None, soc=None, soccfg=None)),
        BaseEventBus(),
        bind_assets=assets.bind,
    )
    entry = _entry(tmp_path)
    context.install_entry(entry)
    assets.fail = True
    with pytest.raises(PointBindingError) as error:
        context.new_point(_new_request(entry, "retained"))
    assert error.value.entry_id == entry.entry_id and error.value.label == "retained"
    assert isinstance(error.value.__cause__, OSError)
    assert entry.list_points() == ["retained"]
    assert context.read_context().target == EntryDraft(entry_id=entry.entry_id)


def test_entry_ledger_is_the_real_capability_and_capture_never_appends(
    context: ContextService, tmp_path: Path
) -> None:
    entry = _entry(tmp_path)
    before = entry.ledger.events()
    context.install_entry(entry)
    ledger = context.entry_ledger(entry.entry_id)
    assert ledger.events() == before
    context.capture_roles({}, focus=None, explicit={})
    assert entry.ledger.events() == before


def test_clear_context_returns_empty_without_deleting_entry(
    context: ContextService, tmp_path: Path
) -> None:
    entry = _entry(tmp_path)
    context.install_entry(entry)
    context.new_point(_new_request(entry, "working"))
    context.clear_context()
    assert context.get_session_env().parameters is None
    assert context.get_session_env().readiness is ContextReadiness.EMPTY
    assert context.get_entry_info() is None
    assert context.list_points() == ()
    assert entry.list_points() == ["working"]
