"""Entry/point publication and fixed reads through the public session owner."""

from collections.abc import Generator
from pathlib import Path

import pytest
from zcu_tools.cfg_model import WaveformAssetInfo
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    ContextSourceBinding,
    EntryDraft,
    EntryInfo,
    PointBindingError,
    PointCreateRequest,
    PointUseRequest,
    SavedPoint,
)
from zcu_tools.gui.session.events import (
    AliasesChangedPayload,
    ContextSwitchedPayload,
)
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import ContextReadiness, SessionEnv
from zcu_tools.resources.entry import (
    ResultEntry,
    RoleResolutionError,
    RoleSpec,
)
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from tests.resources.entry.fakes import (
    register_fakes,
    registry_state,
    restore_registry,
)


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


@pytest.fixture(scope="module", autouse=True)
def registered_entry_models() -> Generator[None]:
    before = registry_state()
    try:
        register_fakes()
        installed = registry_state()
        yield
        assert registry_state() == installed, (
            "session context module polluted registries"
        )
    finally:
        restore_registry(before)


@pytest.fixture(autouse=True)
def registry_guard(request: pytest.FixtureRequest) -> Generator[None]:
    before = registry_state()
    yield
    assert registry_state() == before, f"registry polluter: {request.node.nodeid}"


@pytest.fixture
def rich_entry(tmp_path: Path) -> ResultEntry:
    entry = _entry(tmp_path)
    for name, rate, sense in (("Q1", 4300.0, "R1"), ("Q2", 5100.0, "R2")):
        entry.setup.add_component(name, kind="fake/drive/a", rate=rate, sense=sense)
    for name in ("R1", "R2"):
        entry.setup.add_component(name, kind="fake/sensor", rate=6500.0)
    return entry


@pytest.fixture
def context() -> ContextService:
    return ContextService(
        SessionState(SessionEnv(parameters=None, soc=None, soccfg=None)),
        BaseEventBus(),
        bind_assets=ArbWaveformDatabase.info_at,
        alias_rows=(AliasRow("driver", ""), AliasRow("sense", "$driver.sense")),
    )


def _entry(tmp_path: Path, name: str = "entry") -> ResultEntry:
    return ResultEntry.create(
        name, result_root=tmp_path / "result", database_root=tmp_path / "Database"
    )


def _new_request(label: str, *, clone: str | None = None) -> PointCreateRequest:
    return PointCreateRequest(label, "session description", CommandOrigin("gui"), clone)


def _entry_command(context: ContextService, command: str, entry_id: str) -> object:
    if command == "use":
        return context.use_point(PointUseRequest("new"))
    if command == "new":
        return context.new_point(PointCreateRequest("new", "", CommandOrigin("gui")))
    return context.entry_ledger(entry_id)


def test_alias_defaults_initialize_once_without_resetting_on_point_changes(
    context: ContextService, tmp_path: Path
) -> None:
    entry = _entry(tmp_path)
    entry.setup.add_component("Q1", kind="fake/drive/a", rate=4300.0, sense="R1")
    entry.setup.add_component("R1", kind="fake/sensor", rate=6500.0)
    context.install_entry(entry)
    assert context.get_alias_rows()[0].value == ""
    context.new_point(_new_request("first"))
    assert context.get_alias_rows() == (
        AliasRow("driver", "Q1"),
        AliasRow("sense", "$driver.sense"),
    )
    context.set_alias_rows((AliasRow("driver", ""), AliasRow("sense", "$driver.sense")))
    context.new_point(_new_request("second"))
    assert context.get_alias_rows()[0].value == ""
    capture = context.read_alias_context({"driver": RoleSpec("fake/drive/*")})
    assert capture is not None
    with pytest.raises(RoleResolutionError, match="driver"):
        capture.value("$driver.rate")


def test_alias_defaults_use_an_already_published_snapshot_without_binding_assets(
    context: ContextService,
    tmp_path: Path,
) -> None:
    entry = _entry(tmp_path)
    entry.setup.add_component("Q1", kind="fake/drive/a", rate=4300.0, sense="R1")
    entry.setup.add_component("R1", kind="fake/sensor", rate=6500.0)
    context.install_entry(entry)
    context.new_point(_new_request("first"))

    def no_asset_binding(_path: Path) -> WaveformAssetInfo:
        raise AssertionError("Cached initialization must not reopen assets")

    installed = ContextService(
        SessionState(context.get_session_env()),
        BaseEventBus(),
        bind_assets=no_asset_binding,
        alias_rows=(AliasRow("driver", ""), AliasRow("sense", "$driver.sense")),
    )
    assert installed.get_alias_rows()[0] == AliasRow("driver", "Q1")
    captured = installed.read_alias_context({"driver": RoleSpec("fake/drive/*")})
    assert captured is not None
    assert captured.value("$driver.rate") == 4300.0


def test_empty_alias_submission_before_a_point_prevents_default_rebinding(
    context: ContextService, tmp_path: Path
) -> None:
    assert context.read_alias_context({"driver": RoleSpec("fake/drive/*")}) is None
    context.set_alias_rows(context.get_alias_rows())
    entry = _entry(tmp_path)
    entry.setup.add_component("Q1", kind="fake/drive/a", rate=4300.0)
    context.install_entry(entry)
    context.new_point(_new_request("first"))
    assert context.get_alias_rows()[0].value == ""


def test_first_component_publication_initializes_the_alias_default(
    context: ContextService, tmp_path: Path
) -> None:
    context.install_entry(_entry(tmp_path))
    context.add_component("Q1", "fake/drive/a", {"rate": 4300.0})
    assert context.get_alias_rows()[0] == AliasRow("driver", "Q1")
    context.add_component("Q2", "fake/drive/a", {"rate": 5100.0})
    assert context.get_alias_rows()[0] == AliasRow("driver", "Q1")


def test_global_alias_changes_install_before_the_fact_and_reject_invalid_batches() -> (
    None
):
    state = SessionState(SessionEnv(parameters=None, soc=None, soccfg=None))
    bus = BaseEventBus()
    service = ContextService(state, bus, bind_assets=ArbWaveformDatabase.info_at)
    observed: list[tuple[tuple[AliasRow, ...], int]] = []

    def on_aliases(event: AliasesChangedPayload) -> None:
        assert service.get_alias_rows() == event.rows
        observed.append((event.rows, state.version.snapshot().get("context", 0)))

    bus.subscribe(AliasesChangedPayload, on_aliases)
    rows = (AliasRow("driver", "Q1"), AliasRow("sense", "$driver.sense"))
    service.set_alias_rows(rows)
    assert observed == [(rows, 1)]
    with pytest.raises(InvalidInputError):
        service.set_alias_rows(
            (AliasRow("driver", "Q2"), AliasRow("sense", "$sense.sense"))
        )
    assert service.get_alias_rows() == rows
    assert state.version.snapshot().get("context") == 1
    service.set_alias_rows(rows)
    assert observed == [(rows, 1)]
    assert state.version.snapshot().get("context") == 1


def test_current_alias_captures_follow_point_and_rows_but_prior_capture_stays_fixed(
    context: ContextService,
    rich_entry: ResultEntry,
) -> None:
    context.install_entry(rich_entry)
    target = context.new_point(_new_request("first"))
    assert context.get_alias_rows()[0].value == ""
    declarations = {
        "driver": RoleSpec("fake/drive/*"),
        "sense": RoleSpec("fake/sensor"),
    }
    context.set_alias_rows(
        (AliasRow("driver", "Q1"), AliasRow("sense", "$driver.sense"))
    )
    first = context.read_alias_context(declarations)
    assert first is not None
    context.set_parameter(target, "Q1.rate", 4400.0)
    current = context.read_alias_context(declarations)
    assert current is not None
    assert first.value("$driver.rate") == 4300.0
    assert current.value("$driver.rate") == 4400.0
    context.new_point(_new_request("second"))
    context.set_alias_rows(
        (AliasRow("driver", "Q2"), AliasRow("sense", "$driver.sense"))
    )
    second = context.read_alias_context(declarations)
    assert second is not None
    assert second.value("$driver.rate") == 5100.0
    assert first.target == target
    assert first.roles is not None
    assert first.roles.names == {"driver": "Q1", "sense": "R1"}
    assert second.roles is not None
    assert second.roles.names == {"driver": "Q2", "sense": "R2"}
    assert second.capture_storage().point == "second"
    assert first.capture_storage().point == "first"


def test_one_run_target_resolves_dependencies_without_changing_global_rows_or_version(
    rich_entry: ResultEntry,
) -> None:
    state = SessionState(SessionEnv(parameters=None, soc=None, soccfg=None))
    context = ContextService(
        state,
        BaseEventBus(),
        bind_assets=ArbWaveformDatabase.info_at,
        alias_rows=(AliasRow("driver", ""), AliasRow("sense", "$driver.sense")),
    )
    context.install_entry(rich_entry)
    context.new_point(_new_request("first"))
    before = context.get_alias_rows()
    version = state.version.snapshot()
    declarations = {
        "driver": RoleSpec("fake/drive/*"),
        "sense": RoleSpec("fake/sensor"),
    }
    temporary = context.read_alias_context(declarations, "Q2")
    assert temporary is not None
    assert temporary.value("$driver.rate") == 5100.0
    assert temporary.roles is not None
    assert temporary.roles.names == {"driver": "Q2", "sense": "R2"}
    assert temporary.capture_storage().roles == temporary.roles.names
    assert context.get_alias_rows() == before
    assert state.version.snapshot() == version
    current = context.read_alias_context(declarations)
    assert current is not None
    assert current.roles is not None
    assert current.roles.names == {}
    context.set_alias_rows(
        (AliasRow("driver", "Q1"), AliasRow("sense", "$driver.sense"))
    )
    assert temporary.value("$driver.rate") == 5100.0


@pytest.mark.parametrize("target", ["", "missing", "R1"])
def test_bad_one_run_target_keeps_the_global_table(
    context: ContextService,
    rich_entry: ResultEntry,
    target: str,
) -> None:
    context.install_entry(rich_entry)
    context.new_point(_new_request("first"))
    before = context.get_alias_rows()
    with pytest.raises((InvalidInputError, RoleResolutionError)):
        context.read_alias_context({"driver": RoleSpec("fake/drive/*")}, target)
    assert context.get_alias_rows() == before


def test_historical_target_reads_keep_resolved_alias_names_when_point_links_change(
    context: ContextService,
    rich_entry: ResultEntry,
) -> None:
    context.install_entry(rich_entry)
    target = context.new_point(_new_request("first"))
    context.set_alias_rows(
        (AliasRow("driver", "Q1"), AliasRow("sense", "$driver.sense"))
    )
    declarations = {
        "driver": RoleSpec("fake/drive/*"),
        "sense": RoleSpec("fake/sensor"),
    }
    captured = context.read_alias_context(declarations)
    assert captured is not None
    fixed_names = captured.roles
    assert fixed_names is not None
    assert fixed_names.names == {"driver": "Q1", "sense": "R1"}
    context.set_parameter(target, "Q1.sense", "R2")
    historical = context.read_target(target, roles=fixed_names)
    assert historical.resolve_path("$sense.rate") == "R1.rate"
    current = context.read_alias_context(declarations)
    assert current is not None
    assert current.resolve_path("$sense.rate") == "R2.rate"
    assert captured.value("$driver.sense") == "R1"


def test_target_requires_a_point_and_a_declared_alias(
    context: ContextService,
    rich_entry: ResultEntry,
) -> None:
    with pytest.raises(FailedPreconditionError) as caught:
        context.read_alias_context({"driver": RoleSpec("fake/drive/*")}, "Q1")
    assert caught.value.reason_code == "no_context"
    context.install_entry(rich_entry)
    context.new_point(_new_request("first"))
    with pytest.raises(InvalidInputError, match="declared alias"):
        context.read_alias_context({}, "Q1")


def test_parameter_write_preserves_global_aliases(
    context: ContextService,
    rich_entry: ResultEntry,
) -> None:
    context.install_entry(rich_entry)
    target = context.new_point(_new_request("a"))
    context.set_alias_rows((AliasRow("driver", "Q1"),))
    rows = context.get_alias_rows()
    context.set_parameter(target, "Q1.rate", 4500.0)
    assert context.get_alias_rows() == rows
    assert context.read_context().value("Q1.rate") == 4500.0


def test_empty_queries_have_no_entry_point_or_component_selection(
    context: ContextService,
) -> None:
    assert not context.has_context()
    assert not context.has_draft_context()
    assert not context.is_active_context()
    assert context.get_entry_info() is None
    assert context.list_points() == ()
    assert context.get_alias_rows() == (
        AliasRow("driver", ""),
        AliasRow("sense", "$driver.sense"),
    )


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
    tmp_path: Path,
) -> None:
    from zcu_tools.simulate.fluxonium.predict import FluxoniumPredictor

    predictor = FluxoniumPredictor(
        (5.0, 1.0, 0.5), flux_half=0.0, flux_period=1.0, flux_bias=0.0
    )
    context = ContextService(
        SessionState(
            SessionEnv(parameters=None, soc=None, soccfg=None, predictor=predictor)
        ),
        BaseEventBus(),
        bind_assets=ArbWaveformDatabase.info_at,
        alias_rows=(AliasRow("driver", ""), AliasRow("sense", "$driver.sense")),
    )
    entry = _entry(tmp_path)
    old = context.get_session_env()
    info = context.install_entry(entry)
    current = context.get_session_env()
    assert info.entry_id == entry.entry_id
    assert info.result_path == entry.result_path
    assert info.database_path == entry.database_path
    assert current.readiness is ContextReadiness.DRAFT
    assert current.soc is old.soc and current.values is old.values
    assert current.predictor is predictor
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
    target = context.use_point(PointUseRequest("first"))
    capture = context.read_context()
    assert target == SavedPoint(entry_id=entry.entry_id, point="first")
    assert capture.document().general.description == "first point"
    with first.edit() as edit:
        edit.general.description = "changed outside publication"
    context.use_point(PointUseRequest("second"))
    assert capture.document().general.description == "first point"
    assert capture.target == target
    assert context.read_context().document().general.description == "second point"
    assert context.get_session_env().readiness is ContextReadiness.ACTIVE
    assert {point.label: point.description for point in context.list_points()} == {
        "first": "first point",
        "second": "second point",
    }


def test_cross_entry_ledger_rejects_before_changing_binding(
    context: ContextService, tmp_path: Path
) -> None:
    entry = _entry(tmp_path)
    other = _entry(tmp_path, "other")
    context.install_entry(entry)
    before = context.read_context().target
    with pytest.raises(FailedPreconditionError) as error:
        _entry_command(context, "ledger", other.entry_id)
    assert error.value.reason_code == "entry_not_open"
    assert context.read_context().target == before
    assert entry.list_points() == [] and other.list_points() == []


def test_failed_point_read_keeps_previous_capture(
    context: ContextService, tmp_path: Path
) -> None:
    entry = _entry(tmp_path)
    entry.new_point("good", description="retained")
    context.install_entry(entry)
    before = context.use_point(PointUseRequest("good"))
    with pytest.raises(FileNotFoundError):
        context.use_point(PointUseRequest("missing"))
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
    target = context.new_point(_new_request("destination", clone=clone))
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
        context.new_point(_new_request("retained"))
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
    context.read_alias_context({})
    assert entry.ledger.events() == before


def test_clear_context_returns_empty_without_deleting_entry(
    context: ContextService, tmp_path: Path
) -> None:
    entry = _entry(tmp_path)
    context.install_entry(entry)
    context.new_point(_new_request("working"))
    context.clear_context()
    assert context.get_session_env().parameters is None
    assert context.get_session_env().readiness is ContextReadiness.EMPTY
    assert context.get_entry_info() is None
    assert context.list_points() == ()
    assert entry.list_points() == ["working"]


def test_cross_entry_clone_rejects_before_destination_or_ledger_changes(
    context: ContextService, tmp_path: Path
) -> None:
    entry = _entry(tmp_path)
    other = _entry(tmp_path, "other")
    other.new_point("source")
    context.install_entry(entry)
    before = context.read_context().target
    before_events = entry.ledger.events()
    other_events = other.ledger.events()

    with pytest.raises(ValueError, match="name"):
        context.new_point(_new_request("destination", clone="other/source"))

    assert entry.list_points() == []
    assert other.list_points() == ["source"]
    assert entry.ledger.events() == before_events
    assert other.ledger.events() == other_events
    assert context.read_context().target == before


def test_publication_facts_observe_installed_state_and_one_context_bump(
    rich_entry: ResultEntry,
) -> None:
    rich_entry.new_point("first")
    state = SessionState(SessionEnv(parameters=None, soc=None, soccfg=None))
    bus = BaseEventBus()
    assets = Assets()
    context = ContextService(state, bus, bind_assets=assets.bind)
    observations: list[
        tuple[int, SessionEnv, EntryInfo | None, ContextSourceBinding | None]
    ] = []

    def switched(payload: ContextSwitchedPayload) -> None:
        observations.append(
            (
                state.version.get("context"),
                state.session_env,
                context.get_entry_info(),
                payload.sources,
            )
        )

    subscription = bus.subscribe(ContextSwitchedPayload, switched)
    try:
        assert state.version.get("context") == 0 and observations == []
        info = context.install_entry(rich_entry)
        context.use_point(PointUseRequest("first"))
        assert state.version.get("context") == 2
        context.new_point(_new_request("second"))
        assert state.version.get("context") == 3
        context.clear_context()
        assert [version for version, *_ in observations] == [1, 2, 3, 4]
        for version, env, entry_info, sources in observations:
            if version == 4:
                assert env.readiness is ContextReadiness.EMPTY
                assert env.parameters is None and entry_info is None and sources is None
                continue
            assert entry_info == info
            assert env.parameters is not None
            expected = (
                EntryDraft(entry_id=info.entry_id)
                if version == 1
                else SavedPoint(
                    entry_id=info.entry_id, point="first" if version == 2 else "second"
                )
            )
            assert env.parameters.target == expected
            assert sources is not None and sources.sources
            assert env.result_dir == str(info.result_path)
            assert env.database_path == str(info.database_path)
            assert env.active_label == (
                "" if version == 1 else "first" if version == 2 else "second"
            )
            assert env.readiness is (
                ContextReadiness.DRAFT if version == 1 else ContextReadiness.ACTIVE
            )
    finally:
        subscription.unsubscribe()


def test_failed_preparation_keeps_binding_version_and_publishes_no_facts(
    rich_entry: ResultEntry, tmp_path: Path
) -> None:
    rich_entry.new_point("first")
    state = SessionState(SessionEnv(parameters=None, soc=None, soccfg=None))
    bus = BaseEventBus()
    assets = Assets()
    context = ContextService(state, bus, bind_assets=assets.bind)
    events: list[ContextSwitchedPayload] = []
    switched = bus.subscribe(ContextSwitchedPayload, events.append)
    try:
        context.install_entry(rich_entry)
        target = context.use_point(PointUseRequest("first"))
        version = state.version.get("context")
        events.clear()
        with pytest.raises(FileNotFoundError):
            context.use_point(PointUseRequest("missing"))
        assets.fail = True
        with pytest.raises(OSError, match="asset binding"):
            context.install_entry(_entry(tmp_path, "other"))
        with pytest.raises(PointBindingError):
            context.new_point(_new_request("retained"))
        assert events == []
        assert state.version.get("context") == version
        assert context.read_context().target == target
        assert rich_entry.list_points() == ["first", "retained"]
    finally:
        switched.unsubscribe()
