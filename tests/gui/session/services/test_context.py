"""Entry/point publication and fixed reads through the public session owner."""

from collections.abc import Generator
from datetime import date
from pathlib import Path

import pytest
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError
from zcu_tools.cfg_model import ConstWaveformCfg, WaveformAssetInfo
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    ContextSourceBinding,
    EntryInfo,
    PointBindingError,
    PointCreateRequest,
    PointUseRequest,
    SavedEntry,
    SavedPoint,
)
from zcu_tools.gui.session.events import (
    AliasesChangedPayload,
    ContextSwitchedPayload,
)
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import ContextReadiness, SessionEnv
from zcu_tools.resources.entry import (
    ResultEntry,
    RoleResolutionError,
    component_registry,
    snapshot,
)
from zcu_tools.resources.entry.aliases import EntryAliases
from zcu_tools.resources.parameter_schema import ParameterSchemaPolicy
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from tests.resources.entry.fakes import (
    register_fakes,
    registry_state,
    restore_registry,
)


@pytest.mark.parametrize(
    "mode, clone", [("inherit", "source"), ("empty", "source"), ("clone", None)]
)
def test_new_rejects_incompatible_mode_and_source_before_reads(
    rich_entry: ResultEntry, context: ContextService, mode, clone
) -> None:
    context.install_entry(rich_entry)
    before = context.read_context().target
    with pytest.raises(InvalidInputError):
        context.new_point(
            PointCreateRequest("rejected", "", CommandOrigin("gui"), clone, mode=mode)
        )
    assert context.read_context().target == before
    assert context.list_points(refresh=True) == ()


def test_new_empty_does_not_inherit_global_values(
    rich_entry: ResultEntry, context: ContextService
) -> None:
    rich_entry.save_aliases(EntryAliases(aliases={"driver": "Q1"}))
    with rich_entry.module_library().edit() as draft:
        draft.set_waveform("Q1.waveforms.base", ConstWaveformCfg(length=0.1))
    context.install_entry(rich_entry)
    context.new_point(
        PointCreateRequest("blank", "fresh", CommandOrigin("gui"), mode="empty")
    )

    capture = context.read_context()
    assert capture.whole_document().components == {}
    assert capture.whole_document().general.description == "fresh"
    assert capture.library.waveform_keys() == ()
    assert context.get_alias_rows() == ()
    assert [file.revision for file in context.get_file_revisions()][1:] == [None, None]


def test_open_entry_captures_global_parameters_library_aliases_and_storage(
    rich_entry: ResultEntry, context: ContextService
) -> None:
    rich_entry.save_aliases(EntryAliases(aliases={"driver": "Q1"}))
    context.install_entry(rich_entry)

    capture = context.read_context()

    assert capture.value("$driver.rate") == 4300.0
    assert capture.target is not None and capture.target.point is None
    assert context.capture_storage(capture).point is None
    assert context.capture_storage(capture).result_path == rich_entry.result_path
    assert context.current_point_label() is None
    assert context.get_session_env().active_label is None
    assert context.get_session_env().readiness.value == "global"
    assert context.has_context() and context.is_active_context()
    assert [file.path for file in context.get_file_revisions()] == [
        rich_entry.result_path / name
        for name in ("parameters.yaml", "library.yaml", "aliases.yaml")
    ]


def test_point_and_global_switch_whole_sets_before_publishing_and_keep_captures(
    rich_entry: ResultEntry,
) -> None:
    rich_entry.save_aliases(EntryAliases(aliases={"driver": "Q1"}))
    with rich_entry.module_library().edit() as draft:
        draft.set_waveform("Q1.waveforms.base", ConstWaveformCfg(length=0.1))
    point = rich_entry.new_point("local", mode="empty")
    point.add_component("Q2", kind="fake/drive/a", rate=5200.0, sense="R2")
    state = SessionState(SessionEnv(parameters=None, soc=None, soccfg=None))
    bus = BaseEventBus()
    context = ContextService(state, bus, bind_assets=ArbWaveformDatabase.info_at)
    seen: list[tuple[str, str | None, tuple[Path, ...]]] = []

    def switched(_: ContextSwitchedPayload) -> None:
        env = context.get_session_env()
        seen.append(
            (
                env.readiness.value,
                env.active_label,
                tuple(file.path.parent for file in context.get_file_revisions()),
            )
        )
        assert context.read_context().target is not None

    subscription = bus.subscribe(ContextSwitchedPayload, switched)
    try:
        context.install_entry(rich_entry)
        global_capture = context.read_context()
        context.use_point(PointUseRequest("local"))
        local_capture = context.read_context()
        assert local_capture.value("Q2.rate") == 5200.0
        assert local_capture.library.waveform_keys() == ()
        with pytest.raises(RoleResolutionError):
            local_capture.value("Q1.rate")
        with pytest.raises(RoleResolutionError):
            local_capture.value("$driver")
        context.use_global()
        assert context.read_context().value("$driver.rate") == 4300.0
        assert context.read_context().library.waveform_keys() == ("Q1.waveforms.base",)
        assert global_capture.value("Q1.rate") == 4300.0
        assert local_capture.value("Q2.rate") == 5200.0
        assert context.capture_storage(local_capture).point == "local"
        assert state.version.get("context") == 3
        assert seen == [
            ("global", None, (rich_entry.result_path,) * 3),
            ("point", "local", (rich_entry.result_path / "points/local",) * 3),
            ("global", None, (rich_entry.result_path,) * 3),
        ]
    finally:
        subscription.unsubscribe()


def test_global_preparation_failure_keeps_selected_point_and_publishes_no_fact(
    rich_entry: ResultEntry,
) -> None:
    rich_entry.new_point("local")
    state = SessionState(SessionEnv(parameters=None, soc=None, soccfg=None))
    bus = BaseEventBus()
    context = ContextService(state, bus, bind_assets=ArbWaveformDatabase.info_at)
    context.install_entry(rich_entry)
    target = context.use_point(PointUseRequest("local"))
    before = context.get_session_env()
    binding = context.get_source_binding()
    revisions = context.get_file_revisions()
    version = state.version.get("context")
    facts: list[ContextSwitchedPayload] = []
    subscription = bus.subscribe(ContextSwitchedPayload, facts.append)
    try:
        path = rich_entry.result_path / "library.yaml"
        path.write_text("[invalid yaml", encoding="utf-8")
        with pytest.raises((ValueError, YAMLError)):
            context.use_global()
        assert context.get_session_env() is before
        assert context.read_context().target == target
        assert context.get_source_binding() == binding
        assert context.get_file_revisions() == revisions
        assert state.version.get("context") == version
        assert facts == []
    finally:
        subscription.unsubscribe()


def test_current_point_label_for_storage_free_content_has_no_persistent_point() -> None:
    state = SessionState(
        SessionEnv(parameters=ParameterContext.empty(), soc=None, soccfg=None)
    )
    owner = ContextService(
        state, BaseEventBus(), bind_assets=ArbWaveformDatabase.info_at
    )
    assert owner.current_point_label() is None


def test_current_point_label_follows_selection_without_refreshing_files(
    rich_entry: ResultEntry, context: ContextService
) -> None:
    assert context.current_point_label() is None
    rich_entry.new_point("selected")
    context.install_entry(rich_entry)
    assert context.current_point_label() is None

    context.use_point(PointUseRequest("selected"))
    rich_entry.result_path.rename(rich_entry.result_path.with_name("offline"))
    assert context.current_point_label() == "selected"

    context.clear_context()
    assert context.current_point_label() is None


def test_point_capture_reads_good_leaf_with_bad_sibling(
    rich_entry: ResultEntry, context: ContextService
) -> None:
    rich_entry.new_point("partial")
    path = rich_entry.result_path / "points" / "partial" / "parameters.yaml"
    raw = YAML(typ="safe").load(path)
    raw["components"]["Q1"]["sense"] = ["bad sibling"]
    with path.open("w", encoding="utf-8") as stream:
        YAML(typ="rt").dump(raw, stream)
    before = path.read_bytes()
    context.install_entry(rich_entry)

    context.use_point(PointUseRequest("partial"))
    parameters = context.read_context()

    assert parameters.value("Q1.rate") == 4300.0
    with pytest.raises(ValueError, match="Q1.sense"):
        parameters.value("Q1.sense")
    assert path.read_bytes() == before


def test_point_availability_accepts_parseable_partial_components_and_isolates_bad_yaml(
    rich_entry: ResultEntry, context: ContextService
) -> None:
    rich_entry.new_point("partial")
    rich_entry.new_point("unparseable")
    path = rich_entry.result_path / "points" / "partial" / "parameters.yaml"
    raw = YAML(typ="safe").load(path)
    raw["components"]["Q1"]["sense"] = ["bad sibling"]
    raw["components"]["unused"] = ["bad component"]
    with path.open("w", encoding="utf-8") as stream:
        YAML(typ="rt").dump(raw, stream)
    broken = rich_entry.result_path / "points" / "unparseable" / "parameters.yaml"
    broken.write_text("components: [", encoding="utf-8")
    context.install_entry(rich_entry)

    rows = {row.label: row for row in context.list_points()}

    assert rows["partial"].available
    assert rows["partial"].error is None
    assert not rows["unparseable"].available
    assert rows["unparseable"].error
    context.use_point(PointUseRequest("partial"))
    assert context.read_context().value("Q1.rate") == 4300.0
    with pytest.raises(ValueError, match="Q1.sense"):
        context.read_context().value("Q1.sense")


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
        # Version 2 lets the isolation regression exercise an older valid version.
        component_registry.configure_schema(
            ParameterSchemaPolicy(2, ("synthetic-upgrade",))
        )
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
    )


def _entry(tmp_path: Path, name: str = "entry") -> ResultEntry:
    return ResultEntry.create(
        name, result_root=tmp_path / "result", database_root=tmp_path / "Database"
    )


def _new_request(label: str, *, clone: str | None = None) -> PointCreateRequest:
    return PointCreateRequest(
        label,
        "session description",
        CommandOrigin("gui"),
        clone,
        mode="clone" if clone is not None else "inherit",
    )


def _entry_command(context: ContextService, command: str, entry_name: str) -> object:
    if command == "use":
        return context.use_point(PointUseRequest("new"))
    if command == "new":
        return context.new_point(PointCreateRequest("new", "", CommandOrigin("gui")))
    return context.entry_ledger(entry_name)


def test_global_alias_changes_install_before_the_fact_and_reject_invalid_batches(
    tmp_path: Path,
) -> None:
    state = SessionState(SessionEnv(parameters=None, soc=None, soccfg=None))
    bus = BaseEventBus()
    service = ContextService(state, bus, bind_assets=ArbWaveformDatabase.info_at)
    service.install_entry(_entry(tmp_path))
    version = state.version.get("context")
    observed: list[tuple[tuple[AliasRow, ...], int]] = []

    def on_aliases(event: AliasesChangedPayload) -> None:
        assert service.get_alias_rows() == event.rows
        observed.append((event.rows, state.version.snapshot().get("context", 0)))

    bus.subscribe(AliasesChangedPayload, on_aliases)
    rows = (AliasRow("driver", "Q1"), AliasRow("sense", "$driver.sense"))
    service.apply_alias_rows(rows)
    assert observed == [(rows, version + 1)]
    with pytest.raises(InvalidInputError):
        service.apply_alias_rows(
            (AliasRow("driver", "Q2"), AliasRow("sense", "$sense.sense"))
        )
    assert service.get_alias_rows() == rows
    assert state.version.snapshot().get("context") == version + 1
    service.apply_alias_rows(rows)
    assert observed == [(rows, version + 1)]
    assert state.version.snapshot().get("context") == version + 1


def test_current_alias_captures_follow_point_and_rows_but_prior_capture_stays_fixed(
    context: ContextService,
    rich_entry: ResultEntry,
) -> None:
    context.install_entry(rich_entry)
    target = context.new_point(_new_request("first"))
    assert context.get_alias_rows() == ()
    context.apply_alias_rows(
        (AliasRow("driver", "Q1"), AliasRow("sense", "$driver.sense"))
    )
    first = context.read_context()
    assert first is not None
    context.set_parameter(target, "Q1.rate", 4400.0)
    current = context.read_context()
    assert current is not None
    assert first.value("$driver.rate") == 4300.0
    assert current.value("$driver.rate") == 4400.0
    context.new_point(_new_request("second"))
    context.apply_alias_rows(
        (AliasRow("driver", "Q2"), AliasRow("sense", "$driver.sense"))
    )
    second = context.read_context()
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
    )
    context.install_entry(rich_entry)
    context.new_point(_new_request("first"))
    context.apply_alias_rows(
        (AliasRow("driver", ""), AliasRow("sense", "$driver.sense"))
    )
    before = context.get_alias_rows()
    version = state.version.snapshot()
    temporary = context.read_context(aliases={"driver": "Q2"})
    assert temporary is not None
    assert temporary.value("$driver.rate") == 5100.0
    assert temporary.roles is not None
    assert temporary.roles.names == {"driver": "Q2", "sense": "R2"}
    assert temporary.capture_storage().roles == temporary.roles.names
    assert context.get_alias_rows() == before
    assert state.version.snapshot() == version
    current = context.read_context()
    assert current is not None
    assert current.roles is not None
    assert current.roles.names == {}
    context.apply_alias_rows(
        (AliasRow("driver", "Q1"), AliasRow("sense", "$driver.sense"))
    )
    assert temporary.value("$driver.rate") == 5100.0


@pytest.mark.parametrize("target", ["", "missing"])
def test_bad_one_run_target_keeps_the_global_table(
    context: ContextService,
    rich_entry: ResultEntry,
    target: str,
) -> None:
    context.install_entry(rich_entry)
    context.new_point(_new_request("first"))
    before = context.get_alias_rows()
    with pytest.raises((InvalidInputError, RoleResolutionError)):
        context.read_context(aliases={"driver": target}).value("$driver.rate")
    assert context.get_alias_rows() == before


def test_historical_target_reads_keep_resolved_alias_names_when_point_links_change(
    context: ContextService,
    rich_entry: ResultEntry,
) -> None:
    context.install_entry(rich_entry)
    target = context.new_point(_new_request("first"))
    context.apply_alias_rows(
        (AliasRow("driver", "Q1"), AliasRow("sense", "$driver.sense"))
    )
    captured = context.read_context()
    assert captured is not None
    fixed_names = captured.roles
    assert fixed_names is not None
    assert fixed_names.names == {"driver": "Q1", "sense": "R1"}
    context.set_parameter(target, "Q1.sense", "R2")
    historical = context.read_target(target, roles=fixed_names)
    assert historical.resolve_path("$sense.rate") == "R1.rate"
    current = context.read_context()
    assert current is not None
    assert current.resolve_path("$sense.rate") == "R2.rate"
    assert captured.value("$driver.sense") == "R1"


def test_temporary_aliases_require_parameter_context(
    context: ContextService,
    rich_entry: ResultEntry,
) -> None:
    with pytest.raises(FailedPreconditionError) as caught:
        context.read_context(aliases={"driver": "Q1"})
    assert caught.value.reason_code == "no_context"


def test_parameter_write_preserves_global_aliases(
    context: ContextService,
    rich_entry: ResultEntry,
) -> None:
    context.install_entry(rich_entry)
    target = context.new_point(_new_request("a"))
    context.apply_alias_rows((AliasRow("driver", "Q1"),))
    rows = context.get_alias_rows()
    context.set_parameter(target, "Q1.rate", 4500.0)
    assert context.get_alias_rows() == rows
    assert context.read_context().value("Q1.rate") == 4500.0


def test_empty_queries_have_no_entry_point_or_component_selection(
    context: ContextService,
) -> None:
    assert not context.has_context()
    assert not context.is_active_context()
    assert context.get_entry_info() is None
    assert context.list_points() == ()
    assert context.get_alias_rows() == ()


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


def test_install_entry_preserves_nonparameter_capabilities_without_selecting_point(
    tmp_path: Path,
) -> None:
    context = ContextService(
        SessionState(SessionEnv(parameters=None, soc=None, soccfg=None)),
        BaseEventBus(),
        bind_assets=ArbWaveformDatabase.info_at,
    )
    entry = _entry(tmp_path)
    old = context.get_session_env()
    info = context.install_entry(entry)
    current = context.get_session_env()
    assert info.name == entry.result_path.name
    assert info.result_path == entry.result_path
    assert info.database_path == entry.database_path
    assert current.readiness is ContextReadiness.GLOBAL
    assert current.soc is old.soc and current.values is old.values
    assert current.parameters is not None
    assert context.has_context()
    assert context.list_points() == ()


@pytest.mark.parametrize(
    ("damage", "reason"),
    [
        ("schema", "schema_version"),
        ("kind", "unknown"),
        ("extra", "extra"),
        ("timestamp", "unexpected"),
    ],
)
def test_install_entry_isolates_bad_point_parameters_without_reading_other_files(
    context: ContextService, tmp_path: Path, damage: str, reason: str
) -> None:
    entry = _entry(tmp_path)
    entry.new_point("good", description="usable point")
    entry.new_point("bad", description="retained description")
    source = entry.result_path / "points/bad/parameters.yaml"
    yaml = YAML(typ="safe")
    document = yaml.load(source.read_text())
    if damage == "schema":
        document["schema_version"] = 1
    elif damage == "kind":
        document["components"] = {"Q": {"kind": "unknown/test"}}
    elif damage == "timestamp":
        document["unexpected"] = date(2020, 1, 1)
    else:
        document["unexpected"] = "opaque"
    with source.open("w") as stream:
        yaml.dump(document, stream)
    # Neither summary nor parameter availability consumes these documents.
    for label in ("good", "bad"):
        directory = entry.result_path / "points" / label
        if label == "good":
            (directory / "library.yaml").unlink()
        else:
            (directory / "library.yaml").write_text("[invalid YAML", encoding="utf-8")
        (directory / "aliases.yaml").write_text("[invalid YAML", encoding="utf-8")
    before = {
        path: path.read_bytes()
        for path in entry.result_path.rglob("*")
        if path.is_file()
    }

    context.install_entry(entry)

    points = {point.label: point for point in context.list_points()}
    assert points["good"].available
    assert points["good"].error is None
    assert points["bad"].description == "retained description"
    if damage in ("kind", "extra"):
        assert points["bad"].available
        assert points["bad"].error is None
        with pytest.raises(ValueError, match=reason):
            snapshot(entry.use_point("bad"))
    else:
        assert not points["bad"].available
        assert points["bad"].error is not None
        assert reason in points["bad"].error
        assert str(source) in points["bad"].error
    assert {
        path: path.read_bytes()
        for path in entry.result_path.rglob("*")
        if path.is_file()
    } == before
    # Restore only the files selection currently needs, outside the probe.
    (entry.result_path / "points/good/library.yaml").write_bytes(
        (entry.result_path / "library.yaml").read_bytes()
    )
    (entry.result_path / "points/good/aliases.yaml").unlink()
    context.use_point(PointUseRequest("good"))
    assert context.read_context().whole_document().general.description == "usable point"


def test_use_point_detaches_capture_and_uses_same_entry_library(
    context: ContextService, tmp_path: Path
) -> None:
    entry = _entry(tmp_path)
    first = entry.new_point("first", description="first point")
    entry.new_point("second", description="second point")
    context.install_entry(entry)
    target = context.use_point(PointUseRequest("first"))
    capture = context.read_context()
    assert target == SavedPoint(entry_name=entry.result_path.name, point="first")
    assert capture.whole_document().general.description == "first point"
    with first.edit() as edit:
        edit.general.description = "changed outside publication"
    context.use_point(PointUseRequest("second"))
    assert capture.whole_document().general.description == "first point"
    assert capture.target == target
    assert context.read_context().whole_document().general.description == "second point"
    assert context.get_session_env().readiness is ContextReadiness.POINT
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
    before = context.get_session_env()
    with pytest.raises(FailedPreconditionError) as error:
        _entry_command(context, "ledger", other.result_path.name)
    assert error.value.reason_code == "entry_not_open"
    assert context.get_session_env() == before
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
    assert context.read_context().whole_document().general.description == "retained"


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
    assert info is not None and info.name == first.result_path.name
    assert context.get_session_env().readiness is ContextReadiness.GLOBAL
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
    assert target == SavedPoint(entry_name=entry.result_path.name, point="destination")
    document = context.read_context().whole_document()
    assert document.general.description == "session description"
    assert document.general.ext.get("marker") == (None if clone is None else 7.0)
    assert {point.label for point in context.list_points()} == {"source", "destination"}


@pytest.mark.parametrize("clone", [None, "source"], ids=["setup", "clone"])
def test_duplicate_point_is_typed_input_error_without_side_effects(
    tmp_path: Path, clone: str | None
) -> None:
    state = SessionState(SessionEnv(parameters=None, soc=None, soccfg=None))
    bus = BaseEventBus()
    service = ContextService(state, bus, bind_assets=ArbWaveformDatabase.info_at)
    context = service
    entry = _entry(tmp_path)
    for label in ("copy", "source"):
        entry.new_point(label, description="retained")
    service.install_entry(entry)
    target = context.use_point(PointUseRequest("source"))
    before = context.read_context().document()
    version = state.version.snapshot()
    ledger = entry.ledger.events()
    files = {
        path.relative_to(tmp_path): path.read_bytes() if path.is_file() else None
        for path in tmp_path.rglob("*")
    }
    facts: list[ContextSwitchedPayload] = []
    subscription = bus.subscribe(ContextSwitchedPayload, facts.append)
    try:
        with pytest.raises((FileExistsError, InvalidInputError)) as caught:
            context.new_point(_new_request("copy", clone=clone))
        assert isinstance(caught.value, InvalidInputError)
        assert "exist" in str(caught.value)
        assert context.read_context().target == target
        assert context.read_context().document() == before
        assert [point.label for point in context.list_points()] == ["copy", "source"]
        assert state.version.snapshot() == version
        assert entry.ledger.events() == ledger
        assert facts == []
        assert {
            path.relative_to(tmp_path): path.read_bytes() if path.is_file() else None
            for path in tmp_path.rglob("*")
        } == files
    finally:
        subscription.unsubscribe()


@pytest.mark.parametrize("clone", [None, "source"])
@pytest.mark.parametrize(
    "label",
    [
        "../escape",
        "",
        ".",
        "..",
        "nested/point",
        r"nested\point",
        "\x00",
        "C:point",
        "/absolute",
    ],
    ids=[
        "escape",
        "empty",
        "dot",
        "parent",
        "posix",
        "windows",
        "nul",
        "drive",
        "absolute",
    ],
)
def test_unsafe_point_label_is_typed_input_error_without_side_effects(
    tmp_path: Path, label: str, clone: str | None
) -> None:
    state = SessionState(SessionEnv(parameters=None, soc=None, soccfg=None))
    bus = BaseEventBus()
    service = ContextService(state, bus, bind_assets=ArbWaveformDatabase.info_at)
    context = service
    entry = _entry(tmp_path)
    entry.new_point("source", description="retained")
    service.install_entry(entry)
    target = context.use_point(PointUseRequest("source"))
    before = context.read_context().document()
    version = state.version.snapshot()
    ledger = entry.ledger.events()
    files = {
        path.relative_to(tmp_path): path.read_bytes() if path.is_file() else None
        for path in tmp_path.rglob("*")
    }
    facts: list[ContextSwitchedPayload] = []
    subscription = bus.subscribe(ContextSwitchedPayload, facts.append)
    try:
        with pytest.raises((ValueError, InvalidInputError)) as caught:
            context.new_point(_new_request(label, clone=clone))
        assert isinstance(caught.value, InvalidInputError)
        assert caught.value.reason_code == "point_label_invalid"
        assert "single path component" in str(caught.value)
        assert context.read_context().target == target
        assert context.read_context().document() == before
        assert [point.label for point in context.list_points()] == ["source"]
        assert state.version.snapshot() == version
        assert entry.ledger.events() == ledger
        assert facts == []
        assert {
            path.relative_to(tmp_path): path.read_bytes() if path.is_file() else None
            for path in tmp_path.rglob("*")
        } == files
    finally:
        subscription.unsubscribe()


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
    assert (
        error.value.entry_name == entry.result_path.name
        and error.value.point == "retained"
    )
    assert isinstance(error.value.__cause__, OSError)
    assert entry.list_points() == ["retained"]
    assert context.get_session_env().readiness is ContextReadiness.GLOBAL


def test_entry_ledger_is_the_real_capability_and_capture_never_appends(
    context: ContextService, tmp_path: Path
) -> None:
    entry = _entry(tmp_path)
    before = entry.ledger.events()
    context.install_entry(entry)
    ledger = context.entry_ledger(entry.result_path.name)
    assert ledger.events() == before
    context.new_point(_new_request("working"))
    context.read_context()
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
    before = context.get_session_env()
    before_events = entry.ledger.events()
    other_events = other.ledger.events()

    with pytest.raises(ValueError, match="name"):
        context.new_point(_new_request("destination", clone="other/source"))

    assert entry.list_points() == []
    assert other.list_points() == ["source"]
    assert entry.ledger.events() == before_events
    assert other.ledger.events() == other_events
    assert context.get_session_env() == before


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
            if version == 1:
                assert env.parameters is not None
                assert env.parameters.target == SavedEntry(entry_name=info.name)
            else:
                assert env.parameters is not None
                assert env.parameters.target == SavedPoint(
                    entry_name=info.name, point="first" if version == 2 else "second"
                )
            assert sources is not None and sources.sources
            assert env.result_dir == str(info.result_path)
            assert env.database_path == str(info.database_path)
            assert env.active_label == (
                None if version == 1 else "first" if version == 2 else "second"
            )
            assert env.readiness is (
                ContextReadiness.GLOBAL if version == 1 else ContextReadiness.POINT
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
