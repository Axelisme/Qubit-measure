"""Point-local authored aliases through public storage and session seams."""

from collections.abc import Generator
from hashlib import sha256
from pathlib import Path

import pytest
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import PointUseRequest
from zcu_tools.gui.session.events import AliasesChangedPayload
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.resources.entry import ResultEntry, RoleResolutionError
from zcu_tools.resources.entry.aliases import EntryAliases
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from tests.resources.entry.fakes import register_fakes, registry_state, restore_registry

PRESET = (AliasRow("driver", ""), AliasRow("sense", "$driver.sense"))
AUTHORED = (
    AliasRow("driver", "Q1"),
    AliasRow("sense", "$driver.sense"),
    AliasRow("spare", ""),
)


@pytest.fixture(autouse=True)
def registered_models() -> Generator[None]:
    before = registry_state()
    try:
        register_fakes()
        installed = registry_state()
        yield
        assert registry_state() == installed
    finally:
        restore_registry(before)


def service(bus: BaseEventBus | None = None) -> ContextService:
    return ContextService(
        SessionState(SessionEnv(parameters=None, soc=None, soccfg=None)),
        bus if bus is not None else BaseEventBus(),
        bind_assets=ArbWaveformDatabase.info_at,
        alias_rows=PRESET,
    )


def entry_at(root: Path, name: str = "entry") -> ResultEntry:
    entry = ResultEntry.create(
        name, result_root=root / "result", database_root=root / "Database"
    )
    entry.setup.add_component("Q1", kind="fake/drive/a", rate=4300.0, sense="R1")
    entry.setup.add_component("R1", kind="fake/sensor", rate=6500.0)
    entry.new_point("first")
    return entry


def write_table(entry: ResultEntry, body: str, *, version: str = "1.0") -> None:
    (entry.result_path / "points" / "first" / "aliases.yaml").write_text(
        f'format: zcu.aliases\nformat_version: "{version}"\naliases: {body}\n',
        encoding="utf-8",
    )


def test_list_apply_reopen_capture_storage_and_failed_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = entry_at(tmp_path)
    context = service()
    context.install_entry(entry)
    context.use_point(PointUseRequest("first"))
    rows = (AliasRow("driver", "Q1"), AliasRow("participants", ["$driver.sense", "Q1"]))
    try:
        context.apply_alias_rows(rows)
    except (AttributeError, ValueError) as error:
        pytest.fail(f"Apply must save an ordered alias list: {error}")
    captured = context.read_context()
    assert captured.value("$participants") == ["R1", "Q1"]
    assert captured.lookup_point("$participants").value == ["R1", "Q1"]
    assert captured.capture_storage().roles["participants"] == ["R1", "Q1"]
    with pytest.raises(RoleResolutionError, match="list"):
        captured.resolve_path("$participants.rate")
    assert captured.resolved_aliases()[1].resolved == ["R1", "Q1"]
    context.use_point(PointUseRequest("first"))
    assert context.get_alias_rows() == rows
    assert entry.load_aliases(point="first").aliases["participants"] == [
        "$driver.sense",
        "Q1",
    ]
    events: list[AliasesChangedPayload] = []
    context_bus = BaseEventBus()
    fresh = service(context_bus)
    fresh.install_entry(entry)
    fresh.use_point(PointUseRequest("first"))
    context_bus.subscribe(AliasesChangedPayload, events.append)
    old_file = (entry.result_path / "points/first/aliases.yaml").read_bytes()

    def fail_replace(source: Path, destination: Path) -> Path:
        raise OSError("synthetic list save failure")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", fail_replace)
        with pytest.raises(OSError, match="synthetic list save failure"):
            fresh.apply_alias_rows((AliasRow("participants", ["Q1"]),))
    assert fresh.get_alias_rows() == rows
    assert (entry.result_path / "points/first/aliases.yaml").read_bytes() == old_file
    assert events == []
    context.set_alias_rows((AliasRow("participants", []),))
    with pytest.raises(RoleResolutionError, match="not bound"):
        context.read_context().value("$participants")
    assert captured.value("$participants") == ["R1", "Q1"]
    context.use_point(PointUseRequest("first"))
    assert context.get_alias_rows() == (AliasRow("participants", []),)
    assert entry.load_aliases(point="first").aliases == {"participants": []}


def test_file_revisions_are_installed_snapshots_and_alias_writes_publish_after_disk(
    tmp_path: Path,
) -> None:
    context = service()
    assert hasattr(context, "get_file_revisions"), "Context needs point-file revisions"
    assert context.get_file_revisions() == ()
    entry = entry_at(tmp_path)
    context.install_entry(entry)
    assert context.get_file_revisions() == ()
    context.use_point(PointUseRequest("first"))
    files = context.get_file_revisions()
    assert [item.document for item in files] == ["parameters", "library", "aliases"]
    assert all(item.path.is_absolute() for item in files)
    assert files[2].revision is None
    for item in files[:2]:
        assert item.revision == sha256(item.path.read_bytes()).hexdigest()
    context.set_alias_rows(AUTHORED)
    installed = context.get_file_revisions()
    assert installed[:2] == files[:2]
    assert installed[2].revision == sha256(installed[2].path.read_bytes()).hexdigest()
    assert entry.load_aliases(point="first").aliases["driver"] == "Q1"
    write_table(entry, "{}")
    assert context.get_file_revisions() == installed
    context.use_point(PointUseRequest("first"))
    assert context.get_file_revisions()[2].revision != installed[2].revision
    context.clear_context()
    assert context.get_file_revisions() == ()


def test_saved_table_loads_in_order_without_filling_empty_rows(tmp_path: Path) -> None:
    entry = entry_at(tmp_path)
    write_table(
        entry, '\n  spare: ""\n  driver: Q1\n  sense: $driver.sense', version="1.8"
    )
    context = service()
    context.install_entry(entry)
    context.use_point(PointUseRequest("first"))
    assert context.get_alias_rows() == (AliasRow("spare", ""), *AUTHORED[:2])
    write_table(entry, "{}")
    context.install_entry(entry)
    context.use_point(PointUseRequest("first"))
    assert context.get_alias_rows() == ()
    write_table(entry, '\n  driver: ""\n  sense: $driver.sense')
    context.install_entry(entry)
    context.use_point(PointUseRequest("first"))
    assert context.get_alias_rows() == PRESET


def test_switch_point_reloads_local_table_without_setup_or_preset_fallback(
    tmp_path: Path,
) -> None:
    entry = entry_at(tmp_path)
    entry.new_point("second")
    entry.save_aliases(EntryAliases(aliases={"setup_only": "Q1"}))
    entry.save_aliases(EntryAliases(aliases={"driver": "R1"}), point="second")
    context = service()
    context.install_entry(entry)
    assert context.get_alias_rows() == ()
    context.use_point(PointUseRequest("first"))
    assert context.get_alias_rows() == ()
    context.set_alias_rows(AUTHORED)
    context.use_point(PointUseRequest("second"))
    assert context.get_alias_rows() == (AliasRow("driver", "R1"),)
    assert context.read_context().value("$driver.rate") == 6500.0
    context.use_point(PointUseRequest("first"))
    assert context.get_alias_rows() == AUTHORED


@pytest.mark.parametrize(
    "body,version,expected_error",
    [
        ("[]", "1.0", ValueError),
        ("\n  driver: null", "1.0", ValueError),
        ("\n  driver: 7", "1.0", ValueError),
        ("\n  driver: $sense.rate\n  sense: R1", "1.0", InvalidInputError),
        ("{}", "2.0", ValueError),
    ],
)
def test_bad_file_preserves_old_binding_and_aliases(
    tmp_path: Path, body: str, version: str, expected_error: type[Exception]
) -> None:
    context = service()
    first = entry_at(tmp_path, "one")
    context.install_entry(first)
    context.use_point(PointUseRequest("first"))
    binding = context.get_source_binding()
    rows = context.get_alias_rows()
    entry = first
    entry.new_point("second")
    (entry.result_path / "points" / "second" / "aliases.yaml").write_text(
        f'format: zcu.aliases\nformat_version: "{version}"\naliases: {body}\n',
        encoding="utf-8",
    )
    with pytest.raises(expected_error) as failure:
        context.use_point(PointUseRequest("second"))
    if isinstance(failure.value, InvalidInputError):
        assert failure.value.reason_code == "alias_invalid"
    assert context.get_source_binding() == binding
    assert context.get_alias_rows() == rows
    info = context.get_entry_info()
    assert info is not None
    assert info.entry_id == first.entry_id


def test_apply_round_trip_entry_isolation_and_write_through_set(tmp_path: Path) -> None:
    first = entry_at(tmp_path, "one")
    second = entry_at(tmp_path, "two")
    context = service()
    context.install_entry(first)
    context.use_point(PointUseRequest("first"))
    context.apply_alias_rows(AUTHORED)
    saved = first.load_aliases(point="first")
    assert list(saved.aliases.items()) == [
        ("driver", "Q1"),
        ("sense", "$driver.sense"),
        ("spare", ""),
    ]
    context.set_alias_rows((AliasRow("transient", "Q1"),))
    assert first.load_aliases(point="first").aliases == {"transient": "Q1"}
    reopened = ResultEntry.open(
        "one", result_root=tmp_path / "result", database_root=tmp_path / "Database"
    )
    fresh = service()
    fresh.install_entry(reopened)
    fresh.use_point(PointUseRequest("first"))
    assert fresh.get_alias_rows() == (AliasRow("transient", "Q1"),)
    fresh.install_entry(second)
    fresh.use_point(PointUseRequest("first"))
    fresh.apply_alias_rows(())
    assert second.load_aliases(point="first").aliases == {}
    assert first.load_aliases().aliases == {}
    fresh.install_entry(reopened)
    fresh.use_point(PointUseRequest("first"))
    assert fresh.get_alias_rows() == (AliasRow("transient", "Q1"),)
    fresh.install_entry(second)
    fresh.use_point(PointUseRequest("first"))
    assert fresh.get_alias_rows() == ()


def test_apply_failure_preserves_file_rows_and_events_then_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bus = BaseEventBus()
    context = service(bus)
    entry = entry_at(tmp_path)
    context.install_entry(entry)
    context.use_point(PointUseRequest("first"))
    context.apply_alias_rows(AUTHORED)
    events: list[AliasesChangedPayload] = []
    bus.subscribe(AliasesChangedPayload, events.append)

    def fail_replace(source: Path, destination: Path) -> Path:
        raise OSError("synthetic replace failure")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", fail_replace)
        with pytest.raises(OSError, match="synthetic replace failure"):
            context.apply_alias_rows(())
    assert context.get_alias_rows() == AUTHORED
    assert not events
    fresh = service()
    fresh.install_entry(entry)
    fresh.use_point(PointUseRequest("first"))
    assert fresh.get_alias_rows() == AUTHORED
    context.apply_alias_rows(())
    assert context.get_alias_rows() == ()
    assert len(events) == 1


@pytest.mark.parametrize("opened", [False, True])
def test_apply_without_point_fails_without_changing_memory_or_setup(
    tmp_path: Path,
    opened: bool,
) -> None:
    context = service()
    entry = entry_at(tmp_path)
    entry.save_aliases(EntryAliases(aliases={"setup": "Q1"}))
    if opened:
        context.install_entry(entry)
    context.set_alias_rows(AUTHORED)
    with pytest.raises(FailedPreconditionError) as failure:
        context.apply_alias_rows(())
    assert failure.value.reason_code == "no_context"
    assert context.get_alias_rows() == AUTHORED
    assert entry.load_aliases().aliases == {"setup": "Q1"}
    context.clear_context()
    assert context.get_alias_rows() == ()


def test_invalid_apply_preserves_point_table(tmp_path: Path) -> None:
    context = service()
    entry = entry_at(tmp_path)
    context.install_entry(entry)
    context.use_point(PointUseRequest("first"))
    with pytest.raises(InvalidInputError):
        context.apply_alias_rows((AliasRow("driver", "$later.rate"),))
    assert entry.load_aliases(point="first").aliases == {}
    assert context.get_alias_rows() == ()


def test_missing_names_are_diagnosed_only_when_used(tmp_path: Path) -> None:
    context = service()
    entry = entry_at(tmp_path)
    write_table(entry, "\n  driver: Gone\n  flux_dev: MissingDevice")
    context.install_entry(entry)
    context.use_point(PointUseRequest("first"))
    capture = context.read_context()
    assert capture.value("Q1.rate") == 4300.0
    assert capture.value("$driver") == "Gone"
    assert capture.value("$flux_dev") == "MissingDevice"
    with pytest.raises(RoleResolutionError, match="Unknown path prefix 'Gone'"):
        capture.value("$driver.rate")
    with pytest.raises(
        RoleResolutionError, match="Unknown path prefix 'MissingDevice'"
    ):
        capture.value("$flux_dev.rate")


def test_temporary_capture_and_historical_storage_stay_fixed(tmp_path: Path) -> None:
    """Regression guard: temporary overrides and capture isolation already work."""
    context = service()
    entry = entry_at(tmp_path, "one")
    context.install_entry(entry)
    context.use_point(PointUseRequest("first"))
    context.set_alias_rows(AUTHORED)
    capture = context.read_context()
    storage = context.capture_storage(capture)
    temporary = context.read_context(aliases={"driver": "R1"})
    assert temporary.value("$driver.rate") == 6500.0
    assert context.get_alias_rows() == AUTHORED
    context.install_entry(entry_at(tmp_path, "two"))
    assert capture.value("$driver.rate") == 4300.0
    assert context.capture_storage(capture) == storage
    assert str(storage.entry_id) == entry.entry_id
