"""Public fixed alias captures without consumer declarations or kind policy."""

from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import pytest
from zcu_tools.cfg_model import ConstWaveformCfg
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_control import ContextControlPort
from zcu_tools.gui.session.context_types import DocumentSetRef
from zcu_tools.gui.session.events import (
    AliasesChangedPayload,
    ContextSwitchedPayload,
    EntryDocumentsChangedPayload,
    FileChangedPayload,
    MdChangedPayload,
    MlChangedPayload,
)
from zcu_tools.gui.session.expression import evaluate_scalar_expr
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import RoleBinding, SessionEnv
from zcu_tools.resources.entry import ResultEntry, RoleResolutionError, source_path
from zcu_tools.resources.entry.aliases import EntryAliases
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from tests.gui._context_reads import capture_focus
from tests.gui.app.measure._record_support import require_target
from tests.resources.entry.fakes import register_fakes, registry_state, restore_registry


@dataclass
class CaptureSession:
    entry: ResultEntry
    service: ContextService
    control: ContextControlPort
    state: SessionState
    bus: BaseEventBus


@pytest.fixture
def session(tmp_path: Path) -> Generator[CaptureSession]:
    before = registry_state()
    try:
        register_fakes()
        installed = registry_state()
        entry = ResultEntry.create(
            "entry",
            result_root=tmp_path / "Result",
            database_root=tmp_path / "Database",
        )
        for name, rate, sense in (("Q1", 4300.0, "R1"), ("Q2", 5100.0, "R2")):
            entry.parameters().add_component(
                name, kind="fake/drive/a", rate=rate, sense=sense
            )
        for name, rate in (("R1", 6123.0), ("R2", 7200.0), ("driver", 123.0)):
            entry.parameters().add_component(name, kind="fake/sensor", rate=rate)
        entry.new_point("first")
        entry.new_point("second")
        state = SessionState(SessionEnv(parameters=None, soc=None, soccfg=None))
        bus = BaseEventBus()
        service = ContextService(state, bus, bind_assets=ArbWaveformDatabase.info_at)
        service.install_entry(entry)
        service.use_point("first")
        service.apply_alias_rows(
            (AliasRow("driver", "Q1"), AliasRow("sense", "$driver.sense")),
            point=service.current_point_label(),
        )
        yield CaptureSession(entry, service, service, state, bus)
        assert registry_state() == installed, "capture tests polluted registries"
    finally:
        restore_registry(before)


def test_global_library_watcher_updates_only_global_inspect_and_revision(
    qapp,
    session: CaptureSession,
) -> None:
    from zcu_tools.gui.session.adapters import FileWatcher

    control = session.control
    active = capture_focus(control)
    old_global = control.read_context(point=None)
    environment = session.state.session_env
    version = session.state.version.get("context")
    before = control.get_installed_file_revisions()
    assert len(before) == 6
    path = session.entry.result_path / "library.yaml"
    watcher = FileWatcher(session.service, session.bus, admit_reload=lambda: True)
    try:
        path.write_text(
            'format: zcu.module-library\nformat_version: "2.0"\n'
            "components: {Q1: {waveforms: {global: {style: const, length: 0.2}}}}\n"
        )
        watcher.flush()
        assert (
            control.read_context(point=None)
            .library.waveform_spec("Q1.waveforms.global")
            .length
            == 0.2
        )
        assert capture_focus(control).library.waveform_keys() == ()
        assert active.library.waveform_keys() == ()
        assert old_global.library.waveform_keys() == ()
        assert session.state.session_env is environment
        assert session.state.version.get("context") == version
        after = control.get_installed_file_revisions()
        assert len(after) == 6
        assert control.get_file_revisions() == before[:3]
        assert [new.path for new in after] == [old.path for old in before]
        assert [
            new.path for old, new in zip(before, after, strict=True) if old != new
        ] == [path]
        assert next(file for file in after if file.path == path).revision is not None
        assert control.get_file_sync_status()[0].state == "reloaded"
    finally:
        watcher.close()


def test_default_capture_reads_all_aliases_without_kind_declarations(
    session: CaptureSession,
) -> None:
    rows = (AliasRow("driver", "R1"), AliasRow("custom", "Q2"))
    session.control.apply_alias_rows(rows, point=session.control.current_point_label())
    captured = capture_focus(session.control)
    assert captured.aliases == rows
    assert captured.value("$driver.rate") == 6123.0
    assert evaluate_scalar_expr("$driver.rate + driver.rate", captured) == 6246.0
    assert captured.value("$custom.rate") == 5100.0
    assert dict(captured.capture_storage().roles) == {"driver": "R1", "custom": "Q2"}


def test_capture_keeps_point_library_storage_and_rows_after_later_changes(
    session: CaptureSession,
) -> None:
    control = session.control
    target = require_target(capture_focus(control))
    control.set_waveform(target, "Q1.waveforms.base", ConstWaveformCfg(length=0.05))
    captured = capture_focus(control)
    assert captured.aliases == control.get_alias_rows(
        point=control.current_point_label()
    )
    control.set_parameter(target, "Q1.rate", 4700.0)
    control.set_waveform(target, "Q1.waveforms.base", ConstWaveformCfg(length=0.10))
    control.apply_alias_rows(
        (AliasRow("driver", "Q2"), AliasRow("sense", "$driver.sense")),
        point=control.current_point_label(),
    )
    session.entry.aliases(point="second").save(
        EntryAliases(aliases={"driver": "Q2", "sense": "$driver.sense"})
    )
    control.use_point("second")
    source_path(session.entry.parameters("first")).unlink()
    source_path(session.entry.parameters("second")).unlink()
    assert captured.value("$driver.rate") == 4300.0
    assert captured.value("$sense.rate") == 6123.0
    assert captured.library.waveform_spec("Q1.waveforms.base").length == 0.05
    assert captured.capture_storage().point == "first"
    assert dict(captured.capture_storage().roles) == {"driver": "Q1", "sense": "R1"}
    assert capture_focus(control).value("$driver.rate") == 5100.0
    assert capture_focus(control).target != target


def test_named_override_keeps_dependent_links_without_global_side_effects(
    session: CaptureSession,
) -> None:
    control = session.control
    original = capture_focus(control)
    rows = control.get_alias_rows(point=control.current_point_label())
    assert original.aliases == rows
    environment = session.state.session_env
    binding = control.get_source_binding()
    version = session.state.version.get("context")
    ledger = session.entry.ledger.events()
    events: list[AliasesChangedPayload] = []
    session.bus.subscribe(AliasesChangedPayload, events.append)
    captured = control.read_context(
        aliases={"driver": "Q2", "extra": "R1"}, point=control.current_point_label()
    )
    assert captured.aliases == (
        AliasRow("driver", "Q2"),
        AliasRow("sense", "$driver.sense"),
        AliasRow("extra", "R1"),
    )
    assert captured.value("$sense.rate") == 7200.0
    assert captured.value("$extra.rate") == 6123.0
    assert original.value("$sense.rate") == 6123.0
    assert control.get_alias_rows(point=control.current_point_label()) == rows
    assert session.state.session_env is environment
    assert control.get_source_binding() == binding
    assert session.state.version.get("context") == version
    assert session.entry.ledger.events() == ledger
    assert events == []


@pytest.mark.parametrize(
    ("rows", "alias"),
    [
        ((AliasRow("other", "Q1"),), "driver"),
        ((AliasRow("driver", ""),), "driver"),
        ((AliasRow("driver", "Q1"), AliasRow("sense", "$driver.ext.missing")), "sense"),
    ],
    ids=["missing-row", "unbound", "missing-link-field"],
)
def test_alias_failures_name_requested_alias_without_concrete_fallback(
    session: CaptureSession, rows: tuple[AliasRow, ...], alias: str
) -> None:
    session.control.apply_alias_rows(rows, point=session.control.current_point_label())
    captured = capture_focus(session.control)
    assert captured.aliases == rows
    assert captured.value("driver.rate") == 123.0
    with pytest.raises(RoleResolutionError) as error:
        captured.value(f"${alias}.rate")
    assert error.value.role == alias


def test_unknown_alias_name_is_readable_but_its_component_field_is_not(
    session: CaptureSession,
) -> None:
    rows = (AliasRow("driver", "absent"),)
    session.control.apply_alias_rows(rows, point=session.control.current_point_label())
    captured = capture_focus(session.control)
    assert captured.aliases == rows
    assert captured.value("driver.rate") == 123.0
    assert captured.value("$driver") == "absent"
    with pytest.raises(RoleResolutionError) as error:
        captured.value("$driver.rate")
    assert error.value.role == "absent"


def test_read_target_preserves_historical_names_in_saved_reads(
    session: CaptureSession,
) -> None:
    control = session.control
    captured = capture_focus(control)
    assert captured.aliases == control.get_alias_rows(
        point=control.current_point_label()
    )
    history = captured.roles
    control.apply_alias_rows(
        (AliasRow("driver", "Q2"), AliasRow("sense", "$driver.sense")),
        point=control.current_point_label(),
    )
    fixed = control.read_target(require_target(captured), roles=history)
    assert fixed.value("$driver.rate") == 4300.0
    assert fixed.value("$sense.rate") == 6123.0
    control.use_point("second")
    assert control.read_target(require_target(captured), roles=history).roles == history


@pytest.mark.parametrize("point", [None, "second"], ids=["global", "other-point"])
def test_read_target_reads_fresh_values_without_alias_io_or_publication(
    session: CaptureSession, point: str | None
) -> None:
    control = session.control
    history = capture_focus(control)
    control.apply_alias_rows(
        (AliasRow("driver", "Q2"), AliasRow("sense", "R2")),
        point=control.current_point_label(),
    )
    target = DocumentSetRef(entry_name="entry", point=point)
    external = ResultEntry.open(
        "entry",
        result_root=session.entry.result_path.parent,
        database_root=session.entry.database_path.parent,
    )
    view = external.parameters(point)
    view.Q1.rate = 4800.0
    view.Q1.sense = "R2"
    directory = source_path(view).parent
    (directory / "library.yaml").write_text(
        'format: zcu.module-library\nformat_version: "2.0"\n'
        "components: {Q1: {waveforms: {fresh: {style: const, length: 0.25}}}}\n"
    )
    aliases = directory / "aliases.yaml"
    aliases.unlink(missing_ok=True)
    aliases.mkdir()
    selected = capture_focus(control)
    binding = control.get_source_binding()
    environment = session.state.session_env
    versions = session.state.version.snapshot()
    revisions = control.get_installed_file_revisions()
    events: list[object] = []
    for payload in (
        AliasesChangedPayload,
        ContextSwitchedPayload,
        EntryDocumentsChangedPayload,
        FileChangedPayload,
        MdChangedPayload,
        MlChangedPayload,
    ):
        session.bus.subscribe(payload, events.append)

    current = control.read_target(target, roles=history.roles)

    assert current.target == target
    assert current.value("$driver.rate") == 4800.0
    assert current.value("$sense.rate") == 6123.0
    assert current.library.waveform_spec("Q1.waveforms.fresh").length == 0.25
    assert current.roles == history.roles
    storage = current.capture_storage()
    assert (storage.entry_name, storage.point) == ("entry", point)
    assert dict(storage.roles) == {"driver": "Q1", "sense": "R1"}
    assert history.value("$driver.rate") == 4300.0
    assert history.library.waveform_keys() == ()
    assert control.current_point_label() == "first"
    assert control.get_source_binding() == binding
    assert capture_focus(control).target == selected.target
    assert capture_focus(control).value("$driver.rate") == 5100.0
    assert session.state.session_env is environment
    assert session.state.version.snapshot() == versions
    assert control.get_installed_file_revisions() == revisions
    assert events == []


def test_empty_read_context_keeps_no_context_failure(session: CaptureSession) -> None:
    session.service.clear_context()
    with pytest.raises(FailedPreconditionError) as error:
        capture_focus(session.control)
    assert error.value.reason_code == "no_context"


@pytest.mark.parametrize(
    ("explicit", "names"),
    [
        ({"driver": "Q2"}, {"driver": "Q2", "sense": "R2"}),
        ({"driver": None}, {}),
        ({"driver": None, "sense": "R2"}, {"sense": "R2"}),
    ],
)
def test_manual_history_keeps_fixed_names_storage_and_unbound_errors(
    session: CaptureSession,
    explicit: dict[str, str | None],
    names: dict[str, str],
) -> None:
    ledger = session.entry.ledger.events()
    target = require_target(capture_focus(session.control))
    binding = RoleBinding(None, {}, names, explicit)
    fixed = session.control.read_target(target, roles=binding)
    session.control.apply_alias_rows(
        (AliasRow("driver", "Q1"),), point=session.control.current_point_label()
    )
    assert fixed.roles == binding
    assert dict(binding.explicit) == explicit
    assert dict(fixed.capture_storage().roles) == names
    assert fixed.value("Q1.rate") == 4300.0
    if "driver" not in names:
        with pytest.raises(RoleResolutionError) as error:
            fixed.value("$driver.rate")
        assert error.value.role == "driver"
    assert session.entry.ledger.events() == ledger


@pytest.mark.parametrize(
    ("names", "old_value", "current_value"),
    [
        ({"driver": "Q1", "sense": "R1"}, 4300.0, 4700.0),
        ({"driver": "Q2", "sense": "R2"}, 5100.0, 5100.0),
    ],
)
def test_named_published_reads_keep_fixed_names_and_values_without_source_io(
    session: CaptureSession,
    names: dict[str, str],
    old_value: float,
    current_value: float,
) -> None:
    control = session.control
    first = session.entry.parameters("first")
    second = session.entry.parameters("second")
    with second.edit() as edit:
        edit.set("Q1.rate", 4700.0)
    ledger = session.entry.ledger.events()
    assert capture_focus(control).aliases == control.get_alias_rows(
        point=control.current_point_label()
    )
    captured = control.read_context(aliases=names, point=control.current_point_label())
    original_target = captured.target
    assert dict(captured.capture_storage().roles) == names
    assert captured.resolve_path("$driver.rate") == f"{names['driver']}.rate"
    assert captured.value("$driver.rate") == old_value
    control.apply_alias_rows(
        (AliasRow("driver", "R2"),), point=control.current_point_label()
    )
    assert (
        dict(
            control.read_context(aliases=names, point=control.current_point_label())
            .capture_storage()
            .roles
        )
        == names
    )
    control.use_point("second")
    source_path(first).unlink()
    source_path(second).unlink()
    assert captured.target == original_target
    assert captured.value("$driver.rate") == old_value
    current = control.read_context(aliases=names, point=control.current_point_label())
    assert dict(current.capture_storage().roles) == names
    assert current.value("$driver.rate") == current_value
    assert current.target != captured.target
    assert current.library.document() == captured.library.document()
    assert session.entry.ledger.events() == ledger
