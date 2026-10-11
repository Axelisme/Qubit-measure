"""Public fixed alias captures without consumer declarations or kind policy."""

from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import pytest
from zcu_tools.cfg_model import ConstWaveformCfg
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.context_types import PointUseRequest
from zcu_tools.gui.session.events import AliasesChangedPayload
from zcu_tools.gui.session.expression import evaluate_scalar_expr
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import RoleBinding, SessionEnv
from zcu_tools.resources.entry import ResultEntry, RoleResolutionError, source_path
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from tests.resources.entry.fakes import register_fakes, registry_state, restore_registry


@dataclass
class CaptureSession:
    entry: ResultEntry
    service: ContextService
    control: ContextControlFacet
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
            entry.setup.add_component(name, kind="fake/drive/a", rate=rate, sense=sense)
        for name, rate in (("R1", 6123.0), ("R2", 7200.0), ("driver", 123.0)):
            entry.setup.add_component(name, kind="fake/sensor", rate=rate)
        entry.new_point("first")
        entry.new_point("second")
        state = SessionState(SessionEnv(parameters=None, soc=None, soccfg=None))
        bus = BaseEventBus()
        service = ContextService(state, bus, bind_assets=ArbWaveformDatabase.info_at)
        service.install_entry(entry)
        service.use_point(PointUseRequest("first"))
        service.set_alias_rows(
            (AliasRow("driver", "Q1"), AliasRow("sense", "$driver.sense"))
        )
        yield CaptureSession(
            entry, service, ContextControlFacet(context=service), state, bus
        )
        assert registry_state() == installed, "capture tests polluted registries"
    finally:
        restore_registry(before)


def test_default_capture_reads_all_aliases_without_kind_declarations(
    session: CaptureSession,
) -> None:
    rows = (AliasRow("driver", "R1"), AliasRow("custom", "Q2"))
    session.control.set_alias_rows(rows)
    captured = session.control.read_context()
    assert captured.aliases == rows
    assert captured.value("$driver.rate") == 6123.0
    assert evaluate_scalar_expr("$driver.rate + driver.rate", captured) == 6246.0
    assert captured.value("$custom.rate") == 5100.0
    assert dict(captured.capture_storage().roles) == {"driver": "R1", "custom": "Q2"}


def test_capture_keeps_point_library_storage_and_rows_after_later_changes(
    session: CaptureSession,
) -> None:
    control = session.control
    target = control.read_context().target
    control.set_waveform(target, "Q1.waveforms.base", ConstWaveformCfg(length=0.05))
    captured = control.read_context()
    assert captured.aliases == control.get_alias_rows()
    control.set_parameter(target, "Q1.rate", 4700.0)
    control.set_waveform(target, "Q1.waveforms.base", ConstWaveformCfg(length=0.10))
    control.set_alias_rows(
        (AliasRow("driver", "Q2"), AliasRow("sense", "$driver.sense"))
    )
    control.use_point(PointUseRequest("second"))
    source_path(session.entry.use_point("first")).unlink()
    source_path(session.entry.use_point("second")).unlink()
    assert captured.value("$driver.rate") == 4300.0
    assert captured.value("$sense.rate") == 6123.0
    assert captured.library.waveform_spec("Q1.waveforms.base").length == 0.05
    assert captured.capture_storage().point == "first"
    assert dict(captured.capture_storage().roles) == {"driver": "Q1", "sense": "R1"}
    assert control.read_context().value("$driver.rate") == 5100.0
    assert control.read_context().target != target


def test_named_override_keeps_dependent_links_without_global_side_effects(
    session: CaptureSession,
) -> None:
    control = session.control
    original = control.read_context()
    rows = control.get_alias_rows()
    assert original.aliases == rows
    environment = session.state.session_env
    binding = control.get_source_binding()
    version = session.state.version.get("context")
    ledger = session.entry.ledger.events()
    events: list[AliasesChangedPayload] = []
    session.bus.subscribe(AliasesChangedPayload, events.append)
    captured = control.read_context(aliases={"driver": "Q2", "extra": "R1"})
    assert captured.aliases == (
        AliasRow("driver", "Q2"),
        AliasRow("sense", "$driver.sense"),
        AliasRow("extra", "R1"),
    )
    assert captured.value("$sense.rate") == 7200.0
    assert captured.value("$extra.rate") == 6123.0
    assert original.value("$sense.rate") == 6123.0
    assert control.get_alias_rows() == rows
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
        ((AliasRow("driver", "absent"),), "driver"),
        ((AliasRow("driver", "Q1"), AliasRow("sense", "$driver.ext.missing")), "sense"),
    ],
    ids=["missing-row", "unbound", "missing-component", "missing-link-field"],
)
def test_alias_failures_name_requested_alias_without_concrete_fallback(
    session: CaptureSession, rows: tuple[AliasRow, ...], alias: str
) -> None:
    session.control.set_alias_rows(rows)
    captured = session.control.read_context()
    assert captured.aliases == rows
    assert captured.value("driver.rate") == 123.0
    with pytest.raises(RoleResolutionError) as error:
        captured.value(f"${alias}.rate")
    assert error.value.role == alias


def test_read_target_preserves_historical_names_in_saved_reads(
    session: CaptureSession,
) -> None:
    control = session.control
    captured = control.read_context()
    assert captured.aliases == control.get_alias_rows()
    history = captured.roles
    control.set_alias_rows(
        (AliasRow("driver", "Q2"), AliasRow("sense", "$driver.sense"))
    )
    fixed = control.read_target(captured.target, roles=history)
    assert fixed.value("$driver.rate") == 4300.0
    assert fixed.value("$sense.rate") == 6123.0
    control.use_point(PointUseRequest("second"))
    assert control.read_target(captured.target, roles=history).roles == history


def test_empty_read_context_keeps_no_context_failure(session: CaptureSession) -> None:
    session.service.clear_context()
    with pytest.raises(FailedPreconditionError) as error:
        session.control.read_context()
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
    target = session.control.read_context().target
    binding = RoleBinding(None, {}, names, explicit)
    fixed = session.control.read_target(target, roles=binding)
    session.control.set_alias_rows((AliasRow("driver", "Q1"),))
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
    first = session.entry.use_point("first")
    second = session.entry.use_point("second")
    with second.edit() as edit:
        edit.set("Q1.rate", 4700.0)
    ledger = session.entry.ledger.events()
    assert control.read_context().aliases == control.get_alias_rows()
    captured = control.read_context(aliases=names)
    original_target = captured.target
    assert dict(captured.capture_storage().roles) == names
    assert captured.resolve_path("$driver.rate") == f"{names['driver']}.rate"
    assert captured.value("$driver.rate") == old_value
    control.set_alias_rows((AliasRow("driver", "R2"),))
    assert dict(control.read_context(aliases=names).capture_storage().roles) == names
    control.use_point(PointUseRequest("second"))
    source_path(first).unlink()
    source_path(second).unlink()
    assert captured.target == original_target
    assert captured.value("$driver.rate") == old_value
    current = control.read_context(aliases=names)
    assert dict(current.capture_storage().roles) == names
    assert current.value("$driver.rate") == current_value
    assert current.target != captured.target
    assert current.library.document() == captured.library.document()
    assert session.entry.ledger.events() == ledger
