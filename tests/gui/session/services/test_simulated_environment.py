"""Simulated-environment assembly through real session owners and their ports."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest
from zcu_tools.device.fake import FakeDevice, FakeDeviceInfo
from zcu_tools.gui.app.measure.state import SessionEnv, State
from zcu_tools.gui.event_bus import BaseEventBus, EventOrigin
from zcu_tools.gui.session.events import (
    SimulatedEnvironmentFinishedPayload,
    SocChangedPayload,
)
from zcu_tools.gui.session.hardware_gate import RunBlocksHardwareGate
from zcu_tools.gui.session.operation_handles import OperationHandles, OperationOutcome
from zcu_tools.gui.session.operation_runner import OperationRunner
from zcu_tools.gui.session.ports import OperationConflictError, OperationKind
from zcu_tools.gui.session.services.connection import (
    ConnectMockRequest,
    SoCConnectionService,
)
from zcu_tools.gui.session.services.device import ConnectDeviceRequest, DeviceService
from zcu_tools.gui.session.services.predictor import PredictorService
from zcu_tools.gui.session.services.progress import ProgressService
from zcu_tools.gui.session.services.simulated_environment import (
    FAKE_FLUX_DEVICE_NAME,
    FAKE_FLUX_INITIAL_VALUE,
    SimulatedEnvironmentCoordinator,
)
from zcu_tools.gui.session.state import DeviceStatus
from zcu_tools.program.v2.mocksoc import MockQickSoc

from tests.gui._progress_fakes import DirectProgressTransport
from tests.gui.session.services._device_fakes import FakeDeviceRegistry


class DeferredBackground:
    def __init__(self) -> None:
        self.pending: deque[
            tuple[Callable[[], Any], Callable[[Any], None], Callable[[Exception], None]]
        ] = deque()

    def submit(
        self,
        work: Callable[[], Any],
        *,
        run_in_pool: bool,
        on_done: Callable[[Any], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        self.pending.append((work, on_done, on_error))

    def deliver(self) -> None:
        work, on_done, on_error = self.pending.popleft()
        try:
            result = work()
        except Exception as exc:  # noqa: BLE001 — executor delivers worker failures
            on_error(exc)
        else:
            on_done(result)

    def drain(self) -> None:
        while self.pending:
            self.deliver()


class Session:
    def __init__(self) -> None:
        self.state = State(
            SessionEnv(md=MagicMock(), ml=MagicMock(), soc=None, soccfg=None)
        )
        self.bus = BaseEventBus()
        self.gate = RunBlocksHardwareGate(run_kind="run", bus=self.bus)
        self.handles = OperationHandles()
        self.background = DeferredBackground()
        self.registry = FakeDeviceRegistry()
        self.drivers: dict[str, Any] = {}
        progress = ProgressService(DirectProgressTransport())
        runner = OperationRunner(
            self.gate, self.handles, progress, self.background, self.bus
        )
        self.device = DeviceService(
            self.bus,
            self.state,
            self.gate,
            self.background,
            runner,
            self.handles,
            driver_factory=self.make_driver,
            device_registry=self.registry,
        )
        self.connection = SoCConnectionService(
            self.state, self.bus, self.gate, self.handles, runner
        )
        self.predictor = PredictorService(self.state, self.bus)
        self.environment = SimulatedEnvironmentCoordinator(
            self.bus,
            self.device,
            self.connection,
            self.predictor,
            self.gate,
            self.handles,
        )
        self.finished: list[SimulatedEnvironmentFinishedPayload] = []
        self.bus.subscribe(SimulatedEnvironmentFinishedPayload, self.finished.append)

    def make_driver(self, type_name: str, address: str) -> Any:
        if address in self.drivers:
            return self.drivers[address]
        if type_name == "FakeDevice":
            return FakeDevice()
        raise ValueError(f"No driver for {address}")

    def connect_real(self, name: str, *, fail_close: bool = False) -> MagicMock:
        driver = MagicMock()
        driver.get_info.return_value = FakeDeviceInfo(address=name, value=0.0)
        if fail_close:
            driver.close.side_effect = OSError("close refused")
        self.drivers[name] = driver
        self.device.start_connect_device(ConnectDeviceRequest("YokoGS200", name, name))
        self.background.drain()
        return driver


@pytest.fixture
def session() -> Session:
    return Session()


def test_source_ready_before_soc_publication_and_terminal_origin(
    session: Session,
) -> None:
    published_values: list[tuple[float, DeviceStatus | None]] = []
    terminal_origins: list[EventOrigin] = []

    def observe_soc(payload: SocChangedPayload) -> None:
        assert isinstance(payload.soc, MockQickSoc)
        assert payload.soc.flux_source is not None
        snapshot = session.device.get_device_snapshot(FAKE_FLUX_DEVICE_NAME)
        published_values.append(
            (payload.soc.flux_source(), snapshot.status if snapshot else None)
        )

    session.bus.subscribe(SocChangedPayload, observe_soc)
    session.bus.subscribe_with_meta(
        SimulatedEnvironmentFinishedPayload,
        lambda _payload, meta: terminal_origins.append(meta.origin),
    )
    with session.bus.origin(EventOrigin("agent", "client")):
        token = session.environment.start()
    assert session.handles.known_outcome(token) is None
    assert not session.connection.has_soc()
    session.background.deliver()  # Registered, but initialization has not finished.
    assert FAKE_FLUX_DEVICE_NAME in session.registry.get_all_devices()
    assert not session.connection.has_soc()
    with pytest.raises(OperationConflictError):
        session.gate.ensure_can_start("run")
    session.background.drain()
    assert published_values == [(FAKE_FLUX_INITIAL_VALUE, DeviceStatus.CONNECTED)]
    assert session.handles.known_outcome(token) == OperationOutcome("finished")
    assert session.finished[-1].success
    assert terminal_origins == [EventOrigin("agent", "client", str(token))]
    assert session.predictor.get_predictor() is not None
    assert session.handles.live_count() == 0
    assert session.gate.snapshot() == ()


def test_disconnects_real_devices_then_reuses_valid_simulation(
    session: Session,
) -> None:
    real = session.connect_real("real")
    token = session.environment.start()
    assert not session.connection.has_soc()
    session.background.deliver()
    real.close.assert_called_once()
    assert "real" not in session.registry.get_all_devices()
    session.background.drain()
    assert session.handles.known_outcome(token) == OperationOutcome("finished")
    source = session.registry.get_device(FAKE_FLUX_DEVICE_NAME)
    source.setup(FakeDeviceInfo(address="none", value=0.2))
    soc = session.state.session_env.soc
    predictor = session.predictor.get_predictor()
    another = session.connect_real("another")
    session.environment.start()
    session.background.drain()
    another.close.assert_called_once()
    assert session.registry.get_device(FAKE_FLUX_DEVICE_NAME) is source
    assert source.get_value() == 0.2
    assert session.state.session_env.soc is soc
    assert session.predictor.get_predictor() is predictor


def test_disconnect_failure_reports_partial_state_without_publishing_mock(
    session: Session,
) -> None:
    first = session.connect_real("first")
    failed = session.connect_real("failed", fail_close=True)
    last = session.connect_real("last")
    token = session.environment.start()
    session.background.drain()
    first.close.assert_called_once()
    failed.close.assert_called_once()
    last.close.assert_not_called()
    assert not session.connection.has_soc()
    assert set(session.registry.get_all_devices()) == {"failed", "last"}
    error = session.finished[-1].error_message
    assert error is not None
    assert "Disconnected: ['first']" in error
    assert "not disconnected: ['failed', 'last']" in error
    assert session.handles.known_outcome(token) == OperationOutcome("failed", error)
    assert session.gate.snapshot() == ()


def test_low_level_mock_connect_does_not_assemble_environment(session: Session) -> None:
    session.connection.connect_sync(ConnectMockRequest())
    assert session.connection.has_soc()
    assert session.registry.get_all_devices() == {}
    assert session.predictor.get_predictor() is None
    assert session.finished == []


@pytest.mark.parametrize(
    "kind", ["run", OperationKind.SOC_CONNECT, OperationKind.DEVICE_SETUP]
)
def test_busy_entry_rejected_before_mutation(session: Session, kind: str) -> None:
    real = session.connect_real("real")
    session.gate.register(999, kind, owner_id="busy", origin_kind="user", note="test")
    try:
        with pytest.raises(OperationConflictError):
            session.environment.start()
        real.close.assert_not_called()
        assert session.background.pending == deque()
        assert session.handles.live_count() == 0
    finally:
        session.gate.release(999)


def test_source_setup_failure_cleans_only_new_source(session: Session) -> None:
    source = FakeDevice()
    source.setup = MagicMock(side_effect=OSError("setup refused"))
    session.drivers[""] = source
    token = session.environment.start()
    session.background.drain()
    assert not session.connection.has_soc()
    assert session.registry.get_all_devices() == {}
    error = session.finished[-1].error_message
    assert error is not None and "setup refused" in error
    assert session.handles.known_outcome(token) == OperationOutcome("failed", error)
    assert session.gate.snapshot() == ()


def test_wrong_source_type_rejected_at_binding_without_replacing_soc(
    session: Session,
) -> None:
    # A factory claiming FakeDevice but returning another driver must not bind.
    impostor = MagicMock()
    impostor.get_info.return_value = FakeDeviceInfo(address="none", value=0.0)
    session.drivers[""] = impostor
    token = session.environment.start()
    session.background.drain()
    assert not session.connection.has_soc()
    error = session.finished[-1].error_message
    assert error is not None and "must be a FakeDevice" in error
    assert session.handles.known_outcome(token) == OperationOutcome("failed", error)
    impostor.setup.assert_not_called()
    impostor.close.assert_called_once()
    assert session.registry.get_all_devices() == {}


def test_existing_predictor_is_preserved(session: Session) -> None:
    predictor = MagicMock()
    session.predictor.install_predictor(predictor)
    session.environment.start()
    session.background.drain()
    assert session.finished[-1].success
    assert session.predictor.get_predictor() is predictor


def test_cancelled_source_initialization_cleans_new_driver(session: Session) -> None:
    token = session.environment.start()
    session.background.deliver()
    session.device.cancel_device_operation(FAKE_FLUX_DEVICE_NAME)
    session.background.drain()
    error = session.finished[-1].error_message
    assert error is not None and "cancelled" in error
    assert session.handles.known_outcome(token) == OperationOutcome("failed", error)
    assert session.registry.get_all_devices() == {}
    assert not session.connection.has_soc()
    assert session.gate.snapshot() == ()


def test_cleanup_failure_retains_source_and_reports_both_failures(
    session: Session,
) -> None:
    source = FakeDevice()
    source.setup = MagicMock(side_effect=OSError("setup refused"))
    source.close = MagicMock(side_effect=OSError("close refused"))
    session.drivers[""] = source
    token = session.environment.start()
    session.background.drain()
    error = session.finished[-1].error_message
    assert error is not None and "setup refused" in error and "close refused" in error
    assert session.handles.known_outcome(token) == OperationOutcome("failed", error)
    assert session.registry.get_device(FAKE_FLUX_DEVICE_NAME) is source
    assert not session.connection.has_soc()
    assert session.gate.snapshot() == ()


def test_explicit_simulation_parameters_reach_soc_and_predictor(
    session: Session,
) -> None:
    from zcu_tools.program.v2.sim import DEFAULT_SIMPARAM

    params = DEFAULT_SIMPARAM.model_copy(update={"EJ": 6.0, "EC": 1.5, "EL": 0.8})
    session.environment.start(sim_params=params)
    session.background.drain()
    soc = session.state.session_env.soc
    assert isinstance(soc, MockQickSoc)
    assert soc.sim_params is not None and soc.sim_params.EJ == 6.0
    assert session.finished[-1].success


def test_source_connect_failure_settles_environment_without_soc(
    session: Session,
) -> None:
    source = FakeDevice()
    source.get_info = MagicMock(side_effect=OSError("read failed"))
    session.drivers[""] = source
    token = session.environment.start()
    session.background.drain()
    error = session.finished[-1].error_message
    assert error is not None and "read failed" in error
    assert session.handles.known_outcome(token) == OperationOutcome("failed", error)
    assert session.registry.get_all_devices() == {}
    assert not session.connection.has_soc()
    assert session.handles.live_count() == 0
    assert session.gate.snapshot() == ()
