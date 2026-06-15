"""FLUX-AWARE-MOCK: mock-connect auto-provisions a fake_flux FakeDevice and binds
it as the MockSoc's SimEngine flux source.

These tests drive the *real* Controller connect path (real EventBus + real
ConnectionService / DeviceService) so the SocChangedPayload hook, the
DeviceService registration, and the soc.set_flux_device binding are all exercised
together — the same path "Use MockSoc" and gui_connect_start(kind='mock') take.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import QCoreApplication
from zcu_tools.device import GlobalDeviceManager
from zcu_tools.device.fake import FakeDevice
from zcu_tools.experiment.v2_gui.adapters.fake import FakeAdapter
from zcu_tools.experiment.v2_gui.registry import register_all
from zcu_tools.gui.app.main.adapter import ExpContext
from zcu_tools.gui.app.main.controller import (
    _FAKE_FLUX_DEVICE_NAME,
    _FAKE_FLUX_INITIAL_VALUE,
    Controller,
)
from zcu_tools.gui.app.main.registry import Registry
from zcu_tools.gui.app.main.state import DeviceStatus, State
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.services.connection import (
    ConnectMockRequest,
    ConnectRemoteRequest,
)
from zcu_tools.gui.session.services.device import DisconnectDeviceRequest
from zcu_tools.gui.session.services.io_manager import IOManager
from zcu_tools.meta_tool import MetaDict, ModuleLibrary

# ---------------------------------------------------------------------------
# Fixture: a real Controller wired to a real bus, starting disconnected.
# ---------------------------------------------------------------------------


def _empty_ctx() -> ExpContext:
    # soc=None so the connect path actually runs (no pre-connected MagicMock soc).
    return ExpContext(md=MetaDict(), ml=ModuleLibrary(), soc=None, soccfg=None)


class _Fixture:
    def __init__(self) -> None:
        self.state = State(_empty_ctx())
        self.registry = Registry()
        register_all(self.registry)
        if not self.registry.has("fake"):
            self.registry.register("fake", FakeAdapter)
        self.bus = BaseEventBus()
        io_manager = IOManager()
        io_manager._em = MagicMock()
        self.view = MagicMock()
        self.view.make_live_container = MagicMock(return_value=None)
        self.view.notify_diagnostic = MagicMock()
        self.ctrl = Controller(
            state=self.state,
            registry=self.registry,
            io_manager=io_manager,
            view=self.view,
            bus=self.bus,
        )

    def quiesce(self) -> None:
        self.ctrl._background_svc.quiesce()


@pytest.fixture()
def fx(qapp) -> Iterator[_Fixture]:  # noqa: ARG001
    fixture = _Fixture()
    yield fixture
    fixture.quiesce()
    # fake_flux is registered in the process-global GlobalDeviceManager; drop it so
    # tests stay independent (a second connect would otherwise see it "already
    # registered" at the driver layer).
    GlobalDeviceManager.drop_device(_FAKE_FLUX_DEVICE_NAME, ignore_error=True)


def _process_events() -> None:
    app = QCoreApplication.instance()
    assert app is not None
    app.processEvents()


def _fake_device(name: str) -> FakeDevice:
    dev = GlobalDeviceManager.get_device(name)
    assert isinstance(dev, FakeDevice)
    return dev


def _pump_until(condition: Callable[[], bool], timeout_ms: int = 3000) -> bool:
    app = QCoreApplication.instance()
    assert app is not None
    deadline = time.monotonic() + timeout_ms / 1000
    while time.monotonic() < deadline:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    return False


def _connect_mock(fx: _Fixture) -> None:
    fx.ctrl.start_connect(ConnectMockRequest())
    # Connect (mock) dispatches via QTimer.singleShot(0); then fake_flux connect +
    # setup are async background ops. Pump until fake_flux is CONNECTED with the
    # default operating value applied.
    assert _pump_until(
        lambda: (
            (dev := fx.state.get_device(_FAKE_FLUX_DEVICE_NAME)) is not None
            and dev.status is DeviceStatus.CONNECTED
            and dev.info is not None
            and getattr(dev.info, "value", None) == _FAKE_FLUX_INITIAL_VALUE
        )
    ), "fake_flux was not provisioned to the default operating value"


# ---------------------------------------------------------------------------
# Mock connect provisions fake_flux + binds it on the soc.
# ---------------------------------------------------------------------------


def test_mock_connect_registers_fake_flux_device(fx):
    _connect_mock(fx)

    dev = fx.state.get_device(_FAKE_FLUX_DEVICE_NAME)
    assert dev is not None
    assert dev.type_name == "FakeDevice"
    assert dev.status is DeviceStatus.CONNECTED
    # FakeDevice -> unit "none".
    assert fx.ctrl.get_device_unit(_FAKE_FLUX_DEVICE_NAME) == "none"


def test_mock_connect_sets_soc_flux_device(fx):
    _connect_mock(fx)

    soc = fx.state.exp_context.soc
    assert soc is not None
    # set_flux_device records the name on the soc's internal SimParams copy.
    assert getattr(soc, "_sim_params").flux_device == _FAKE_FLUX_DEVICE_NAME


def test_mock_connect_initial_value_is_reduced_flux_one(fx):
    """The provisioned value (0.5 under DEFAULT_SIMPARAM) places the mock at the
    documented default operating point (reduced flux = 1.0, f01 ~5423 MHz)."""
    _connect_mock(fx)

    dev = fx.state.get_device(_FAKE_FLUX_DEVICE_NAME)
    assert dev is not None and dev.info is not None
    assert getattr(dev.info, "value") == _FAKE_FLUX_INITIAL_VALUE
    assert _FAKE_FLUX_INITIAL_VALUE == 0.5


# ---------------------------------------------------------------------------
# Idempotent re-connect: the has-device guard prevents double registration.
# ---------------------------------------------------------------------------


def test_reconnect_does_not_double_register_fake_flux(fx):
    _connect_mock(fx)
    # Hand-edit the device value to detect any re-provisioning stomp.
    _fake_device(_FAKE_FLUX_DEVICE_NAME).set_value(0.123)

    # Connect again (e.g. the user presses "Use MockSoc" a second time). The mock
    # soc allows re-connect; the SocChangedPayload fires again.
    fx.ctrl.start_connect(ConnectMockRequest())
    # Let the singleShot + any (unwanted) background ops run.
    assert _pump_until(lambda: not fx.ctrl._conn_svc.is_connect_active())
    fx.quiesce()
    _process_events()

    # Exactly one fake_flux entry, value untouched (no re-ramp), binding intact.
    assert fx.state.get_device(_FAKE_FLUX_DEVICE_NAME) is not None
    assert _fake_device(_FAKE_FLUX_DEVICE_NAME).get_value() == 0.123
    soc = fx.state.exp_context.soc
    assert getattr(soc, "_sim_params").flux_device == _FAKE_FLUX_DEVICE_NAME


# ---------------------------------------------------------------------------
# FLUX-AWARE-MOCK auto-reconnect: when fake_flux is MEMORY_ONLY (disconnected)
# on mock-connect, the controller should reconnect it rather than skip.
# ---------------------------------------------------------------------------


def _disconnect_fake_flux(fx: _Fixture) -> None:
    """Disconnect fake_flux so it lands in MEMORY_ONLY state, then quiesce."""
    fx.ctrl._dev_svc.start_disconnect_device(
        DisconnectDeviceRequest(name=_FAKE_FLUX_DEVICE_NAME, remember=True)
    )
    assert _pump_until(
        lambda: (
            (dev := fx.state.get_device(_FAKE_FLUX_DEVICE_NAME)) is not None
            and dev.status is DeviceStatus.MEMORY_ONLY
        )
    ), "fake_flux did not reach MEMORY_ONLY after disconnect"


def test_mock_connect_reconnects_disconnected_fake_flux(fx):
    """FLUX-AWARE-MOCK: if fake_flux is MEMORY_ONLY (e.g. restored from persistence
    in disconnected state), Use MockSoc must auto-reconnect it so the device becomes
    live again without a manual user action."""
    # First connect: provisions and ramps fake_flux.
    _connect_mock(fx)

    # Simulate the 'disconnected at startup' scenario: disconnect the device.
    _disconnect_fake_flux(fx)
    assert fx.state.get_device(_FAKE_FLUX_DEVICE_NAME) is not None
    assert (
        fx.state.get_device(_FAKE_FLUX_DEVICE_NAME).status is DeviceStatus.MEMORY_ONLY
    )  # type: ignore[union-attr]

    # Use MockSoc again — the controller must fire the reconnect path.
    fx.ctrl.start_connect(ConnectMockRequest())
    # Wait for fake_flux to come back CONNECTED (reconnect is async).
    assert _pump_until(
        lambda: (
            (dev := fx.state.get_device(_FAKE_FLUX_DEVICE_NAME)) is not None
            and dev.status is DeviceStatus.CONNECTED
        )
    ), "fake_flux was not reconnected after Use MockSoc with MEMORY_ONLY device"

    # Binding must still be in place on the new soc.
    soc = fx.state.exp_context.soc
    assert getattr(soc, "_sim_params").flux_device == _FAKE_FLUX_DEVICE_NAME


def test_mock_connect_skips_reconnect_when_already_connected(fx):
    """FLUX-AWARE-MOCK: if fake_flux is already CONNECTED, Use MockSoc must NOT
    trigger a redundant reconnect — only the set_flux_device binding is repeated."""
    _connect_mock(fx)
    # Record a sentinel value; a spurious reconnect would reset it to 0.0.
    _fake_device(_FAKE_FLUX_DEVICE_NAME).set_value(0.777)

    # Use MockSoc a second time while fake_flux is still CONNECTED.
    fx.ctrl.start_connect(ConnectMockRequest())
    assert _pump_until(lambda: not fx.ctrl._conn_svc.is_connect_active())
    fx.quiesce()
    _process_events()

    # Value must be untouched — no reconnect / re-setup fired.
    assert _fake_device(_FAKE_FLUX_DEVICE_NAME).get_value() == 0.777
    soc = fx.state.exp_context.soc
    assert getattr(soc, "_sim_params").flux_device == _FAKE_FLUX_DEVICE_NAME


# ---------------------------------------------------------------------------
# FLUX-AWARE-MOCK: mock connect also installs a SimParams-matched predictor.
# ---------------------------------------------------------------------------


def test_mock_connect_installs_sim_predictor(fx):
    """Mock connect installs a FluxoniumPredictor derived from the mock soc's
    SimParams, so predict_freq matches the SimEngine's physics out of the box."""
    from zcu_tools.gui.session.services.predictor_from_sim import (
        build_predictor_from_simparams,
    )

    _connect_mock(fx)

    predictor = fx.ctrl._pred_svc.get_predictor()
    assert predictor is not None

    # The installed predictor predicts the same f01 as one built directly from the
    # mock soc's own SimParams (the production builder, reused here).
    soc = fx.state.exp_context.soc
    sim_params = soc.sim_params  # type: ignore[union-attr]
    assert sim_params is not None
    reference = build_predictor_from_simparams(sim_params)
    # Compare at the provisioned operating value (reduced flux = 1.0).
    value = _FAKE_FLUX_INITIAL_VALUE
    assert abs(predictor.predict_freq(value) - reference.predict_freq(value)) < 1e-6


def test_mock_connect_does_not_overwrite_user_predictor(fx):
    """A predictor the user already loaded must survive a subsequent mock connect:
    the provisioner installs its sim predictor only when none is present."""
    from zcu_tools.simulate.fluxonium.predict import FluxoniumPredictor

    # User loads their own predictor (distinct params) before connecting.
    user_predictor = FluxoniumPredictor(
        params=(5.0, 1.2, 0.9), flux_half=0.1, flux_period=2.0, flux_bias=0.05
    )
    fx.ctrl._pred_svc.install_predictor(user_predictor)

    _connect_mock(fx)

    # The user's predictor is untouched (identity preserved).
    assert fx.ctrl._pred_svc.get_predictor() is user_predictor


# ---------------------------------------------------------------------------
# Remote connect must NOT provision fake_flux.
# ---------------------------------------------------------------------------


def test_remote_connect_does_not_provision_fake_flux(fx, monkeypatch):
    """A non-mock connect leaves fake_flux unregistered and never calls
    set_flux_device (the remote soc is not a MockQickSoc)."""

    # Stub the remote connect to return a plain (non-mock) soc so the worker path
    # completes without real hardware. SocChangedPayload(is_mock=False) results.
    fake_soc = MagicMock(name="remote_soc")
    fake_soccfg = MagicMock(name="remote_soccfg")
    monkeypatch.setattr(
        "zcu_tools.remote.make_soc_proxy",
        lambda ip, port: (fake_soc, fake_soccfg),
        raising=False,
    )

    fx.ctrl.start_connect(ConnectRemoteRequest(ip="127.0.0.1", port=1234))
    assert _pump_until(lambda: fx.state.exp_context.soc is fake_soc)
    fx.quiesce()
    _process_events()

    assert fx.state.get_device(_FAKE_FLUX_DEVICE_NAME) is None
    fake_soc.set_flux_device.assert_not_called()
