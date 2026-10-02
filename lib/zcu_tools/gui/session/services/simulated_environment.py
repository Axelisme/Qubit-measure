"""Explicit, owner-loop assembly of the simulated measurement environment.

The coordinator sequences DeviceService operations and publishes through the SoC
owner only after the flux source is ready. It holds one SoC exclusion lease
across the sequence; child device operations retain their own handles and leases.
A failed attempt reports partial disconnections and cleans up only its new driver.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from zcu_tools.device.fake import FakeDevice, FakeDeviceInfo
from zcu_tools.gui.session.events import (
    DeviceOperationFinishedPayload,
    DeviceSetupFinishedPayload,
    SimulatedEnvironmentFinishedPayload,
)
from zcu_tools.gui.session.operation_handles import OperationOutcome
from zcu_tools.gui.session.ports import OperationKind
from zcu_tools.gui.session.services.device import (
    ConnectDeviceRequest,
    DisconnectDeviceRequest,
    SetupDeviceRequest,
)
from zcu_tools.gui.session.state import DeviceStatus

if TYPE_CHECKING:
    from zcu_tools.gui.event_bus import BaseEventBus, EventMeta
    from zcu_tools.gui.session.operation_handles import OperationHandles
    from zcu_tools.gui.session.ports import ExclusionGate
    from zcu_tools.gui.session.services.connection import SoCConnectionService
    from zcu_tools.gui.session.services.device import DeviceService
    from zcu_tools.gui.session.services.predictor import PredictorService
    from zcu_tools.program.v2.sim import SimParams

logger = logging.getLogger(__name__)
FAKE_FLUX_DEVICE_NAME = "fake_flux"
FAKE_FLUX_INITIAL_VALUE = 0.5


@dataclass
class _Attempt:
    token: int
    remaining: list[str]
    sim_params: SimParams | None
    disconnected: list[str] = field(default_factory=list)
    child_token: int | None = None
    created_source: bool = False
    remember_source: bool = False
    failure: str | None = None


class SimulatedEnvironmentCoordinator:
    """One explicit entry, with no reaction to general SoC change notifications.

    Call on the session owner loop. Re-entry while active is rejected by the
    shared gate. Successful repeated entry keeps an already bound MockSoc and
    FakeDevice, including their current operating value and custom parameters.
    """

    def __init__(
        self,
        bus: BaseEventBus,
        device: DeviceService,
        connection: SoCConnectionService,
        predictor: PredictorService,
        gate: ExclusionGate,
        handles: OperationHandles,
    ) -> None:
        self._bus = bus
        self._device = device
        self._connection = connection
        self._predictor = predictor
        self._gate = gate
        self._handles = handles
        self._attempt: _Attempt | None = None
        bus.subscribe_with_meta(
            DeviceOperationFinishedPayload, self._on_device_finished
        )
        bus.subscribe_with_meta(DeviceSetupFinishedPayload, self._on_setup_finished)

    def start(self, *, sim_params: SimParams | None = None) -> int:
        """Disconnect real devices, prepare the flux source, then publish readiness.

        The returned handle covers the whole sequence, not just SoC creation.
        Existing run, SoC connect, or device mutations reject entry before any
        resource changes. Later real-device connections remain allowed.
        """
        self._gate.ensure_can_start(OperationKind.SOC_CONNECT)
        self._gate.ensure_can_start(OperationKind.DEVICE_DISCONNECT)
        devices = self._device.get_connected_devices()
        real_devices = [
            name
            for name, driver in devices.items()
            if not isinstance(driver, FakeDevice)
        ]
        origin = self._bus.current_origin
        token = self._handles.create(cancel_hook=None, origin=origin)
        self._gate.register(
            token,
            OperationKind.SOC_CONNECT,
            owner_id="simulated_environment",
            origin_kind=origin.kind,
            note="set up simulated environment",
        )
        self._attempt = _Attempt(token, real_devices, sim_params)
        with self._bus.origin(self._handles.event_origin(token)):
            try:
                self._disconnect_next()
            except Exception as exc:  # noqa: BLE001 — settle the environment operation
                self._fail(exc)
        return token

    def _disconnect_next(self) -> None:
        attempt = self._require_attempt()
        if attempt.remaining:
            attempt.child_token = self._device.start_disconnect_device(
                DisconnectDeviceRequest(attempt.remaining[0])
            )
        else:
            self._prepare_source()

    def _prepare_source(self) -> None:
        attempt = self._require_attempt()
        source = self._device.get_device_snapshot(FAKE_FLUX_DEVICE_NAME)
        if source is not None and source.status == DeviceStatus.CONNECTED:
            self._publish_environment()
            return
        attempt.remember_source = source is not None
        if source is None:
            attempt.child_token = self._device.start_connect_device(
                ConnectDeviceRequest("FakeDevice", FAKE_FLUX_DEVICE_NAME, "")
            )
        else:
            if source.type_name != "FakeDevice":
                raise TypeError("The simulated flux source must be a FakeDevice")
            attempt.child_token = self._device.start_reconnect_device(
                FAKE_FLUX_DEVICE_NAME
            )

    def _matches_child(self, meta: EventMeta) -> bool:
        attempt = self._attempt
        return (
            attempt is not None
            and attempt.child_token is not None
            and meta.origin.operation_id == str(attempt.child_token)
        )

    def _on_device_finished(
        self, payload: DeviceOperationFinishedPayload, meta: EventMeta
    ) -> None:
        if not self._matches_child(meta):
            return
        attempt = self._require_attempt()
        with self._bus.origin(self._handles.event_origin(attempt.token)):
            if attempt.failure is not None:
                if not payload.success:
                    attempt.failure += (
                        f"; source cleanup failed: {payload.error_message}"
                    )
                self._finish(attempt.failure)
                return
            try:
                if not payload.success:
                    raise RuntimeError(
                        payload.error_message or f"{payload.action} failed"
                    )
                if payload.action == "disconnect":
                    attempt.disconnected.append(attempt.remaining.pop(0))
                    self._disconnect_next()
                else:
                    attempt.created_source = True
                    self._get_flux_source()
                    attempt.child_token = self._device.start_setup_device(
                        SetupDeviceRequest(
                            FAKE_FLUX_DEVICE_NAME,
                            FakeDeviceInfo(
                                address="none", value=FAKE_FLUX_INITIAL_VALUE
                            ),
                        )
                    )
            except Exception as exc:  # noqa: BLE001 — settle a failed continuation
                self._fail(exc)

    def _on_setup_finished(
        self, payload: DeviceSetupFinishedPayload, meta: EventMeta
    ) -> None:
        if not self._matches_child(meta):
            return
        attempt = self._require_attempt()
        with self._bus.origin(self._handles.event_origin(attempt.token)):
            try:
                if payload.outcome != "finished":
                    raise RuntimeError(
                        payload.error_message or "Flux source setup cancelled"
                    )
                self._publish_environment()
            except Exception as exc:  # noqa: BLE001 — settle a failed continuation
                self._fail(exc)

    def _get_flux_source(self) -> FakeDevice:
        self._gate.ensure_can_start(
            OperationKind.DEVICE_SETUP, resource_id=FAKE_FLUX_DEVICE_NAME
        )
        source = self._device.get_connected_devices()[FAKE_FLUX_DEVICE_NAME]
        if not isinstance(source, FakeDevice):
            raise TypeError("The simulated flux source must be a FakeDevice")
        return source

    def _publish_environment(self) -> None:
        from zcu_tools.gui.session.services.predictor_from_sim import (
            build_predictor_from_simparams,
        )
        from zcu_tools.program.v2.mocksoc import make_mock_soc
        from zcu_tools.program.v2.sim import DEFAULT_SIMPARAM

        source = self._get_flux_source()
        sim_params = self._require_attempt().sim_params
        pair = (
            self._connection.get_bound_mock(source.get_value)
            if sim_params is None
            else None
        )
        if pair is None:
            pair = make_mock_soc(
                sim=sim_params if sim_params is not None else DEFAULT_SIMPARAM
            )
            pair[0].set_flux_source(source.get_value)
        soc, soccfg = pair
        if self._predictor.get_predictor() is None:
            if soc.sim_params is None:
                raise ValueError(
                    "Simulated environment requires physics simulation parameters"
                )
            predictor = build_predictor_from_simparams(soc.sim_params)
            self._predictor.install_predictor(predictor)
        self._connection.install_prepared_mock(soc, soccfg)
        self._finish(None)

    def _fail(self, exc: Exception) -> None:
        attempt = self._require_attempt()
        logger.error("Simulated environment setup failed", exc_info=exc)
        attempt.failure = (
            f"Simulated environment incomplete: {exc}. "
            f"Disconnected: {attempt.disconnected}; not disconnected: {attempt.remaining}"
        )
        if attempt.created_source:
            try:
                attempt.child_token = self._device.start_disconnect_device(
                    DisconnectDeviceRequest(
                        FAKE_FLUX_DEVICE_NAME, remember=attempt.remember_source
                    )
                )
                return
            except Exception as cleanup_error:  # noqa: BLE001 — retain cleanup failure
                attempt.failure += f"; source cleanup failed: {cleanup_error}"
        self._finish(attempt.failure)

    def _finish(self, error: str | None) -> None:
        attempt = self._require_attempt()
        self._attempt = None
        try:
            self._handles.settle(
                attempt.token,
                OperationOutcome("finished" if error is None else "failed", error),
            )
        finally:
            self._gate.release(attempt.token)
        self._bus.emit(
            SimulatedEnvironmentFinishedPayload(
                success=error is None, error_message=error
            )
        )

    def _require_attempt(self) -> _Attempt:
        if self._attempt is None:
            raise RuntimeError("No simulated environment setup is active")
        return self._attempt
