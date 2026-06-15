"""MockFluxProvisioner — auto-provision the offline MockSoc's flux source.

FLUX-AWARE-MOCK: when the user connects the offline MockSoc, a flux source must
exist so a mock spectroscopy / flux sweep actually responds to flux. The mock
SimEngine reads the operating flux live from a named FakeDevice in
GlobalDeviceManager (engine._reduced_operating_flux); without a binding it is
pinned at the fixed reduced flux = 1.0. A measurement-session GUI has no
persistent "the flux device" single source of truth, so this mints one on
mock-connect rather than asking the user to wire it up by hand for every offline
session.

This lives in the session layer (not an app controller) because every
measurement-session app (measure / autofluxdep) builds on ``DeviceService`` +
the session ``EventBus`` and wants the exact same behaviour; promoting it here
makes the provisioning a single shared implementation both apps reuse via
``build_session_services`` (the apps subscribe nothing extra). It depends on the
session ``DeviceService`` (for register / reconnect / setup + state queries), the
``ConnectionService`` (for installing the sim predictor through the existing
predictor seam), and ``SocChangedPayload`` on the bus — all session-layer concepts.

Two Use-Mock effects share one owner and one ``SOC_CHANGED`` handler:
  1. the ``fake_flux`` source binding/provisioning (so a sweep responds to flux);
  2. a ``FluxoniumPredictor`` derived from the mock soc's *own* SimParams, installed
     through the predictor seam so the predicted f01 matches the simulated physics
     out of the box — but never stomping a predictor the user already loaded.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from zcu_tools.gui.session.events import SocChangedPayload
from zcu_tools.gui.session.services.device import (
    ConnectDeviceRequest,
    SetupDeviceRequest,
)
from zcu_tools.gui.session.state import DeviceStatus

if TYPE_CHECKING:
    from zcu_tools.gui.event_bus import BaseEventBus
    from zcu_tools.gui.session.services.device import DeviceService
    from zcu_tools.gui.session.services.predictor import PredictorService

logger = logging.getLogger(__name__)

# FLUX-AWARE-MOCK: the auto-provisioned flux source for the offline MockSoc.
# A fixed well-known name so the binding (soc.set_flux_device) and the
# device-manager lookup agree across reconnects.
FAKE_FLUX_DEVICE_NAME = "fake_flux"

# FLUX-AWARE-MOCK: initial FakeDevice value placing the mock at reduced flux = 1.0
# (the documented default operating point, f01 ~5423 MHz) under DEFAULT_SIMPARAM.
FAKE_FLUX_INITIAL_VALUE = 0.5


class MockFluxProvisioner:
    """Subscribes SOC_CHANGED and provisions/binds the fake_flux source.

    Constructed by ``build_session_services`` with the session bus + DeviceService;
    it self-subscribes to ``SocChangedPayload`` and chains the one-shot
    initial-value ramp off ``DeviceService.device_connected``. The app controllers
    need no flux-provisioning code of their own.
    """

    def __init__(
        self,
        bus: BaseEventBus,
        device: DeviceService,
        predictor: PredictorService,
    ) -> None:
        self._bus = bus
        self._dev_svc = device
        # PredictorService owns the predictor seam (exp_context.predictor +
        # PredictorChangedPayload). The provisioner installs the sim predictor
        # through it rather than poking exp_context directly, so the one write path
        # (set_context + event emit) stays unduplicated.
        self._pred_svc = predictor
        # FLUX-AWARE-MOCK: set while our auto-provisioned fake_flux connect is in
        # flight, so the connect-success handler ramps it to the default operating
        # point exactly once (and never touches a user's own FakeDevice).
        self._fake_flux_pending_setup: bool = False

        # SocChangedPayload is the right hook: it fires on every *successful*
        # connect (never on disconnect — that path emits nothing), carries
        # ``is_mock`` and the concrete ``soc`` handle, and is emitted on the Qt
        # main thread (ConnectionService._apply_connection), so this honours the
        # State main-thread invariant and routes through DeviceService.
        bus.subscribe(SocChangedPayload, self._on_soc_changed)
        # The initial-value ramp is chained off connect-success (connect carries
        # no initial-value channel; a setup cannot overlap the connect because
        # DeviceService serialises mutations through the OperationGate).
        device.device_connected.connect(self._on_device_connected)

    def _on_soc_changed(self, payload: SocChangedPayload) -> None:
        """On a successful mock connect, bind a FakeDevice as the SimEngine's flux
        source (FLUX-AWARE-MOCK).

        Only the offline MockSoc is flux-aware; ``payload.is_mock`` gates this so a
        real remote connect is untouched. Two effects, both on the Qt main thread:

        1. ``soc.set_flux_device(fake_flux)`` records the binding on the mock soc's
           internal SimParams. This only stores a *name*; the engine resolves and
           type-checks the device lazily at acquire time, so binding before the
           device exists is fine (the order here is therefore not load-bearing).
        2. Provision the ``fake_flux`` FakeDevice through DeviceService so it lands
           in State.devices — that is what lets the user edit its value in the
           device dialog and lets ``new_context(bind_device=...)`` read it. Going
           through ``GlobalDeviceManager.register_device`` directly would skip
           State and leave the GUI unable to see or change the value.

        Re-connecting the mock (or pressing "Use MockSoc" again) must not double-
        register or stomp a fake_flux the user already has: a status guard makes
        the provisioning idempotent. Disconnect deliberately does NOT drop
        fake_flux (it persists as a remembered device across reconnects).
        """
        if not payload.is_mock or payload.soc is None:
            return

        # The flux binding lives on the concrete MockQickSoc; SocProtocol does not
        # declare set_flux_device (it is mock-only). Narrow via the concrete type
        # so a non-mock soc that somehow set is_mock=True fails loudly rather than
        # silently no-op'ing.
        from zcu_tools.program.v2.mocksoc import MockQickSoc

        soc = payload.soc
        if not isinstance(soc, MockQickSoc):
            raise TypeError(
                "is_mock connect did not yield a MockQickSoc "
                f"(got {type(soc).__name__}); cannot bind fake_flux"
            )
        soc.set_flux_device(FAKE_FLUX_DEVICE_NAME)

        # FLUX-AWARE-MOCK: install a predictor matching this soc's simulated physics
        # so a fresh mock session predicts f01 consistently out of the box.
        self._provision_sim_predictor(soc)

        existing = self._dev_svc.get_device_snapshot(FAKE_FLUX_DEVICE_NAME)

        if existing is not None and existing.status is not DeviceStatus.MEMORY_ONLY:
            # Already live (connected or in a transient operation) — nothing to do;
            # the binding above is the only needed effect.
            return

        if existing is not None and existing.status is DeviceStatus.MEMORY_ONLY:
            # Remembered but disconnected (e.g. restored from persistence in a
            # disconnected state). Auto-reconnect via the same path as
            # gui_device_reconnect so the device becomes live again. No
            # initial-value setup: the user's last-known value is already persisted
            # and is loaded by the driver on connect; we never stomp it.
            # Fire-and-forget (async via BackgroundService).
            self._dev_svc.start_reconnect_device(FAKE_FLUX_DEVICE_NAME)
            return

        # Not recorded at all — provision a brand-new FakeDevice (first connect).
        self._fake_flux_pending_setup = True
        try:
            self._dev_svc.start_connect_device(
                ConnectDeviceRequest(
                    type_name="FakeDevice",
                    name=FAKE_FLUX_DEVICE_NAME,
                    address="none",
                    remember=True,
                )
            )
        except Exception:
            # Provisioning failed to even start — clear the pending flag so a later
            # connect can retry. The mock still works, just pinned at the fixed
            # operating point until a flux device is bound.
            self._fake_flux_pending_setup = False
            raise

    def _provision_sim_predictor(self, soc: object) -> None:
        """Install a FluxoniumPredictor matching the mock soc's SimParams.

        FLUX-AWARE-MOCK. Read the *actual* params off the connected soc (not the
        DEFAULT_SIMPARAM constant) so a future parameterised mock stays consistent.
        Overwrite policy: never stomp a predictor the user already loaded — if
        ``exp_context.predictor`` is already set, leave it. Either way, log INFO so
        the choice is visible. A white-noise mock (no SimParams) has no physics to
        derive from, so it is skipped (the fake_flux binding above already raised if
        the soc was somehow not a real mock).
        """
        from zcu_tools.gui.session.services.predictor_from_sim import (
            build_predictor_from_simparams,
        )

        sim_params = getattr(soc, "sim_params", None)
        if sim_params is None:
            # White-noise mock: no SimParams to derive a predictor from.
            logger.info(
                "MockFluxProvisioner: mock soc carries no SimParams; "
                "no sim predictor installed"
            )
            return

        if self._pred_svc.get_predictor() is not None:
            logger.info(
                "MockFluxProvisioner: predictor already present; "
                "leaving the user's predictor untouched"
            )
            return

        predictor = build_predictor_from_simparams(sim_params)
        self._pred_svc.install_predictor(predictor)
        logger.info(
            "MockFluxProvisioner: installed sim predictor from mock SimParams "
            "(EJ=%r EC=%r EL=%r)",
            sim_params.EJ,
            sim_params.EC,
            sim_params.EL,
        )

    def _on_device_connected(self, req: ConnectDeviceRequest) -> None:
        """Ramp the just-connected fake_flux to the default operating point.

        FLUX-AWARE-MOCK: a freshly connected FakeDevice sits at value 0.0 (reduced
        flux 0.5, f01 ~582 MHz). When *we* auto-provisioned the flux source on
        mock-connect, ramp it to the documented default operating point (reduced
        flux 1.0) via a setup. The flag makes this a one-shot tied to our own
        provisioning, so a user reconnecting their own FakeDevice is never silently
        re-ramped.
        """
        if req.name == FAKE_FLUX_DEVICE_NAME and self._fake_flux_pending_setup:
            self._fake_flux_pending_setup = False
            self._setup_fake_flux_initial_value()

    def _setup_fake_flux_initial_value(self) -> None:
        """Set the just-connected fake_flux to the default operating value.

        Routed through DeviceService.start_setup_device (the same path the device
        dialog uses) rather than poking the driver, so State's cached info reflects
        the new value.
        """
        from zcu_tools.device.fake import FakeDeviceInfo

        self._dev_svc.start_setup_device(
            SetupDeviceRequest(
                name=FAKE_FLUX_DEVICE_NAME,
                # address "none" mirrors the FakeDevice's own address (see
                # FakeDevice.get_info); BaseDeviceInfo.address is required.
                info=FakeDeviceInfo(address="none", value=FAKE_FLUX_INITIAL_VALUE),
            )
        )
