"""SessionControllerMixin — the body of SessionControllerPort, shared by apps.

Both measurement-session app Controllers (measure: ``gui/app/main/controller``;
autofluxdep: ``gui/app/autofluxdep/controller``) implement
:class:`~zcu_tools.gui.session.controller_port.SessionControllerPort` as a wall of
one-line forwards into the same five session services (soc_connection / predictor
/ context / device / startup) plus the progress service. Those forwards were
byte-identical across the two apps; this mixin holds the single copy.

Design (Candidate #14, Option B — abstract service accessors):

- The mixin declares a small set of **abstract accessors** for the services it
  forwards into (``_soc_svc`` / ``_pred_svc`` / ``_ctx_svc`` / ``_dev_svc`` /
  ``_startup_svc`` / ``_progress_svc``), as annotation-only attribute declarations
  with explicit service types. pyright treats each as an attribute the concrete
  Controller must supply, and enforces the declared service type at every forward.
  Each app satisfies them however its own attribute layout already provides the
  service — both measure and autofluxdep keep the same flat ``self._soc_svc`` etc.
  they already assign (measure from ``build_app_services``; autofluxdep aliased off
  its ``SessionServices`` bundle), so NO rename of the controllers' existing
  attributes is needed. (Annotation-only rather than ``@property``: a ``property``
  is a data descriptor whose ``__set__`` raises ``AttributeError``, so the
  controllers' existing ``self._soc_svc = ...`` assignments would crash at runtime;
  a bare annotation gives the identical pyright enforcement with no name clash and
  no churn across the ~130 existing call sites.)
- The mixin provides every **identical** forward reading those accessors.

Each app keeps as its own override the methods whose body genuinely diverges:

- ``apply_startup_project`` — measure returns the resolved-project dict (WIRE-44);
  autofluxdep returns ``bool``. Different return contract, kept per-app.
- ``get_project_root`` — reads the app's own ``self._project_root`` (app state, not
  a session service).
- ``get_bus`` — returns the app-specific ``EventBus`` subtype.

Layering: this module lives in the shared ``gui/session/`` layer, so it MUST NOT
import Qt or ``gui.app.*`` (guarded by ``tests/gui/test_shared_layer.py``). Every
collaborator type — the session services and the request/result dataclasses — is
referenced under ``TYPE_CHECKING`` only, so importing the mixin pulls in nothing
heavy. pyright still enforces the accessor return types and the forward signatures
against the port at each app's dialog call site.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from zcu_tools.device.base import BaseDeviceInfo
    from zcu_tools.gui.session.pbar_host import ProgressBarModel
    from zcu_tools.gui.session.services.connection import (
        ConnectRequest,
        SoCConnectionService,
    )
    from zcu_tools.gui.session.services.context import ContextService
    from zcu_tools.gui.session.services.device import (
        ConnectDeviceRequest,
        DeviceEntry,
        DeviceService,
        DeviceSnapshot,
        DisconnectDeviceRequest,
        SetupDeviceRequest,
    )
    from zcu_tools.gui.session.services.predictor import (
        LoadPredictorRequest,
        PredictCurveRequest,
        PredictCurveResult,
        PredictFreqRequest,
        PredictMatrixCurveRequest,
        PredictMatrixCurveResult,
        PredictorService,
        SetModelParamsRequest,
    )
    from zcu_tools.gui.session.services.progress import ProgressService
    from zcu_tools.gui.session.services.startup import (
        PersistedStartup,
        StartupConnectionRequest,
        StartupService,
    )
    from zcu_tools.gui.session.types import SocCfgHandle
    from zcu_tools.meta_tool import MetaDict, ModuleLibrary


class SessionControllerMixin:
    """The shared body of ``SessionControllerPort`` (identical forwards).

    A Controller mixes this in and supplies the abstract service accessors as its
    own attributes; the forwards then read the services through the accessors.
    """

    # --- abstract service accessors --------------------------------------
    # Annotation-only: the concrete Controller must supply each as an attribute
    # (both apps already assign these exact names). The explicit service types let
    # pyright enforce that each app feeds the forwards the right service type.
    _soc_svc: SoCConnectionService
    _pred_svc: PredictorService
    _ctx_svc: ContextService
    _dev_svc: DeviceService
    _startup_svc: StartupService
    _progress_svc: ProgressService

    # --- setup dialog: startup -------------------------------------------
    def get_persisted_startup(self) -> PersistedStartup:
        return self._startup_svc.get_persisted()

    def remember_startup_connection(self, req: StartupConnectionRequest) -> None:
        self._startup_svc.remember_connection(req)

    # --- setup dialog: context switching ---------------------------------
    def use_context(self, label: str) -> None:
        self._ctx_svc.use_context(label)

    def new_context(
        self,
        bind_device: str | None = None,
        clone_from: str | None = None,
    ) -> None:
        """Create a new flux context, optionally bound to a flux device.

        ``bind_device`` (a connected device name) decides the flux unit/value: the
        unit comes from the device-type whitelist (Fast-Fail if the device is
        unknown or its type is not whitelisted) and the value is *read* from the
        device's current state (never set). ``bind_device=None`` makes an unbound
        context (unit="none", no value). ``clone_from`` is the label of an existing
        context to clone its ml/md from. The new context's label is derived
        automatically by ``ExperimentManager`` — callers cannot name it directly.
        """
        if bind_device is not None:
            unit = self._dev_svc.get_device_unit_strict(bind_device)
            value = self._dev_svc.get_device_value_for_new_context(bind_device)
        else:
            unit, value = "none", None
        self._ctx_svc.new_context(value=value, unit=unit, clone_from=clone_from)

    def get_context_labels(self) -> list[str]:
        return self._ctx_svc.get_context_labels()

    def get_active_context_label(self) -> str | None:
        return self._ctx_svc.get_active_context_label()

    # --- setup dialog: connection ----------------------------------------
    def start_connect(self, req: ConnectRequest) -> int:
        return self._soc_svc.start_connect(req)

    def bind_connection_outcome(
        self,
        on_finished: Callable[[], None],
        on_failed: Callable[[str], None],
    ) -> None:
        """Bind the single connection observer (the open SetupDialog) without
        exposing the service. Drops all prior slots first so a re-created dialog
        does not leak the previous observer (a no-arg ``disconnect()`` clears every
        connection), guaranteeing exactly the latest observer."""
        for signal in (
            self._soc_svc.connection_finished,
            self._soc_svc.connection_failed,
        ):
            try:
                signal.disconnect()
            except (TypeError, RuntimeError):
                pass  # no existing connections
        self._soc_svc.connection_finished.connect(on_finished)
        self._soc_svc.connection_failed.connect(on_failed)

    def get_soccfg(self) -> SocCfgHandle | None:
        return self._soc_svc.get_soccfg()

    def get_device_unit(self, name: str) -> str:
        return self._dev_svc.get_device_unit(name)

    # --- predictor dialog: load / clear / predict ------------------------
    def load_predictor(self, req: LoadPredictorRequest) -> None:
        self._pred_svc.load_predictor(req)

    def set_predictor_model_params(self, req: SetModelParamsRequest) -> None:
        self._pred_svc.set_model_params(req)

    def clear_predictor(self) -> None:
        self._pred_svc.clear_predictor()

    def predict_freq(self, req: PredictFreqRequest) -> float:
        return self._pred_svc.predict_freq(req)

    def predict_freq_curve(self, req: PredictCurveRequest) -> PredictCurveResult:
        return self._pred_svc.predict_freq_curve(req)

    def predict_matrix_element_curve(
        self, req: PredictMatrixCurveRequest
    ) -> PredictMatrixCurveResult:
        return self._pred_svc.predict_matrix_element_curve(req)

    def get_predictor_info(self) -> dict | None:
        return self._pred_svc.get_predictor_info()

    # --- device dialog: lifecycle ----------------------------------------
    def start_connect_device(self, req: ConnectDeviceRequest) -> int:
        return self._dev_svc.start_connect_device(req)

    def start_disconnect_device(self, req: DisconnectDeviceRequest) -> int:
        return self._dev_svc.start_disconnect_device(req)

    def start_reconnect_device(self, name: str) -> int:
        return self._dev_svc.start_reconnect_device(name)

    def start_setup_device(self, req: SetupDeviceRequest) -> int:
        return self._dev_svc.start_setup_device(req)

    def forget_device(self, name: str) -> None:
        self._dev_svc.forget_device(name)

    def cancel_device_operation(self, name: str) -> None:
        self._dev_svc.cancel_device_operation(name)

    # --- device dialog: queries ------------------------------------------
    def list_devices(self) -> list[DeviceEntry]:
        return self._dev_svc.list_devices()

    def get_device_snapshot(self, name: str) -> DeviceSnapshot | None:
        return self._dev_svc.get_device_snapshot(name)

    def get_device_info(self, name: str) -> BaseDeviceInfo | None:
        return self._dev_svc.get_device_info(name)

    def poll_device_info(self, name: str) -> None:
        # Dialog-scoped off-main live-read (best-effort); result flows back via
        # DEVICE_CHANGED. DeviceService owns the worker/main-thread split.
        self._dev_svc.poll_device_info(name)

    def is_memory_device(self, name: str) -> bool:
        return self._dev_svc.is_memory_device(name)

    # --- device dialog: progress -----------------------------------------
    def attach_progress(
        self, owner_id: str, listener: Callable[[], None]
    ) -> Callable[[], None]:
        """A View subscribes (by its own tab_id / device_name) to progress changes
        for that owner; returns a disposer. The listener fires whenever the owner's
        live operation's bars change (and across operation rotation), and re-reads
        via ``progress_bars``."""
        return self._progress_svc.attach_by_owner(owner_id, listener)

    def progress_bars(
        self, owner_id: str
    ) -> tuple[tuple[int, ProgressBarModel], ...]:
        """Live (handle_id, ProgressBarModel) pairs for the owner's current
        operation (empty if none live)."""
        return self._progress_svc.bars_for_owner(owner_id)

    # --- inspect dialog: md edit + ml view/rename/delete ------------------
    def get_current_md(self) -> MetaDict:
        return self._ctx_svc.get_current_md()

    def get_current_ml(self) -> ModuleLibrary:
        return self._ctx_svc.get_current_ml()

    def coerce_md_value(self, key: str, text: str) -> Any:
        return self._ctx_svc.coerce_md_value(key, text)

    def set_md_attr(self, key: str, value: Any) -> None:
        self._ctx_svc.set_md_attr(key, value)

    def del_md_attr(self, key: str) -> None:
        self._ctx_svc.del_md_attr(key)

    def rename_ml_module(self, old: str, new: str) -> None:
        self._ctx_svc.rename_ml_module(old, new)

    def rename_ml_waveform(self, old: str, new: str) -> None:
        self._ctx_svc.rename_ml_waveform(old, new)

    def del_ml_module(self, name: str) -> None:
        self._ctx_svc.del_ml_module(name)

    def del_ml_waveform(self, name: str) -> None:
        self._ctx_svc.del_ml_waveform(name)
