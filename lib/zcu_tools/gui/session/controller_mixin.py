"""SessionControllerMixin — Controller forwards into session services.

The measure Controller (``gui/app/measure/controller``) exposes one-line
forwards into four session services (soc_connection / context / device /
settings). Shared dialogs now use explicit control facets; app-local and
compatibility callers may still use these forwards. The forwarding
implementation lives here.

Design (Candidate #14, Option B — abstract service accessors):

- The mixin declares a small set of **abstract accessors** for the services it
  forwards into (``_soc_svc`` / ``_ctx_svc`` / ``_dev_svc`` /
  ``_settings_svc``), as annotation-only attribute declarations with explicit
  service types. pyright treats each as an attribute the concrete Controller must
  supply, and enforces the declared service type at every forward. Each app
  satisfies them however its own attribute layout already provides the service —
  measure keeps the flat ``self._soc_svc`` etc. assigned from
  ``build_app_services``, so no rename of its existing attributes is needed. (Annotation-only rather than ``@property``: a ``property``
  is a data descriptor whose ``__set__`` raises ``AttributeError``, so the
  controllers' existing ``self._soc_svc = ...`` assignments would crash at
  runtime; a bare annotation gives the identical pyright enforcement with no name
  clash and no churn across the ~130 existing call sites.)
- The mixin provides every **identical** forward reading those accessors.

The device dialog lifecycle/query surface lives on ``DeviceControlPort``; the
predictor dialog load/query/compute surface lives on ``PredictorControlPort``.
App-local run progress widgets use ``ProgressControlPort`` directly. Shared
inspect/remote context consumers use ``ContextControlPort`` directly.

Each app keeps as its own override the methods whose body genuinely diverges:

- ``get_project_root`` — reads the app's own ``self._project_root`` (app state, not
  a session service).
- ``get_bus`` — returns the app-specific ``EventBus`` subtype.

Entry New/Open is not forwarded here: the setup dialog and measure remote
entry handlers use ``SetupControlPort`` directly.

Layering: this module lives in the shared ``gui/session/`` layer, so it MUST NOT
import Qt or ``gui.app.*`` (guarded by ``tests/gui/test_shared_layer.py``). Every
collaborator type — the session services and the request/result dataclasses — is
referenced under ``TYPE_CHECKING`` only, so importing the mixin pulls in nothing
heavy. pyright still enforces the accessor return types and the forward signatures
inside the concrete controllers.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from zcu_tools.gui.event_bus import BaseEventBus, EventSubscriptions
from zcu_tools.gui.session.events import ConnectionFinishedPayload

if TYPE_CHECKING:
    from zcu_tools.gui.session.device_models import DeviceEntry
    from zcu_tools.gui.session.services.connection import (
        ConnectRequest,
        SoCConnectionService,
    )
    from zcu_tools.gui.session.services.context import ContextService
    from zcu_tools.gui.session.services.device import DeviceService
    from zcu_tools.gui.session.services.project_settings import ProjectSettingsService
    from zcu_tools.gui.session.types import SocCfgHandle
    from zcu_tools.gui.session.value_lookup import ScalarValue, ValueInfo


class SessionControllerMixin:
    """Shared identical forwards backed by session services.

    A Controller mixes this in and supplies the abstract service accessors as its
    own attributes; the forwards then read the services through the accessors.
    """

    # --- abstract service accessors --------------------------------------
    # Annotation-only: the concrete Controller must supply each as an attribute
    # (both apps already assign these exact names). The explicit service types let
    # pyright enforce that each app feeds the forwards the right service type.
    _soc_svc: SoCConnectionService
    _ctx_svc: ContextService
    _dev_svc: DeviceService
    _settings_svc: ProjectSettingsService

    def get_bus(self) -> BaseEventBus:
        """Return the concrete app bus; every app controller overrides this."""
        raise NotImplementedError

    # --- setup dialog: project settings ----------------------------------
    def get_left_panel_width(self) -> int:
        return self._settings_svc.get_left_panel_width()

    def list_value_sources(self) -> tuple[ValueInfo, ...]:
        return self._ctx_svc.list_value_sources()

    def read_value_source(
        self, key: str, type_name: str | None = None
    ) -> tuple[ValueInfo, ScalarValue]:
        return self._ctx_svc.read_value_source(key, type_name)

    # --- setup dialog: connection ----------------------------------------
    def start_connect(self, req: ConnectRequest) -> int:
        return self._soc_svc.start_connect(req)

    def bind_connection_outcome(
        self,
        on_finished: Callable[[], None],
        on_failed: Callable[[str], None],
    ) -> None:
        """Replace the single SetupDialog connection-outcome observer."""
        subscriptions = getattr(self, "_connection_outcome_subscriptions", None)
        if subscriptions is not None:
            subscriptions.unsubscribe_all()
        subscriptions = EventSubscriptions()
        self._connection_outcome_subscriptions = subscriptions

        def dispatch(payload: ConnectionFinishedPayload) -> None:
            if payload.success:
                on_finished()
            else:
                on_failed(payload.error_message or "SoC connection failed")

        subscriptions.subscribe(self.get_bus(), ConnectionFinishedPayload, dispatch)

    def get_soccfg(self) -> SocCfgHandle | None:
        return self._soc_svc.get_soccfg()

    def get_device_unit(self, name: str) -> str:
        return self._dev_svc.get_device_unit(name)

    # --- setup dialog: device list/unit lookup ----------------------------
    def list_devices(self) -> list[DeviceEntry]:
        return self._dev_svc.list_devices()
