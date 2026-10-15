"""RemoteControlAdapter — measure-gui's second View (driving adapter).

The RPC face onto the Controller, peer to the Qt ``MainWindow`` (ADR-0067 /
ADR-0068): the second user-facing client (user = an automation agent / another
server). The shared dispatch scaffolding (the EndpointRouter seam, the
main-thread marshal, the EventBus push fan-out) lives in
:class:`RemoteControlServiceBase`; this is the *richest* of the three apps and
layers measure-gui's own dispatch policy on top by overriding the base seams:
``_extra_start`` / ``_extra_stop`` wire the out-of-band diagnostic channel
(``notify_diagnostic``), pushed via ``endpoint.broadcast`` independent of
EventBus.

Handlers receive *this adapter* (not the bare ctrl), so they reach tab-resource
commands through ``adapter.tab_control``, run/analyze commands through
``adapter.run_analyze_control``, generic operation await/progress through
``adapter.operation_control``, save commands through ``adapter.save_control``,
writeback commands through ``adapter.writeback_control``,
other app commands through
``adapter.ctrl.<façade>``, context commands through
``adapter.context_control``, device commands through ``adapter.device_control``,
predictor commands through ``adapter.predictor_control``, and View-side surfaces
(render/snapshot) through ``adapter.render_view``. Construct after ``Controller``
/ ``MainWindow`` exist; inert until ``start()``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from qtpy.QtWidgets import QApplication

from zcu_tools.gui.remote.control_service import (
    ControlOptions,
    RemoteControlServiceBase,
)
from zcu_tools.gui.remote.framing import encode_line

if TYPE_CHECKING:
    # Type-only: importing Controller at runtime would form a cycle
    # (controller.py imports remote.dialogs). String annotation keeps pyright
    # checking handler/ctrl method names while the runtime import never happens.
    from zcu_tools.gui.app.measure.controller import Controller, RenderView, Severity
    from zcu_tools.gui.app.measure.services.operation_control import (
        OperationControlPort,
    )
    from zcu_tools.gui.app.measure.services.run_analyze_control import (
        RunAnalyzeControlPort,
    )
    from zcu_tools.gui.app.measure.services.save_control import SaveControlPort
    from zcu_tools.gui.app.measure.services.tab_control import TabControlPort
    from zcu_tools.gui.app.measure.services.tab_form import TabFormResource
    from zcu_tools.gui.app.measure.services.writeback_control import (
        WritebackControlPort,
    )
    from zcu_tools.gui.session.context_control import ContextControlPort
    from zcu_tools.gui.session.device_control import DeviceControlPort
    from zcu_tools.gui.session.ports import OwnerScheduler
    from zcu_tools.gui.session.predictor_control import PredictorControlPort

from .dispatch import METHOD_REGISTRY
from .events import EVENT_SERIALIZERS, wire_event_name
from .wire_version import GUI_VERSION, WIRE_VERSION

logger = logging.getLogger(__name__)


class RemoteControlAdapter(RemoteControlServiceBase):
    """Driving adapter: an NDJSON RPC face onto the measure-gui ``Controller``.

    Holds the concrete ``Controller`` (app command face), exposes the shared
    context/device/predictor-control facets, exposes the app-local tab-control
    run/analyze-control, operation-control, save-control, and writeback-control facets, and pulls EventBus from it via
    ``get_bus()``. Dispatch handlers reach tab commands through ``adapter.tab_control``,
    run/analyze commands through ``adapter.run_analyze_control``, operation
    await/progress through ``adapter.operation_control``, save commands through
    ``adapter.save_control``, writeback commands through
    ``adapter.writeback_control``, other app commands through ``adapter.ctrl``,
    context commands through ``adapter.context_control``, device commands through
    ``adapter.device_control``, predictor commands through ``adapter.predictor_control``,
    and the canvas-bearing View's pure-read surface through ``adapter.render_view``
    (screenshot / snapshot / dialog).
    ``render_view`` is None in a headless process; render handlers fail-fast then.
    """

    def _local_modal_open(self) -> bool:
        return QApplication.activeModalWidget() is not None

    ctrl: Controller
    tab_control: TabControlPort
    cfg_lookup: Callable[[str], TabFormResource]
    run_analyze_control: RunAnalyzeControlPort
    operation_control: OperationControlPort
    save_control: SaveControlPort
    writeback_control: WritebackControlPort
    context_control: ContextControlPort
    device_control: DeviceControlPort
    predictor_control: PredictorControlPort

    def __init__(
        self,
        controller: Controller,
        opts: ControlOptions,
        owner_scheduler: OwnerScheduler,
        render_view: RenderView | None = None,
    ) -> None:
        super().__init__(
            controller,
            opts,
            owner_scheduler=owner_scheduler,
            wire_version=WIRE_VERSION,
            gui_version=GUI_VERSION,
            server_name="RemoteControlServer",
            method_registry=METHOD_REGISTRY,
            control_identity=controller.control_identity,
            event_serializers=EVENT_SERIALIZERS,
            wire_event_name=wire_event_name,
        )
        self.render_view = render_view
        self.tab_control = controller.tab_control
        self.cfg_lookup = controller.cfg_resources.lookup
        self.run_analyze_control = controller.run_analyze_control
        self.operation_control = controller.operation_control
        self.save_control = controller.save_control
        self.writeback_control = controller.writeback_control
        self.context_control = controller.context_control
        self.device_control = controller.device_control
        self.predictor_control = controller.predictor_control

    # ------------------------------------------------------------------
    # Base seams
    # ------------------------------------------------------------------

    def _get_bus(self):
        return self.ctrl.get_bus()

    def _extra_start(self) -> None:
        # Become a diagnostic-only View (ADR-0068): receive ctrl error/info
        # fan-out and push it to clients out-of-band of EventBus.
        self.ctrl.add_diagnostic_sink(self)

    def _extra_stop(self) -> None:
        self.ctrl.remove_diagnostic_sink(self)

    # ------------------------------------------------------------------
    # Diagnostic channel (DiagnosticSink impl) — independent of EventBus
    # ------------------------------------------------------------------

    def notify_diagnostic(self, severity: Severity, title: str, message: str) -> None:
        """Push a Controller diagnostic to every client. Runs on the Qt main
        thread (ctrl fans out there). Deliberately *not* gated by event
        subscription and *not* routed through EventBus — diagnostics must reach
        the agent regardless of what it subscribed to, and a channel that
        reports a fault must not be the faulty channel (ADR-0068)."""
        try:
            line = encode_line(
                {
                    "event": "diagnostic",
                    "payload": {
                        "severity": severity,
                        "title": title,
                        "message": message,
                    },
                }
            )
        except Exception:  # pragma: no cover — payload is plain strings
            logger.exception("failed to encode diagnostic %r/%r", severity, title)
            return
        # Diagnostics reach every client, regardless of subscription.
        self._endpoint.broadcast(line, predicate=lambda link: True)


__all__ = ["ControlOptions", "RemoteControlAdapter"]
