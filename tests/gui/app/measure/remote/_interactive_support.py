"""Interactive analysis builders, socket requests and frontend input simulation."""

from __future__ import annotations

import socket
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal
from uuid import uuid4

import numpy as np
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from qtpy.QtWidgets import QPushButton
from zcu_tools.datafile import RunMetadata, RunSnapshot, SoftwareProvenance
from zcu_tools.experiment.native_persistence import save_run
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import AnalyzeRequest
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.program.v2.sweep import SweepCfg

from tests.zcu_lab.v2.onetone.flux_dep.flux_dep_support import (
    make_cfg as make_onetone_cfg,
)
from zcu_lab.v2._support.measure.flux_pick_frontend import FluxPickFrontend
from zcu_lab.v2._support.measure.flux_pick_plugin import (
    make_flux_pick_plugin,
    render_flux_pick,
)
from zcu_lab.v2._support.measure.interactive_flux_pick import FluxPickParams
from zcu_lab.v2.onetone.flux_dep.core import FluxDepExp, FluxDepResult
from zcu_lab.v2.twotone.fluxdep.core import FreqFluxResult

from ._helpers import Fixture, recv_response, send


class DeferredBackground:
    def __init__(self) -> None:
        self.pending = []

    def run_background(self, compute, on_done, on_error) -> None:
        self.pending.append((compute, on_done, on_error))


class InteractiveFixture(Fixture):
    def __init__(self, *, project_root: str) -> None:
        super().__init__(active_label="interactive-test", project_root=project_root)
        self.widgets: list[FluxPickFrontend] = []


def load_native_onetone(
    fx: InteractiveFixture, tab_id: str, path: Path, result: FluxDepResult
) -> None:
    """Write and publicly load history captured from the current session source."""
    parameters = fx.ctrl.context_control.read_context()
    assert parameters is not None
    storage = parameters.capture_storage()
    cfg = make_onetone_cfg()
    cfg.sweep.freq = SweepCfg(
        start=float(result.freqs[0]),
        stop=float(result.freqs[-1]),
        step=float((result.freqs[-1] - result.freqs[0]) / (len(result.freqs) - 1)),
        expts=len(result.freqs),
    )
    cfg.sweep.flux = SweepCfg(
        start=float(result.values[0]),
        stop=float(result.values[-1]),
        step=float((result.values[-1] - result.values[0]) / (len(result.values) - 1)),
        expts=len(result.values),
    )
    spec = FluxDepExp().persistence_spec()
    metadata = RunMetadata(
        run_id="20261008T000000Z-a1b2c3",
        experiment=spec.tag,
        started_at="2026-10-08T00:00:00+00:00",
        finished_at="2026-10-08T00:01:00+00:00",
        completion="complete",
        labber_path=None,
        snapshot=RunSnapshot(
            entry_id=storage.entry_id,
            entry_name=storage.entry_name,
            point=storage.point,
            description=storage.description,
            roles=storage.roles,
        ),
        provenance=SoftwareProvenance(
            software_versions={},
            git_commit=None,
            git_dirty=None,
            qick_version=None,
            soc_fingerprint=None,
            hostname=None,
        ),
    )
    save_run(RunRecord(cfg, result), path, spec=spec, metadata=metadata)
    fx.ctrl.load_tab_result(tab_id, str(path))
    assert fx.state.get_tab(tab_id).run.metadata == metadata


def start_mounted(
    fx: InteractiveFixture,
    window: MainWindow,
    adapter: str,
    *,
    analyze_params: FluxPickParams | None = None,
    pane: Literal["run", "data"] | None = None,
    native_source: Path | None = None,
):
    tab_id = fx.ctrl.new_tab(adapter)
    devs = np.linspace(-5.0, 5.0, 60)
    freqs = np.linspace(4.0, 5.0, 30)
    signals = np.exp(-(devs[:, None] ** 2)) * np.exp(
        1j * devs[:, None] * freqs[None, :] / 10
    )
    # Synthetic run input is fixture setup; starting analysis, mounting, RPC,
    # result publication and writeback must all use their production seams.
    run_result = (
        RunRecord(cfg=None, result=FluxDepResult(devs, freqs, signals))
        if adapter == "onetone/flux_dep"
        else RunRecord(cfg=None, result=FreqFluxResult(devs, freqs, signals))
    )
    if native_source is None:
        parameters = fx.ctrl.context_control.read_context()
        fx.state.update_tab_result(tab_id, run_result, parameters=parameters)
    else:
        assert isinstance(run_result.result, FluxDepResult)
        load_native_onetone(fx, tab_id, native_source, run_result.result)
    tab_widget = window._tab_widgets[tab_id]  # pyright: ignore[reportPrivateUsage] - test fixture locates the mounted presentation
    if pane is not None:
        tab_widget.select_pane(pane)
    token = fx.ctrl.run_analyze_control.analyze(
        tab_id, FluxPickParams() if analyze_params is None else analyze_params
    )
    widget = tab_widget.interactive_frontend()
    assert isinstance(widget, FluxPickFrontend)
    return tab_id, token, widget


def start_interactive(fx, *, background=None):
    tab_id = fx.ctrl.new_tab("fake")
    devs = np.linspace(-5.0, 5.0, 60)
    freqs = np.linspace(4.0, 5.0, 30)
    signals = np.exp(-(devs[:, None] ** 2)) * np.ones((1, 30))
    plots = Plots(NonPresentingHost())
    analyze_params = FluxPickParams()
    plugin = make_flux_pick_plugin(
        AnalyzeRequest(
            run_result=SimpleNamespace(signals=signals, values=devs, freqs=freqs),
            analyze_params=analyze_params,
            parameters=None,
            predictor=None,
        ),
        force_magnitude=True,
        plots=plots,
        result_builder=render_flux_pick,
    )
    plugin.bind_background(background or fx.ctrl.run_background)
    # Set up a real AnalyzeService operation; RPC interactions below always go
    # through the shipped socket and RunAnalyzeControlFacet, not a handler stub.
    token = fx.ctrl._analyze_svc.start_plugin(
        tab_id,
        plugin,
        QtOwnerScheduler(),
        analyze_params_instance=analyze_params,
        plots=plots,
    )
    fx.view.interactive_presentation.return_value = None
    return tab_id, token, plugin


def rpc(sock: socket.socket, method: str, params: dict[str, Any]) -> dict[str, Any]:
    rid = uuid4().hex
    send(sock, {"id": rid, "method": method, "params": params})
    return recv_response(sock, rid)


def interact(
    sock: socket.socket, tab_id: str, payload=None, **params
) -> dict[str, Any]:
    args = {"tab_id": tab_id, **params}
    if payload is not None:
        args["payload"] = payload
    return rpc(sock, "tab.interact", args)


def button(widget: FluxPickFrontend, name: str) -> QPushButton:
    return next(
        button for button in widget.findChildren(QPushButton) if button.text() == name
    )


def pointer(canvas: FigureCanvasQTAgg, name: str, x: float, y: float = 4.5) -> None:
    px, py = canvas.figure.axes[0].transData.transform((x, y))
    event = MouseEvent(name, canvas, int(px), int(py), button=MouseButton.LEFT)
    canvas.callbacks.process(name, event)
