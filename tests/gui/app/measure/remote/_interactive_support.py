"""Interactive analysis builders, socket requests and frontend input simulation."""

from __future__ import annotations

import socket
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from qtpy.QtWidgets import QPushButton
from zcu_tools.analysis.fluxdep.line_state import FluxPickInputs
from zcu_tools.experiment.metadata import RunMetadata, RunSnapshot, SoftwareProvenance
from zcu_tools.experiment.native_persistence import save_run
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter, Interactive
from zcu_tools.gui.app.measure.flux_pick_frontend import FluxPickFrontend
from zcu_tools.gui.app.measure.flux_pick_plugin import (
    make_flux_pick_plugin,
    render_flux_pick,
)
from zcu_tools.gui.app.measure.interactive_flux_pick import (
    FluxPickParams,
    FluxPickResult,
)
from zcu_tools.gui.app.measure.ui.exp_tab_widget import ExpTabWidget
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.program.v2.sweep import SweepCfg

from tests.gui.app.measure._flux_adapter import (
    SpectrumAdapter,
    SpectrumAdapter_make_interactive_frontend,
    SpectrumAdapter_make_interactive_plugin,
    SpectrumCfg,
    SpectrumCore,
    SpectrumResult,
)
from tests.gui.app.measure._record_support import require_record

from ._helpers import Fixture, recv_response, send


class DeferredBackground:
    def __init__(self) -> None:
        self.pending = []

    def run_background(self, compute, on_done, on_error) -> None:
        self.pending.append((compute, on_done, on_error))


class InteractiveFixture(Fixture):
    def __init__(self, *, project_root: str) -> None:
        super().__init__(active_label="interactive-test", project_root=project_root)
        self.widgets: list[FluxPickFrontend[FluxPickResult]] = []
        self.registry.register(
            "synthetic/interactive",
            Adapter(
                spec=SpectrumAdapter.spec,
                make_form=SpectrumAdapter.make_form,
                run=SpectrumAdapter.run,
                guide=None,
                analysis=Interactive(
                    plugin=lambda req, *, plots: (
                        SpectrumAdapter_make_interactive_plugin(
                            replace(req, parameters=None), plots=plots
                        )
                    ),
                    frontend=SpectrumAdapter_make_interactive_frontend,
                    options=lambda source, env: FluxPickParams(),
                ),
            ),
        )


def load_native_spectrum(
    fx: InteractiveFixture, tab_id: str, path: Path, result: SpectrumResult
) -> None:
    """Write and publicly load history captured from the current session source."""
    parameters = fx.ctrl.context_control.read_context()
    assert parameters is not None
    storage = parameters.capture_storage()
    cfg = SpectrumCfg()
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
    spec = SpectrumCore.AXES_SPEC
    metadata = RunMetadata(
        run_id="20261008T000000Z-a1b2c3",
        experiment=spec.tag,
        started_at="2026-10-08T00:00:00+00:00",
        finished_at="2026-10-08T00:01:00+00:00",
        completion="complete",
        snapshot=RunSnapshot(
            entry_id=storage.entry_id, point=storage.point, roles=storage.roles
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
    save_run(replace(RunRecord(cfg, result), metadata=metadata), path, spec=spec)
    fx.ctrl.run_analyze_control.load_tab_result(tab_id, str(path))
    assert require_record(fx.state.get_tab(tab_id).run).metadata == metadata


def start_mounted(
    fx: InteractiveFixture,
    window: MainWindow,
    adapter: str,
    *,
    analyze_params: FluxPickParams | None = None,
    pane: Literal["run", "data"] | None = None,
    native_source: Path | None = None,
):
    tab_id = fx.ctrl.tab_control.new_tab(adapter)
    devs = np.linspace(-5.0, 5.0, 60)
    freqs = np.linspace(4.0, 5.0, 30)
    signals = np.exp(-(devs[:, None] ** 2)) * np.exp(
        1j * devs[:, None] * freqs[None, :] / 10
    )
    # Synthetic run input is fixture setup; starting analysis, mounting, RPC,
    # result publication and writeback must all use their production seams.
    run_result = RunRecord(cfg=None, result=SpectrumResult(devs, freqs, signals))
    if native_source is None:
        parameters = fx.ctrl.context_control.read_context()
        fx.state.update_tab_result(tab_id, run_result, parameters=parameters)
    else:
        assert isinstance(run_result.result, SpectrumResult)
        load_native_spectrum(fx, tab_id, native_source, run_result.result)
    tab_widget = next(
        tab for tab in window.findChildren(ExpTabWidget) if tab.tab_id == tab_id
    )
    if pane is not None:
        tab_widget.select_pane(pane)
    token = fx.ctrl.run_analyze_control.analyze(
        tab_id, FluxPickParams() if analyze_params is None else analyze_params
    )
    widget = tab_widget.interactive_frontend()
    assert isinstance(widget, FluxPickFrontend)
    return tab_id, token, widget


def start_interactive(fx, *, background=None):
    tab_id = fx.ctrl.tab_control.new_tab("synthetic/interactive")
    devs = np.linspace(-5.0, 5.0, 60)
    freqs = np.linspace(4.0, 5.0, 30)
    signals = np.exp(-(devs[:, None] ** 2)) * np.ones((1, 30))
    fx.state.update_tab_result(
        tab_id,
        RunRecord(
            None, SpectrumResult(devs, freqs, np.asarray(signals, dtype=np.complex128))
        ),
        parameters=fx.ctrl.context_control.read_context(),
    )
    plots = Plots(NonPresentingHost())
    plugin = make_flux_pick_plugin(
        FluxPickInputs(np.asarray(signals, dtype=np.complex128), devs, freqs),
        force_magnitude=True,
        plots=plots,
        result_builder=render_flux_pick,
    )
    plugin.bind_background(background or fx.ctrl.run_background)
    # Keep the public AnalyzeService seam for explicit background injection.
    # Mounted host/facet pairing is exercised by start_mounted instead.
    token = fx.ctrl._analyze_svc.start_plugin(
        tab_id,
        plugin,
        QtOwnerScheduler(),
        analyze_params_instance=FluxPickParams(),
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


def button(widget: FluxPickFrontend[FluxPickResult], name: str) -> QPushButton:
    return next(
        button for button in widget.findChildren(QPushButton) if button.text() == name
    )


def pointer(canvas: FigureCanvasQTAgg, name: str, x: float, y: float = 4.5) -> None:
    px, py = canvas.figure.axes[0].transData.transform((x, y))
    event = MouseEvent(name, canvas, int(px), int(py), button=MouseButton.LEFT)
    canvas.callbacks.process(name, event)
