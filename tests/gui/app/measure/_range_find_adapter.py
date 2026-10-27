"""Caller-owned spectrum and labels for range-find framework contracts."""

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from qtpy.QtWidgets import QApplication
from zcu_tools.analysis.dips import DipFinderOptions
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import AxesSpec
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.metadata import RunMetadata, RunSnapshot, SoftwareProvenance
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    AnalyzeRequest,
    Interactive,
    SessionEnv,
)
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.range_find_frontend import (
    draw_range_find,
    make_range_find_frontend,
)
from zcu_tools.gui.app.measure.range_find_plugin import RangeFindPlugin, RangeFindState
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
)
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.plotting.plots import Plots

from tests.gui._dialog_fakes import RecordingDialogPresenter


class SpectrumCfg(ConfigBase):
    samples: int = 101


@dataclass
class PickerParams:
    pass


@dataclass
class SpectrumData:
    freqs: NDArray[np.float64]
    magnitude: NDArray[np.float64]


def spectrum() -> SpectrumData:
    freqs = np.arange(101, dtype=np.float64) + 5000
    magnitude = 1 + 0.001 * np.sin(np.arange(101, dtype=np.float64))
    magnitude[20], magnitude[70] = 0.3, 0.5
    return SpectrumData(freqs, magnitude)


def make_form(context: FormContext) -> Form[SpectrumCfg]:
    return Form(SpectrumCfg, {})


def run(
    cfg: SpectrumCfg, *, context: RunContext
) -> RunRecord[SpectrumCfg, SpectrumData]:
    return RunRecord(cfg, spectrum())


def make_plugin(
    req: AnalyzeRequest[RunRecord[SpectrumCfg, SpectrumData], PickerParams],
    *,
    plots: Plots,
    options: DipFinderOptions | None = None,
    components: tuple[str, ...] = ("west", "east"),
) -> RangeFindPlugin[RangeFindState]:
    data = req.run_result.result

    def build(state: RangeFindState) -> RangeFindState:
        plots.subplots("fit")
        plots.draw(
            "fit",
            lambda figure: draw_range_find(figure, data.freqs, data.magnitude, state),
        )
        return state

    return RangeFindPlugin(
        data.freqs,
        data.magnitude,
        components=components,
        options=options if options is not None else DipFinderOptions(max_dips=2),
        build_result=build,
    )


RangeAdapter = Adapter(
    spec=AxesSpec(
        axes=(AxesSpec.Axis("freqs", "Frequency", "Hz", scale=1e6),),
        z=AxesSpec.Z("magnitude", "Magnitude", "a.u."),
        result_type=SpectrumData,
        cfg_type=SpectrumCfg,
        tag="synthetic/spectrum",
    ),
    make_form=make_form,
    run=run,
    guide=None,
    analysis=Interactive(
        plugin=make_plugin,
        frontend=make_range_find_frontend,
        options=lambda source, env: PickerParams(),
    ),
)


def _spectrum_record(
    parameters: ParameterContext,
    data: SpectrumData | None = None,
) -> RunRecord[SpectrumCfg, SpectrumData]:
    storage = parameters.capture_storage()
    metadata = RunMetadata(
        run_id="20261027T000000Z-a1b2c3",
        experiment=RangeAdapter.spec.tag,
        started_at="2026-10-27T00:00:00Z",
        finished_at="2026-10-27T00:01:00Z",
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
    return RunRecord(
        SpectrumCfg(), spectrum() if data is None else data, metadata=metadata
    )


@dataclass
class RangeApplication:
    state: State
    ctrl: Controller
    window: MainWindow
    tab_id: str


@contextmanager
def range_application(
    tmp_path: Path,
    qapp: QApplication,
    *,
    options: DipFinderOptions | None = None,
    data: SpectrumData | None = None,
    components: tuple[str, ...] = ("west", "east"),
) -> Generator[RangeApplication]:
    registry = Registry()
    adapter = Adapter(
        spec=RangeAdapter.spec,
        make_form=make_form,
        run=run,
        guide=None,
        analysis=Interactive(
            plugin=partial(make_plugin, options=options, components=components),
            frontend=make_range_find_frontend,
            options=lambda source, env: PickerParams(),
        ),
    )
    registry.register("spectrum", adapter)
    state = State(SessionEnv(parameters=None, soc=None, soccfg=None))
    ctrl = Controller(state, registry, None, BaseEventBus(), project_root=str(tmp_path))
    ctrl.setup_control.new_entry(EntryRequest("synthetic"))
    ctrl.setup_control.new_point(PointCreateRequest("test", "", CommandOrigin("gui")))
    window = MainWindow(
        ctrl, templates=lambda: {}, dialog_presenter=RecordingDialogPresenter()
    )
    ctrl.add_view(window)
    tab_id = ctrl.tab_control.new_tab("spectrum")
    parameters = ctrl.context_control.read_context()
    state.update_tab_result(
        tab_id, _spectrum_record(parameters, data), parameters=parameters
    )
    state.update_tab_analyze_param_instance(tab_id, PickerParams())
    window.refresh_tab_analyze_form(tab_id)
    window.refresh_tab_interaction(tab_id)
    window.show()
    qapp.processEvents()
    try:
        yield RangeApplication(state, ctrl, window, tab_id)
    finally:
        ctrl.run_analyze_control.cancel_analyze(tab_id)
        ctrl.tab_control.close_tab(tab_id)
        ctrl.close()
        window.deleteLater()
        qapp.processEvents()
