"""JPA adapters keep selected records, numerical outputs, and figures separate."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.jpa.jpa_check import CheckResult
from zcu_tools.experiment.v2.jpa.jpa_flux import FluxResult
from zcu_tools.experiment.v2.jpa.jpa_freq import FreqResult
from zcu_tools.experiment.v2.jpa.jpa_power import PowerResult
from zcu_tools.experiment.v2_gui.measure.adapters.jpa.check import JpaCheckAdapter
from zcu_tools.experiment.v2_gui.measure.adapters.jpa.flux import JpaFluxAdapter
from zcu_tools.experiment.v2_gui.measure.adapters.jpa.freq import JpaFreqAdapter
from zcu_tools.experiment.v2_gui.measure.adapters.jpa.power import JpaPowerAdapter
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    MetaDictWriteback,
    NoAnalyzeParams,
    SessionEnv,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.lowering import make_sweep_range
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.resources.context import MetaDict, ModuleLibrary


@pytest.fixture
def plot_factory():
    sessions = []

    def make():
        plots = Plots(NonPresentingHost())
        sessions.append(plots)
        return plots

    yield make
    for plots in sessions:
        plots.finish(present=False)
        plots.release()


@pytest.fixture
def raw_cfg():
    pulse = {
        "type": "pulse",
        "ch": 1,
        "nqz": 1,
        "freq": 6000.0,
        "gain": 0.1,
        "waveform": {"style": "const", "length": 1.0},
    }
    return {
        "modules": {
            "pi_pulse": pulse,
            "readout": {
                "type": "readout/pulse",
                "pulse_cfg": pulse,
                "ro_cfg": {
                    "ro_ch": 0,
                    "ro_freq": 6000.0,
                    "ro_length": 1.0,
                    "trig_offset": 0.1,
                },
            },
        },
        "dev": {},
        "reps": 10,
        "rounds": 1,
        "relax_delay": 5.0,
    }


@pytest.fixture(params=["freq", "flux", "power"])
def calibration_case(request):
    signals = np.array([1.0, 1.0, 9.0, 1.0, 1.0])
    return {
        "freq": (
            JpaFreqAdapter(),
            FreqResult(np.linspace(12000.0, 12004.0, 5), signals),
            "jpa_freq",
            12000.0,
            12004.0,
            {"best_freq": 12002.0},
        ),
        "flux": (
            JpaFluxAdapter(),
            FluxResult(np.linspace(-0.02, 0.02, 5), signals),
            "jpa_flux",
            -0.02,
            0.02,
            {"best_flux": 0.0},
        ),
        "power": (
            JpaPowerAdapter(),
            PowerResult(np.linspace(-20.0, -16.0, 5), signals),
            "jpa_power",
            -20.0,
            -16.0,
            {"best_power": -18.0},
        ),
    }[request.param]


@pytest.mark.parametrize("with_cfg", [False, True])
def test_selected_record_drives_calibration_and_scalar_writeback(
    calibration_case, raw_cfg, plot_factory, with_cfg
):
    adapter, data, sweep_name, start, stop, expected = calibration_case
    raw_cfg["sweep"] = {sweep_name: make_sweep_range(start, stop, expts=5)}
    cfg = adapter.ExpCfg_cls.model_validate(raw_cfg)
    source = RunRecord(cfg if with_cfg else None, data)
    other = RunRecord(None, replace(data, signals=np.roll(data.signals, 2)))
    ctx = SessionEnv(MetaDict(), ModuleLibrary(), None, None)
    params = NoAnalyzeParams()
    other_plots = plot_factory()
    other_answer = adapter.analyze(
        AnalyzeRequest(other, params, ctx.md, ctx.ml, None), plots=other_plots
    )
    plots = plot_factory()
    answer = adapter.analyze(
        AnalyzeRequest(source, params, ctx.md, ctx.ml, None), plots=plots
    )

    assert answer.to_summary_dict() == expected
    assert other_answer.to_summary_dict() != expected
    assert tuple(plots) == ("fit",)
    assert plots["fit"] is not other_plots["fit"]
    peak_marker = plots["fit"].axes[0].lines[-1]
    np.testing.assert_allclose(peak_marker.get_xdata(), next(iter(expected.values())))
    items = adapter.get_writeback_items(WritebackRequest(source, answer, ctx))
    assert len(items) == 1
    assert isinstance(items[0], MetaDictWriteback)
    assert {items[0].target_name: items[0].proposed_value} == {
        key.replace("best_", "best_jpa_"): value for key, value in expected.items()
    }


@pytest.mark.parametrize("with_cfg", [False, True])
def test_check_renders_selected_off_on_traces_without_numeric_output(
    raw_cfg, plot_factory, with_cfg
):
    adapter = JpaCheckAdapter()
    del raw_cfg["modules"]["pi_pulse"]
    raw_cfg["sweep"] = {"freq": make_sweep_range(6000.0, 6002.0, expts=3)}
    cfg = adapter.ExpCfg_cls.model_validate(raw_cfg)
    data = CheckResult(
        np.array([0.0, 1.0]),
        np.array([6000.0, 6001.0, 6002.0]),
        np.array([[1j, 2j, 3j], [4j, 5j, 6j]]),
    )
    source = RunRecord(cfg if with_cfg else None, data)
    other = RunRecord(None, replace(data, signals=10 * data.signals))
    ctx = SessionEnv(MetaDict(), ModuleLibrary(), None, None)
    params = NoAnalyzeParams()
    other_plots = plot_factory()
    adapter.analyze(
        AnalyzeRequest(other, params, ctx.md, ctx.ml, None), plots=other_plots
    )
    plots = plot_factory()
    answer = adapter.analyze(
        AnalyzeRequest(source, params, ctx.md, ctx.ml, None), plots=plots
    )

    assert answer.to_summary_dict() == {}
    assert tuple(plots) == ("fit",)
    assert plots["fit"] is not other_plots["fit"]
    for line, signal in zip(plots["fit"].axes[0].lines, data.signals, strict=True):
        np.testing.assert_allclose(line.get_xdata(), data.freqs)
        np.testing.assert_allclose(line.get_ydata(), np.abs(signal))
