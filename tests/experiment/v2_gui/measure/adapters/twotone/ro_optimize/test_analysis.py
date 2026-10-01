"""Readout adapters bind analysis and writeback to the selected source."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.twotone.ro_optimize.freq import FreqResult
from zcu_tools.experiment.v2.twotone.ro_optimize.freq_gain import FreqGainResult
from zcu_tools.experiment.v2.twotone.ro_optimize.length import LengthResult
from zcu_tools.experiment.v2.twotone.ro_optimize.power import PowerResult
from zcu_tools.experiment.v2_gui.measure.adapters.twotone.ro_optimize.freq import (
    RoOptFreqAdapter,
    RoOptFreqAnalyzeParams,
)
from zcu_tools.experiment.v2_gui.measure.adapters.twotone.ro_optimize.freq_gain import (
    RoOptFreqGainAdapter,
    RoOptFreqGainAnalyzeParams,
)
from zcu_tools.experiment.v2_gui.measure.adapters.twotone.ro_optimize.length import (
    RoOptLengthAdapter,
    RoOptLengthAnalyzeParams,
)
from zcu_tools.experiment.v2_gui.measure.adapters.twotone.ro_optimize.power import (
    RoOptPowerAdapter,
    RoOptPowerAnalyzeParams,
)
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    MetaDictWriteback,
    ModuleWriteback,
    SessionEnv,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.lowering import (
    make_sweep_range,
    schema_to_raw_dict,
)
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.program.v2 import PulseReadoutCfg
from zcu_tools.resources.context import MetaDict, ModuleLibrary


@pytest.fixture(params=["freq", "freq_gain", "length", "power"])
def readout_case(request):
    freqs = np.array([6000.0, 6010.0, 6020.0])
    gains = np.array([0.1, 0.2, 0.3])
    snrs = np.array([1.0, 5.0, 1.0])
    cases = {
        "freq": (
            RoOptFreqAdapter(),
            RoOptFreqAnalyzeParams(smooth=0.05, smooth_method="gaussian"),
            FreqResult(freqs, snrs),
            {"freq": make_sweep_range(6000.0, 6020.0, expts=3)},
            {"best_freq": 6010.0},
        ),
        "freq_gain": (
            RoOptFreqGainAdapter(),
            RoOptFreqGainAnalyzeParams(smooth=0.05, smooth_method="gaussian"),
            FreqGainResult(freqs, gains, np.outer(snrs, snrs)),
            {
                "freq": make_sweep_range(6000.0, 6020.0, expts=3),
                "gain": make_sweep_range(0.1, 0.3, expts=3),
            },
            {"best_freq": 6010.0, "best_gain": 0.2},
        ),
        "length": (
            RoOptLengthAdapter(),
            RoOptLengthAnalyzeParams(
                duration_t0=2.0, smooth=0.05, smooth_method="gaussian"
            ),
            LengthResult(np.array([1.0, 2.0, 3.0]), np.array([4.5, 5.0, 1.0])),
            {"length": make_sweep_range(1.0, 3.0, expts=3)},
            {"best_length": 1.0},
        ),
        "power": (
            RoOptPowerAdapter(),
            RoOptPowerAnalyzeParams(
                penalty_ratio=0.5, smooth=0.05, smooth_method="gaussian"
            ),
            PowerResult(gains, np.array([4.8, 5.0, 1.0])),
            {"gain": make_sweep_range(0.1, 0.3, expts=3)},
            {"best_gain": 0.1},
        ),
    }
    return cases[request.param]


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


@pytest.mark.parametrize("with_cfg", [False, True])
def test_selected_source_drives_fit_and_readout_writeback(
    readout_case, plot_factory, with_cfg
):
    adapter, params, data, sweep, expected = readout_case
    ctx = SessionEnv(MetaDict(), ModuleLibrary(), None, None)
    ctx.md.best_ro_freq = 6100.0
    ctx.md.best_ro_gain = 0.6
    ctx.md.best_ro_length = 4.0
    pulse = {
        "type": "pulse",
        "ch": 1,
        "nqz": 1,
        "freq": 6000.0,
        "gain": 0.1,
        "waveform": {"style": "const", "length": 1.0},
    }
    cfg = adapter.ExpCfg_cls.model_validate(
        {
            "modules": {
                "qub_pulse": pulse,
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
            "sweep": sweep,
            "reps": 10,
            "rounds": 1,
            "relax_delay": 5.0,
        }
    )
    source = RunRecord(cfg if with_cfg else None, data)
    other = RunRecord(None, replace(data, signals=np.roll(data.signals, 1, axis=0)))
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
    assert plots["fit"].axes[0].get_xlabel()

    items = adapter.get_writeback_items(WritebackRequest(source, answer, ctx))
    scalars = {
        item.target_name: item.proposed_value
        for item in items
        if isinstance(item, MetaDictWriteback)
    }
    assert scalars == {
        key.replace("best_", "best_ro_"): value for key, value in expected.items()
    }
    modules = [item for item in items if isinstance(item, ModuleWriteback)]
    if not with_cfg:
        assert modules == []
        return
    assert len(modules) == 1
    module = modules[0]
    assert module.target_name == "readout_dpm"
    assert module.role_id == "readout_dpm"
    assert module.edit_schema is not None
    proposed = PulseReadoutCfg.model_validate(
        schema_to_raw_dict(module.edit_schema, ctx.md, ctx.ml)
    )
    assert proposed.pulse_cfg.ch == 1
    assert proposed.ro_cfg.ro_ch == 0
    assert proposed.pulse_cfg.freq == expected.get("best_freq", 6100.0)
    assert proposed.ro_cfg.ro_freq == expected.get("best_freq", 6100.0)
    assert proposed.pulse_cfg.gain == expected.get("best_gain", 0.6)
    assert proposed.ro_cfg.ro_length == expected.get("best_length", 4.0)
    assert source.cfg is not None
    assert source.cfg.modules.readout.pulse_cfg.freq == 6000.0
