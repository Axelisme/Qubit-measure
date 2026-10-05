"""freq experiment core."""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.fitting.resonance.hanger import HangerModel
from zcu_tools.analysis.fitting.resonance.transmission import TransmissionModel
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment.axes_spec import MHZ_TO_HZ, AxesSpec, Axis, ZSpec
from zcu_tools.experiment.base import PersistableExperiment
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import ProgramV2Cfg, ReadoutCfg
from zcu_tools.program.v2.sweep import SweepCfg

from zcu_lab.v2.onetone.freq.core import (
    FreqAnalysis,
    FreqAnalyzeOptions,
    FreqCfg,
    FreqExp,
    FreqResult,
)


class FakeFreqSweepCfg(ProgramV2Cfg):
    freq: SweepCfg


@dataclass(frozen=True)
class HangerSimParams:
    """Ground-truth params for a HangerModel lineshape (hanger / notch)."""

    freq: float = 6000.0
    Ql: float = 5000.0
    Qc_abs: float = 6000.0
    phi: float = 0.0
    a0_abs: float = 1.0
    edelay: float = 0.05
    noise_scale: float = 0.05


@dataclass(frozen=True)
class TransmissionSimParams:
    """Ground-truth params for a TransmissionModel lineshape (no Qc / phi)."""

    freq: float = 6000.0
    Ql: float = 5000.0
    a0_abs: float = 1.0
    edelay: float = 0.05
    noise_scale: float = 0.05


Param: TypeAlias = "HangerSimParams | TransmissionSimParams"


class FakeFreqModuleCfg(ConfigBase):
    # Mirrors the real onetone ExpCfg modules: readout only. No init_pulse (no
    # qubit-drive pulse) and no reset (one-tone runs without a qubit reset).
    readout: ReadoutCfg


class FakeFreqCfg(ProgramV2Cfg, ExpCfgModel):
    sweep: FakeFreqSweepCfg
    modules: FakeFreqModuleCfg
    fast_mode: bool = False  # skip per-point sleep; set True in tests


FakeFreqRunResult: TypeAlias = RunRecord[FakeFreqCfg, FreqResult]


class FakeFreqExp(PersistableExperiment[FreqResult, FakeFreqCfg]):
    """Simulated FreqExp: same run/analyze/save interface, no hardware required.

    The ground-truth resonance (``model_type`` + ``params``) is supplied at
    construction, NOT carried in the cfg — so the cfg's sweep is set
    independently and the analysis must genuinely find the dip.
    """

    AXES_SPEC = AxesSpec(
        axes=(Axis("freqs", "Frequency", "Hz", scale=MHZ_TO_HZ),),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=FreqResult,
        cfg_type=FakeFreqCfg,
        tag="fake/freq",
    )

    def __init__(self, model_type: Literal["t", "hm"], params: Param) -> None:
        self._model_type = model_type
        self._params = params

    def _clean_signals(self, freqs: NDArray[np.float64]) -> NDArray[np.complex128]:
        p = self._params
        a0 = complex(p.a0_abs)
        if self._model_type == "hm":
            assert isinstance(p, HangerSimParams)
            Qc = complex(p.Qc_abs * np.exp(-1j * p.phi))
            return HangerModel.calc_signals(
                freqs, p.freq, p.Ql, Qc, p.phi, a0, p.edelay
            )
        assert isinstance(p, TransmissionSimParams)
        return TransmissionModel.calc_signals(freqs, p.freq, p.Ql, a0, p.edelay)

    def run(self, config: FakeFreqCfg, *, context: RunContext) -> FreqResult:
        cfg = deepcopy(config)
        sweep = cfg.sweep.freq
        freqs = np.linspace(sweep.start, sweep.stop, sweep.expts)

        clean = self._clean_signals(freqs)
        sigma = self._params.noise_scale / np.sqrt(cfg.reps * cfg.rounds)
        rng = np.random.default_rng()

        viewer = context.plots.liveplot_1d(
            "measurement", "Frequency (MHz)", "Amplitude"
        )
        signals_buffer = SignalBuffer(
            (len(freqs),),
            on_update=lambda data: viewer.update(freqs, np.abs(data)),
        )
        with Schedule(cfg, signals_buffer, stop=context.cancel_signal) as sched:
            accumulated = np.zeros(len(freqs), dtype=np.complex128)
            rounds_done = 0
            for _round_idx, _step in sched.repeat("round", cfg.rounds):
                noise = rng.normal(0, sigma * np.sqrt(cfg.rounds), len(freqs))
                noise_i = rng.normal(0, sigma * np.sqrt(cfg.rounds), len(freqs))
                accumulated += clean + noise + 1j * noise_i
                rounds_done += 1
                if not cfg.fast_mode:
                    for _ in range(len(freqs)):
                        time.sleep(0.0005)
                signals_buffer.set(accumulated / rounds_done)
            if rounds_done == 0:
                signals_buffer.set(accumulated)
            signals_buffer.trigger_update(flush=True)
            signals = signals_buffer.array

        return FreqResult(freqs=freqs, signals=signals)

    @staticmethod
    def analyze(
        source: FakeFreqRunResult,
        options: FreqAnalyzeOptions,
        *,
        plots: Plots,
    ) -> FreqAnalysis:
        # Fitting needs only measured data, not acquisition cfg or simulation truth.
        fitting_source = RunRecord[FreqCfg, FreqResult](cfg=None, result=source.result)
        return FreqExp().analyze(fitting_source, options, plots=plots)
