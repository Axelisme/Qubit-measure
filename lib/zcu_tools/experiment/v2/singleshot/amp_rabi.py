from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
    record_result,
    retrieve_result,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runner import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.program.v2 import (
    ProgramV2Cfg,
    PulseCfg,
    ReadoutCfg,
    ResetCfg,
    SweepCfg,
    sweep2param,
)
from zcu_tools.utils.fitting import fit_rabi

from .util import (
    calc_populations,
    classify_result,
    correct_populations,
    raw_population_signal,
)


@dataclass(frozen=True)
class AmpRabiResult:
    gains: NDArray[np.float64]
    signals: NDArray[np.float64]
    population_states: NDArray[np.int64] = field(
        default_factory=lambda: np.array([0, 1], dtype=np.int64)
    )
    cfg_snapshot: AmpRabiCfg | None = None


@dataclass(frozen=True)
class AmpRabiFit:
    pi_gain: float
    pi_gain_error: float
    pi2_gain: float
    pi2_gain_error: float
    frequency: float
    amplitude: float


class AmpRabiSweepCfg(ConfigBase):
    gain: SweepCfg


class AmpRabiModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    qub_pulse: PulseCfg
    readout: ReadoutCfg


class AmpRabiCfg(ProgramV2Cfg, ExpCfgModel):
    modules: AmpRabiModuleCfg
    sweep: AmpRabiSweepCfg


class AmpRabiExp(PersistableExperiment[AmpRabiResult, AmpRabiCfg]):
    AXES_SPEC = AxesSpec(
        axes=(
            Axis("population_states", "GE Population", "None", dtype=np.int64),
            Axis("gains", "Gain", "a.u.", dtype=np.float64),
        ),
        z=ZSpec("signals", "Population", "a.u.", dtype=np.float64),
        result_type=AmpRabiResult,
        cfg_type=AmpRabiCfg,
        tag="singleshot/amp_rabi",
    )

    @record_result
    def run(
        self,
        soc,
        soccfg,
        cfg: AmpRabiCfg,
        g_center: complex,
        e_center: complex,
        radius: float,
    ) -> AmpRabiResult:
        classify_result(np.empty(0, dtype=np.complex128), g_center, e_center, radius)
        cfg = deepcopy(cfg)
        setup_devices(cfg, progress=True)
        gains = sweep2array(
            cfg.sweep.gain,
            "gain",
            {"soccfg": soccfg, "gen_ch": cfg.modules.qub_pulse.ch},
        )
        with LivePlot1D(
            "Pulse gain (a.u.)",
            "Classified population",
            segment_kwargs={
                "num_lines": 3,
                "line_kwargs": [
                    {"label": label, "color": color}
                    for label, color in (
                        ("Ground", "blue"),
                        ("Excited", "red"),
                        ("Other", "green"),
                    )
                ],
            },
        ) as viewer:
            viewer.get_ax().set_ylim(-0.02, 1.02)
            buffer = SignalBuffer(
                (gains.size, 2),
                dtype=np.float64,
                on_update=lambda data: viewer.update(gains, calc_populations(data).T),
            )
            with Schedule(cfg, buffer) as sched:
                modules = sched.cfg.modules
                modules.qub_pulse.set_param(
                    "gain", sweep2param("gain", sched.cfg.sweep.gain)
                )
                (
                    sched.prog_builder(soc, soccfg)
                    .add_reset("reset", modules.reset)
                    .add_pulse("qubit_pulse", modules.qub_pulse)
                    .add_readout("readout", modules.readout)
                    .declare_sweep("gain", sched.cfg.sweep.gain)
                    .build_and_acquire(
                        raw2signal_fn=raw_population_signal,
                        g_center=g_center,
                        e_center=e_center,
                        ge_radius=radius,
                    )
                )
                sched.trigger_update(flush=True)
        return AmpRabiResult(gains=gains, signals=buffer.array, cfg_snapshot=cfg)

    @retrieve_result
    def analyze(
        self,
        result: AmpRabiResult | None = None,
        *,
        confusion_matrix: NDArray[np.float64] | None = None,
    ) -> tuple[AmpRabiFit, Figure]:
        if result is None:
            raise ValueError("No amp Rabi result found")
        if result.signals.shape != (result.gains.size, 2) or not np.array_equal(
            result.population_states, [0, 1]
        ):
            raise ValueError("Amp Rabi requires (gain, G/E) populations")
        populations = correct_populations(
            calc_populations(result.signals), confusion_matrix
        )
        valid = np.isfinite(result.gains) & np.isfinite(populations).all(axis=-1)
        order = np.flatnonzero(valid)[np.argsort(result.gains[valid])]
        gains, ground = result.gains[order], populations[order, 0]
        if (
            gains.size < 5
            or np.unique(gains).size != gains.size
            or np.ptp(ground) < 1e-8
        ):
            raise ValueError(
                "Amp Rabi fitting requires at least five distinct gains with population contrast"
            )
        pi_gain, pi_err, pi2_gain, pi2_err, freq, _, fitted, (params, _) = fit_rabi(
            gains, ground, decay=False
        )
        fit = AmpRabiFit(pi_gain, pi_err, pi2_gain, pi2_err, freq, abs(params[1]))
        fig, ax = plt.subplots()
        plot_order = np.argsort(result.gains)
        for state, (label, color) in enumerate(
            (("Ground", "blue"), ("Excited", "red"), ("Other", "green"))
        ):
            ax.plot(
                result.gains[plot_order],
                populations[plot_order, state],
                ".-",
                color=color,
                label=label,
            )
        ax.plot(gains, fitted, "--", color="black", label="Ground Rabi fit")
        ax.set(
            xlabel="Pulse gain (a.u.)",
            ylabel="Population"
            if confusion_matrix is not None
            else "Classified population",
            ylim=(-0.02, 1.02),
            title=f"Pi gain = {pi_gain:.5g}; amplitude = {fit.amplitude:.3g}\nOther is not calibrated leakage",
        )
        ax.legend()
        ax.grid(True)
        return fit, fig
