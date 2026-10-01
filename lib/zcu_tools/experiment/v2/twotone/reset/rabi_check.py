from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    IDENTITY,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
    config,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runtime import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Branch,
    Module,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    SweepCfg,
    sweep2param,
)
from zcu_tools.utils.process import find_rotate_angle

from .rabi_check_fit import RabiCheckFit, fit_reset_rabi


@dataclass(frozen=True)
class RabiCheckResult:
    gains: NDArray[np.float64]
    signals: NDArray[np.complex128]
    reset_states: NDArray[np.int64] = field(
        default_factory=lambda: np.array([0, 1, 2], dtype=np.int64)
    )


def reset_rabi_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    # Anchor all branches to the before-reset IQ axis. Branch offsets must not
    # rotate the projection or independently rescale the measured contrasts.
    try:
        angle = find_rotate_angle(signals[0])
    except ValueError:
        # Live buffers can be empty before the first acquisition completes.
        angle = 0.0
    return (signals * np.exp(-1j * angle)).real


class RabiCheckModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    rabi_pulse: PulseCfg
    tested_reset: ResetCfg
    readout: ReadoutCfg


class RabiCheckSweepCfg(ConfigBase):
    gain: SweepCfg


class RabiCheckCfg(ProgramV2Cfg, ExpCfgModel):
    modules: RabiCheckModuleCfg
    sweep: RabiCheckSweepCfg


def _rabi_check_sequence(
    modules: RabiCheckModuleCfg, gain_sweep: SweepCfg
) -> tuple[Module, ...]:
    # Both Pulse instances copy this swept cfg, so the post-reset drive follows
    # the same gain axis while keeping a distinct module name in diagnostics.
    modules.rabi_pulse.set_param("gain", sweep2param("gain", gain_sweep))
    return (
        Reset("reset", modules.reset),
        Pulse("rabi_pulse", modules.rabi_pulse),
        Branch(
            "reset_sel",
            [],
            Reset("tested_reset_1", modules.tested_reset),
            [
                Reset("tested_reset_2", modules.tested_reset),
                Pulse("rabi_pulse_after_reset", modules.rabi_pulse),
            ],
        ),
        Readout("readout", modules.readout),
    )


class RabiCheckExp(PersistableExperiment[RabiCheckResult, RabiCheckCfg]):
    AXES_SPEC = AxesSpec(
        axes=(
            Axis("gains", "Amplitude", "a.u.", scale=IDENTITY, dtype=np.float64),
            Axis("reset_states", "Reset", "None", scale=IDENTITY, dtype=np.int64),
        ),
        z=ZSpec("signals", "Signal", "a.u.", dtype=np.complex128),
        result_type=RabiCheckResult,
        cfg_type=RabiCheckCfg,
        tag="twotone/reset/rabi_check",
    )

    def run(self, cfg: RabiCheckCfg, *, context: RunContext) -> RabiCheckResult:
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg
        setup_devices(
            cfg,
            context.devices,
            progress=True,
            cancel_signal=context.cancel_signal.event,
        )
        modules = cfg.modules

        gains = sweep2array(
            cfg.sweep.gain,
            "gain",
            {"soccfg": soccfg, "gen_ch": modules.rabi_pulse.ch},
        )

        viewer = context.plots.liveplot_1d(
            "measurement", "Pulse gain", "Amplitude", num_lines=3
        )
        signals_buffer = SignalBuffer(
            (3, len(gains)),
            on_update=lambda data: viewer.update(gains, reset_rabi_signal2real(data)),
        )
        with Schedule(cfg, signals_buffer, stop=context.cancel_signal) as sched:
            _ = (
                sched.prog_builder(soc, soccfg)
                .add(*_rabi_check_sequence(sched.cfg.modules, sched.cfg.sweep.gain))
                .declare_sweep("reset_sel", 3)
                .declare_sweep("gain", sched.cfg.sweep.gain)
                .build_and_acquire()
            )

        return RabiCheckResult(gains, signals_buffer.array)

    def analyze(
        self,
        source: RunRecord[RabiCheckCfg, RabiCheckResult],
        options: None,
        *,
        plots: Plots,
    ) -> RabiCheckFit:
        """Return descriptive contrast fits and publish fit, without reset fidelity.

        Frequency is fitted only to the before-reset branch. All branches share
        its IQ projection and frequency; after-reset also includes a 2f term.
        """
        del options
        result = source.result
        gains, signals = result.gains, result.signals
        if gains.ndim != 1 or signals.shape != (3, gains.size):
            raise ValueError("Reset-check signals must have shape (3, number of gains)")
        if not np.array_equal(result.reset_states, [0, 1, 2]):
            raise ValueError("Reset-check branch order must be [0, 1, 2]")
        if np.any(np.isinf(signals)):
            raise ValueError("Reset-check signals must not contain infinities")
        real_signals = reset_rabi_signal2real(signals)
        fit = fit_reset_rabi(gains, real_signals)
        fig, _ = plots.subplots(
            "fit",
            nrows=2,
            ncols=1,
            sharex=True,
            figsize=(max(config.figsize[0], 9), max(config.figsize[1], 6)),
            gridspec_kw={"height_ratios": [3, 1]},
            layout="constrained",
        )
        ax, residual_ax = fig.axes
        dense_gains = np.linspace(np.min(gains), np.max(gains), 600)
        branches = (fit.before, fit.reset, fit.after)
        labels = ("Before reset", "Reset only", "Reset + Rabi")
        for index, (branch, label) in enumerate(zip(branches, labels, strict=True)):
            color = f"C{index}"
            ax.plot(gains, real_signals[index], ".", color=color, label=label)
            ax.plot(
                dense_gains, branch.evaluate(dense_gains, fit.frequency), color=color
            )
            residual_ax.plot(
                gains,
                real_signals[index] - branch.evaluate(gains, fit.frequency),
                ".",
                color=color,
                label=f"{label}: RMS={branch.residual_rms:.3g}",
            )
        ax.set_title(
            f"Reset Rabi check | f = {fit.frequency:.5g} cycles/gain\n"
            f"A before = {fit.before.amplitude:.4g}, A after = {fit.after.amplitude:.4g}, "
            f"relative contrast = {fit.contrast_ratio:.4g}\n"
            f"Reset residual amplitude = {fit.reset.amplitude:.3g}, "
            f"after 2f amplitude = {fit.after.second_harmonic_amplitude:.3g}",
            fontsize=11,
        )
        ax.set_ylabel("Projected IQ (a.u.)")
        residual_ax.set_xlabel("Pulse gain (a.u.)")
        residual_ax.set_ylabel("Residual (a.u.)")
        residual_ax.axhline(0, color="gray", linewidth=0.8)
        for axes in (ax, residual_ax):
            axes.legend(fontsize=9)
            axes.grid(True)
        return fit
