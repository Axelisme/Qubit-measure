from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import AxesSpec, Experiment, ProgramExpCfg, config
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import Pulse, Readout, Reset, SweepCfg, sweep2param


@dataclass(frozen=True)
class DriveFreqResult:
    """Probe gains, frequencies in MHz, and mean complex IQ [frequency, gain]."""

    gains: NDArray[np.float64]
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]


def drivefreq_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    avg_len = max(int(0.05 * signals.shape[1]), 1)

    mist_signals = signals - np.mean(signals[:, :avg_len])

    return np.abs(mist_signals)


class DriveFreqModuleCfg(ConfigBase):
    reset: Reset.Cfg | None = None
    init_pulse: Pulse.Cfg | None = None
    probe_pulse: Pulse.Cfg
    readout: Readout.Cfg


class DriveFreqSweepCfg(ConfigBase):
    freq: SweepCfg
    gain: SweepCfg


class DriveFreqCfg(ProgramExpCfg):
    modules: DriveFreqModuleCfg
    sweep: DriveFreqSweepCfg


class DriveFreqExp(Experiment[DriveFreqResult, DriveFreqCfg]):
    # inner freqs stores MHz on disk (disk Hz) -> scale=1e6; outer gains -> IDENTITY
    AXES_SPEC: ClassVar[AxesSpec[DriveFreqResult, DriveFreqCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("freqs", "Pulse frequency", "Hz", scale=1e6),
            AxesSpec.Axis("gains", "Pulse gain", "a.u."),
        ),
        z=AxesSpec.Z("signals", "Signal", "a.u."),
        result_type=DriveFreqResult,
        cfg_type=DriveFreqCfg,
        cfg_schema_version="2.0",
        tag="mist/",
    )

    def run(
        self,
        config: DriveFreqCfg,
        *,
        context: RunContext,
    ) -> DriveFreqResult:
        """Map the MIST response versus probe frequency and gain.

        Reset, apply the optional init pulse, apply the probe, then read
        out. Sweep both probe frequency and gain on the board. Return
        hardware frequencies in MHz, gains, and mean complex IQ shaped
        [frequency, gain], retaining that axis order in the live map.
        The IQ change measures the probe-induced readout response, not
        calibrated state populations. ``config`` supplies the probe and
        readout modules and both sweep ranges.
        """
        # 1. Prepare both hardware axes using the probe's generator channel.
        cfg = deepcopy(config)
        soc, soccfg = context.soc, context.soccfg
        modules = cfg.modules

        freq_sweep = cfg.sweep.freq
        gain_sweep = cfg.sweep.gain

        probe_pulse = modules.probe_pulse
        freqs = sweep2array(
            freq_sweep, "freq", {"soccfg": soccfg, "gen_ch": probe_pulse.ch}
        )
        gains = sweep2array(
            gain_sweep, "gain", {"soccfg": soccfg, "gen_ch": probe_pulse.ch}
        )

        viewer = context.plots.liveplot_2d(
            "measurement", "Pulse frequency (MHz)", "Pulse gain (a.u.)"
        )
        signals_buffer = SignalBuffer(
            (len(freqs), len(gains)),
            on_update=lambda data: viewer.update(
                freqs, gains, drivefreq_signal2real(data)
            ),
        )

        # 2. Bind frequency and gain sweeps to the same probe pulse.
        with Schedule(cfg, signals_buffer, scope=context.scope) as sched:
            modules = sched.cfg.modules
            modules.probe_pulse.set_param(
                "freq", sweep2param("freq", sched.cfg.sweep.freq)
            )
            modules.probe_pulse.set_param(
                "gain", sweep2param("gain", sched.cfg.sweep.gain)
            )

            # 3. Reset and initialize before each probe/readout sequence.
            _ = (
                sched.prog_builder(soc, soccfg)
                .add(
                    Reset("reset", modules.reset),
                    Pulse("init_pulse", modules.init_pulse),
                    Pulse("probe_pulse", modules.probe_pulse),
                    Readout("readout", modules.readout),
                )
                .declare_sweep("freq", sched.cfg.sweep.freq)
                .declare_sweep("gain", sched.cfg.sweep.gain)
                .build_and_acquire()
            )

        # 4. Preserve complex IQ; the baseline-subtracted magnitude is a view.
        return DriveFreqResult(gains=gains, freqs=freqs, signals=signals_buffer.array)

    def analyze(
        self,
        source: RunRecord[DriveFreqCfg, DriveFreqResult],
        options: None,
        *,
        plots: Plots,
    ) -> None:
        """Plot MIST magnitude versus probe MHz and gain from ``source``.

        Subtract one mean IQ reference pooled over all frequencies and the
        first 5% of gain samples (at least one). ``options`` is None; this
        response map returns no population fit or calibration value.
        """
        del options
        result = source.result

        freqs, gains, signals = result.freqs, result.gains, result.signals

        real_signals = drivefreq_signal2real(signals)

        _, ax = plots.subplots("fit", figsize=config.figsize)

        ax.imshow(
            real_signals.T,
            extent=(freqs[0], freqs[-1], gains[0], gains[-1]),
            aspect="auto",
            origin="lower",
            interpolation="none",
            cmap="RdBu_r",
        )
        ax.set_xlabel("Pulse frequency (MHz)", fontsize=14)
        ax.set_ylabel("Pulse gain (a.u.)", fontsize=14)
