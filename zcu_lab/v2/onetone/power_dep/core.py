from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import AfterValidator
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.v2.runtime import RunOptions
from zcu_tools.experiment.v2.runtime.schedule import (
    Schedule,
    SignalBuffer,
)
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.v2.utils.snr import snr_checker
from zcu_tools.program.v2 import GainSweep, PulseReadout, Reset, SweepCfg, TimelineBuilder
from zcu_tools.utils.process import minus_background, rescale

from zcu_lab.v2.support import live_heatmap, live_line


@dataclass(frozen=True)
class PowerDepResult:
    gains: NDArray[np.float64]
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]


def gaindep_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rescale(minus_background(np.abs(signals), axis=-1), axis=-1)


def _positive_snr_or_none(value: float | None) -> float | None:
    if value is not None and value <= 0:
        raise ValueError("SNR threshold must be positive or left blank (None)")
    return value


class PowerDepCfg(ConfigBase):

    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]
    gain: Annotated[GainSweep, FieldMeta(label="Gain", unit="a.u.")]
    freq: Annotated[SweepCfg, FieldMeta(label="Frequency", unit="MHz")]
    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ] = 1
    rounds: Annotated[
        int,
        FieldMeta(label="Rounds", description="Number of software acquisition rounds."),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ] = 1.0
    earlystop_snr: Annotated[
        float | None,
        AfterValidator(_positive_snr_or_none),
        FieldMeta(
            label="Early-stop SNR",
            description="Positive SNR threshold; leave blank (None) to disable early stopping.",
            decimals=3,
        ),
    ] = None


class PowerDepExp(Experiment[PowerDepResult, PowerDepCfg]):
    # freqs stored as Hz on disk (MHz * 1e6); gains stored as-is (IDENTITY)
    AXES_SPEC: ClassVar[AxesSpec[PowerDepResult, PowerDepCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("freqs", label="Frequency", unit="Hz", scale=1e6),
            AxesSpec.Axis("gains", label="Power", unit="a.u."),
        ),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u."),
        result_type=PowerDepResult,
        cfg_type=PowerDepCfg,
        cfg_schema_version="3.0",
        tag="onetone/power_dep",
    )

    def run(self, config: PowerDepCfg, *, context: RunContext) -> PowerDepResult:
        """Map the resonator spectrum versus readout gain.

        For each host-side gain, apply optional reset and sweep the readout
        frequency on the board. Return hardware-rounded gains (a.u.),
        frequencies (MHz), and mean complex IQ shaped (Ngain, Nfreq).
        Gain-dependent resonance changes appear in this map; gain is not
        a calibrated power in watts.

        cfg.earlystop_snr can end averaging of the current gain's trace
        once its SNR reaches the threshold; None disables this early stop.
        """

        # Prepare the gain points and matched generator/ADC frequency grid.
        cfg = deepcopy(config)
        soc, soccfg = context.soc, context.soccfg

        gain_sweep = cfg.gain

        readout_cfg = cfg.readout
        gains = sweep2array(
            gain_sweep,
            "gain",
            {"soccfg": soccfg, "gen_ch": readout_cfg.gen_ch},
            allow_array=True,
        )
        freqs = sweep2array(
            cfg.freq,
            "freq",
            {
                "soccfg": soccfg,
                "gen_ch": readout_cfg.gen_ch,
                "ro_ch": readout_cfg.ro_ch,
            },
        )

        # Normalize magnitude only for display and the per-trace SNR check.
        current_snr = 0.0

        def update_snr(snr: float) -> None:
            nonlocal current_snr
            current_snr = snr

        fig, (heat_ax, line_ax) = context.plots.subplots(1, 2, num="measurement")
        heat_ax.set(xlabel="Power (a.u.)", ylabel="Frequency (MHz)")
        line_ax.set_xlabel("Frequency (MHz)")
        heatmap = live_heatmap(heat_ax, uniform=True)
        lines = live_line(line_ax, num_lines=10)
        for artist in line_ax.lines[:-1]:
            artist.set(marker="None", alpha=0.3, color="red")
        line_ax.lines[-1].set(label="current line", marker=".", color="C0")

        @context.plots.on_plot_thread
        def update(data):
            real = gaindep_signal2real(data)
            valid = np.flatnonzero(~np.all(np.isnan(real), axis=1))
            current = int(valid[-1]) if valid.size else -1
            history = np.full((10, len(freqs)), np.nan)
            for offset in range(min(10, current + 1)):
                history[-offset - 1] = real[current - offset]
            artists = (*heatmap.set(gains, freqs, real), *lines.set(freqs, history))
            if current_snr:
                heat_ax.set_title(f"snr = {current_snr:.1f}")
            return artists

        fig.show()
        signals_buffer = SignalBuffer(
            (len(gains), len(freqs)),
            on_update=update,
        )
        with Schedule(None, signals_buffer, scope=context.scope) as sched:
            for _, step in sched.scan("gain", gains.tolist()):

                # Repeat reset/readout across frequency at this fixed gain.
                builder = TimelineBuilder(
                    soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay,
                )
                freq = builder.add_sweep("freq", cfg.freq)
                builder.emit(Reset("reset", cfg.reset))
                builder.emit(
                    PulseReadout("readout", cfg.readout, freq=freq, gain=step.value),
                )
                step.run_acquire(
                    builder.build(), soc,
                    options=RunOptions(
                        stop_condition=snr_checker(
                            signals_buffer[step], cfg.earlystop_snr,
                            gaindep_signal2real, after_check=update_snr,
                        ),
                    ),
                    rounds=cfg.rounds,
                )

        # Keep the acquired complex response, not the normalized preview.
        return PowerDepResult(
            gains=gains,
            freqs=freqs,
            signals=signals_buffer.array,
        )
