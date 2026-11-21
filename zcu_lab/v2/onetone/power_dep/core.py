from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.artist import Artist
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
from zcu_tools.program.v2 import (
    PulseReadout,
    Reset,
    SweepCfg,
    TimelineBuilder,
)
from zcu_tools.utils.process import minus_background, rescale

from zcu_lab.v2.support import live_heatmap, live_line


@dataclass(frozen=True)
class PowerDepResult:
    gains: NDArray[np.float64]
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]


def gaindep_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Return the real-valued measurement view of complex IQ without changing the input."""
    return rescale(minus_background(np.abs(signals), axis=-1), axis=-1)


def _positive_snr_or_none(value: float | None) -> float | None:
    """Return a positive SNR limit or None; reject nonpositive configured thresholds."""
    if value is not None and value <= 0:
        raise ValueError("SNR threshold must be positive or left blank (None)")
    return value


class PowerDepCfg(ConfigBase):
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    readout: Annotated[
        PulseReadout.Cfg,
        FieldMeta(
            label="Readout",
            description="Set the pulse and ADC window used at each gain and frequency.",
        ),
    ]
    gain: Annotated[
        SweepCfg,
        FieldMeta(
            label="Readout gain",
            unit="a.u.",
            description="Enter the readout DAC full-scale fractions to sweep, not power in watts.",
        ),
    ]
    freq: Annotated[
        SweepCfg,
        FieldMeta(
            label="Readout RF frequency",
            unit="MHz",
            description="Enter the absolute RF sweep endpoints in MHz for each gain.",
        ),
    ]
    reps: Annotated[
        int,
        FieldMeta(
            label="Reps",
            unit="count",
            description="Enter the shots per frequency and gain in each round.",
        ),
    ]
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Enter how many frequency sweeps to average at each gain.",
        ),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="µs",
            decimals=3,
            description="Enter the relaxation wait in µs after each shot.",
        ),
    ]
    earlystop_snr: Annotated[
        float | None,
        AfterValidator(_positive_snr_or_none),
        FieldMeta(
            label="Early-stop SNR",
            unit="a.u.",
            description="Set the positive signal-to-noise ratio at which to stop averaging a gain trace, or leave blank to use all rounds.",
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

        # 1. Prepare the gain points and matched generator/ADC frequency grid.
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

        # 2. Normalize magnitude only for display and the per-trace SNR check.
        current_snr: float | None = None

        def update_snr(snr: float) -> None:
            """Store the SNR estimate for the current gain only."""
            nonlocal current_snr
            current_snr = snr

        fig, _ = context.plots.subplots(2, 1, num="measurement", layout="constrained")
        fig.suptitle("One-tone power dependence · Resonator · IQ magnitude")
        heat_ax, line_ax = fig.axes
        heat_ax.set(xlabel="Readout gain (a.u.)", ylabel="Readout RF frequency (MHz)")
        heat_ax.grid(False)
        line_ax.set(
            xlabel="Readout RF frequency (MHz)", ylabel="Normalized magnitude (a.u.)"
        )
        heatmap = live_heatmap(heat_ax, uniform=True)
        fig.colorbar(heat_ax.images[0], ax=heat_ax, label="Normalized magnitude (a.u.)")
        line = live_line(line_ax)
        progress = heat_ax.set_title(
            f"Gain settings measured: 0 / {len(gains)}", fontsize=11
        )
        slice_label = line_ax.set_title("Latest gain spectrum", fontsize=11)

        @context.plots.on_plot_thread
        def update(data: NDArray[np.complex128]) -> Iterable[Artist]:
            """Display measured gain columns and the latest available spectrum."""
            valid = np.flatnonzero(np.any(np.isfinite(data), axis=1))
            if not valid.size:
                return ()
            real = np.full(data.shape, np.nan)
            real[valid] = gaindep_signal2real(data[valid])
            current = valid[-1]
            progress.set_text(f"Gain settings measured: {valid.size} / {len(gains)}")
            snr = "" if current_snr is None else f" · SNR {current_snr:.1f}"
            slice_label.set_text(f"Latest gain: {gains[current]:.4g} a.u.{snr}")
            return (
                *heatmap.set(gains, freqs, real),
                *line.set(freqs, real[current]),
                progress,
                slice_label,
            )

        fig.show()
        signals_buffer = SignalBuffer(
            (len(gains), len(freqs)),
            on_update=update,
        )
        with Schedule(signals_buffer, scope=context.scope) as sched:
            for _, step in sched.scan("gain", gains.tolist()):
                current_snr = None
                # Repeat reset/readout across frequency at this fixed gain.
                builder = TimelineBuilder(
                    soccfg,
                    reps=cfg.reps,
                    relax_delay=cfg.relax_delay,
                )
                freq = builder.add_sweep("freq", cfg.freq)
                if cfg.reset is not None:
                    builder.emit(Reset("reset", cfg.reset))
                builder.emit(
                    PulseReadout("readout", cfg.readout, freq=freq, gain=step.value),
                )
                step.run_acquire(
                    builder.build(),
                    soc,
                    options=RunOptions(
                        stop_condition=snr_checker(
                            signals_buffer[step],
                            cfg.earlystop_snr,
                            gaindep_signal2real,
                            after_check=update_snr,
                        ),
                    ),
                    rounds=cfg.rounds,
                )

        # 3. Keep the acquired complex response, not the normalized preview.
        signals_buffer.trigger_update(flush=True)
        return PowerDepResult(
            gains=gains,
            freqs=freqs,
            signals=signals_buffer.array,
        )
