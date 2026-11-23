from __future__ import annotations

import logging
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.artist import Artist
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic import AfterValidator
from zcu_tools.analysis.fitting import HangerModel, TransmissionModel
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime import RunOptions
from zcu_tools.experiment.v2.runtime.schedule import (
    Schedule,
    SignalBuffer,
)
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.v2.utils.snr import snr_checker
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    PulseReadout,
    Reset,
    SweepCfg,
    TimelineBuilder,
)
from zcu_tools.utils.process import minus_background, rescale

from zcu_lab.v2.support import live_heatmap, live_line
from zcu_lab.v2.support.resonator import ResonanceFit, ResonanceOptions, fit_resonance

logger = logging.getLogger(__name__)


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


def _fit_gain_rows(result: PowerDepResult) -> dict[int, ResonanceFit]:
    """Fit raw IQ per gain; warn and omit unavailable fits, keyed by row index."""
    fits: dict[int, ResonanceFit] = {}
    options = ResonanceOptions()
    for index, (gain, row) in enumerate(zip(result.gains, result.signals, strict=True)):
        if len(result.freqs) < 6:
            # The fitter needs four samples after trimming the two endpoints.
            logger.warning(
                "Gain %.6g a.u.: resonance fit needs at least six RF points", gain
            )
            continue
        try:
            fits[index] = fit_resonance(result.freqs, row, options)
        except (ValueError, RuntimeError) as exc:
            logger.warning("Gain %.6g a.u.: resonance fit unavailable: %s", gain, exc)
    return fits


def _plot_fit_spectra(
    result: PowerDepResult, fits: dict[int, ResonanceFit], plots: Plots
) -> None:
    """Draw each successful fit and its values beside the measured spectrum."""
    if not fits:
        return
    dense = np.linspace(result.freqs[1], result.freqs[-2], max(201, len(result.freqs)))
    rows_per_figure = 6
    fit_rows = list(fits)
    for start in range(0, len(fit_rows), rows_per_figure):
        page = 1 + start // rows_per_figure
        rows = fit_rows[start : start + rows_per_figure]
        panel_rows = (len(rows) + 1) // 2
        fig, _ = plots.subplots(
            panel_rows,
            2,
            num=f"resonance fits {page:02d}",
            figsize=(11, 2.4 + 1.8 * panel_rows),
            layout="constrained",
        )
        fig.suptitle("One-tone power dependence · Resonator · Per-gain fits")
        fig.supylabel("Magnitude of mean IQ (a.u.)")
        axes = fig.axes
        for ax, index in zip(axes[: len(rows)], rows, strict=True):
            fit = fits[index]
            params = fit.params
            if "Qc" in params:
                prediction = HangerModel.calc_signals(dense, **params)
            else:
                prediction = TransmissionModel.calc_signals(dense, **params)
            ax.plot(
                result.freqs,
                np.abs(result.signals[index]),
                ".",
                label="Measured samples",
            )
            ax.plot(dense, np.abs(prediction), label="Complex fit")
            rms = fit.fit_quality.normalized_residual_rms
            residual = "unavailable" if rms is None else f"{rms:.3g}"
            ax.set_title(
                f"Gain = {result.gains[index]:.6g} a.u.\n"
                f"f₀ = {fit.freq:.6f} MHz · FWHM = {fit.fwhm:.4g} MHz\n"
                f"Normalized residual RMS = {residual}",
                fontsize=11,
            )
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        for ax in axes[2 * (panel_rows - 1) : len(rows)]:
            ax.set_xlabel("Readout RF frequency (MHz)")
        for unused in axes[len(rows) :]:
            unused.remove()
        fig.legend(
            *axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=2
        )


class _PowerMap:
    """Draw the gain map and the ten cuts ending at the latest plottable gain."""

    CUT_SLOTS = 10

    def __init__(
        self,
        plots: Plots,
        gains: NDArray[np.float64],
        freqs: NDArray[np.float64],
        *,
        live: bool = False,
    ) -> None:
        self.gains, self.freqs, self.live = gains, freqs, live
        columns = 1 if live else 2
        self.figure, _ = plots.subplots(
            2,
            columns,
            num="measurement",
            figsize=None if live else (11, 6.4),
            layout="constrained",
        )
        self.figure.suptitle("One-tone power dependence · Resonator · IQ magnitude")
        heat_ax, line_ax = self.figure.axes[::columns]
        self.map_ax = heat_ax
        self.profile_axes = self.figure.axes[1::2] if not live else []
        if not live:
            line_ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        heat_ax.set(xlabel="Readout gain (a.u.)", ylabel="Readout RF frequency (MHz)")
        heat_ax.grid(False)
        line_ax.set(
            xlabel="Readout RF frequency (MHz)", ylabel="Normalized magnitude (a.u.)"
        )
        self.heatmap = live_heatmap(heat_ax, uniform=True)
        self.figure.colorbar(
            heat_ax.images[0], ax=heat_ax, label="Normalized magnitude (a.u.)"
        )
        self.lines = live_line(line_ax, num_lines=self.CUT_SLOTS)
        for artist in line_ax.lines[:-1]:
            artist.set(marker="None", alpha=0.3, color="red")
        line_ax.lines[-1].set(marker=".", color="C0")
        heat_ax.set_title("Gain map", fontsize=11)
        self.slice_label = line_ax.set_title("Latest gain spectrum", fontsize=11)

    def set(
        self, data: NDArray[np.complex128], *, snr: float | None = None
    ) -> tuple[Artist, ...]:
        measured = np.flatnonzero(np.any(np.isfinite(data), axis=1))
        if self.live and not measured.size:
            return ()
        real = np.full(data.shape, np.nan)
        if measured.size:
            real[measured] = gaindep_signal2real(data[measured])
        valid = np.flatnonzero(~np.all(np.isnan(real), axis=1))
        history = np.full((self.CUT_SLOTS, len(self.freqs)), np.nan)
        latest = "Latest gain spectrum"
        if valid.size:
            current = valid[-1]
            count = min(len(history), current + 1)
            history[-count:] = real[current + 1 - count : current + 1]
            latest = f"Latest gain: {self.gains[current]:.4g} a.u."
            if self.live and snr is not None and current == measured[-1]:
                latest += f" · SNR {snr:.1f}"
        self.slice_label.set_text(latest)
        return (
            *self.heatmap.set(self.gains, self.freqs, real),
            *self.lines.set(self.freqs, history),
            self.slice_label,
        )


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

        power_map = _PowerMap(context.plots, gains, freqs, live=True)

        @context.plots.on_plot_thread
        def update(data: NDArray[np.complex128]) -> Iterable[Artist]:
            """Display measured gain columns and the latest spectra."""
            return power_map.set(data, snr=current_snr)

        power_map.figure.show()
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

    def analyze(
        self,
        source: RunRecord[PowerDepCfg, PowerDepResult],
        options: None,
        *,
        plots: Plots,
    ) -> None:
        """Plot the retained map, cuts and per-gain resonance fits without writeback.

        Pass None for options. A row that cannot be fitted logs a warning and
        contributes no fitted values; the measurement figure is still drawn.
        """
        del options
        result = source.result

        # 1. Redraw the measured map and cuts from the retained run or load.
        power_map = _PowerMap(plots, result.gains, result.freqs)
        plots.on_plot_thread(power_map.set)(result.signals)

        # 2. Show how the successful complex fits change with readout gain.
        fits = _fit_gain_rows(result)
        fitted_gains = result.gains[list(fits)]
        center_ax, width_ax = power_map.profile_axes
        gain_limits = power_map.map_ax.get_xlim()
        for ax in power_map.profile_axes:
            ax.sharex(power_map.map_ax)
        center_ax.plot(fitted_gains, [fit.freq for fit in fits.values()], "o")
        width_ax.plot(fitted_gains, [fit.fwhm for fit in fits.values()], "o")
        center_ax.set(xlabel="Readout gain (a.u.)", ylabel="Fitted center f₀ (MHz)")
        width_ax.set(xlabel="Readout gain (a.u.)", ylabel="Fitted FWHM (MHz)")
        center_ax.ticklabel_format(useOffset=False)
        power_map.map_ax.set_xlim(gain_limits)

        # 3. Show the data, model and fitted values for each successful gain.
        _plot_fit_spectra(result, fits, plots)
