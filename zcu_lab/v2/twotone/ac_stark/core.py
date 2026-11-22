from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.artist import Artist
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from pydantic import Field
from pydantic.dataclasses import dataclass as validated_dataclass
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, config
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime import RunOptions
from zcu_tools.experiment.v2.runtime.schedule import (
    Schedule,
    SignalBuffer,
)
from zcu_tools.experiment.v2.utils.round_zcu import round_zcu_gain, sweep2array
from zcu_tools.experiment.v2.utils.snr import snr_checker
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Pulse,
    Readout,
    Reset,
    SweepCfg,
    TimelineBuilder,
)
from zcu_tools.utils.process import rotate2real

from zcu_lab.v2.support import live_heatmap, live_line
from zcu_lab.v2.support.resonance import get_resonance_freq


@dataclass(frozen=True)
class AcStarkResult:
    """Stark gains, probe frequencies in MHz, and mean IQ shaped (Ngain, Nfreq)."""

    gains: NDArray[np.float64]
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]


@validated_dataclass(frozen=True)
class AcStarkAnalyzeOptions:
    """Dispersive chi/kappa in MHz; degree 1 or 2; optional exclusive gain cutoff."""

    chi: float
    kappa: float
    deg: Annotated[int, Field(ge=1, le=2)] = 1
    cutoff: float | None = None


@dataclass(frozen=True)
class AcStarkAnalysis:
    """Mean photon number coefficient: n = ac_coeff * Stark gain squared."""

    ac_coeff: float


def acstark_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Convert mean complex IQ to the real contrast used by live and analysis plots."""
    real_signals = rotate2real(signals).real

    valid_mask = np.any(~np.isnan(real_signals), axis=1)

    if not np.any(valid_mask):
        return real_signals

    valid_signals = real_signals[valid_mask, :]

    min_vals = np.nanmin(valid_signals, axis=1, keepdims=True)
    max_vals = np.nanmax(valid_signals, axis=1, keepdims=True)
    valid_signals = (valid_signals - min_vals) / (max_vals - min_vals)

    real_signals[valid_mask, :] = valid_signals

    return real_signals


class AcStarkCfg(ConfigBase):
    reset: Reset.Cfg | None = None
    stark_tone: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Stark resonator tone",
            description="Fill in the resonator drive pulse at the readout RF; the gain sweep sets its amplitude.",
        ),
    ]
    probe: Annotated[
        Pulse.Cfg,
        FieldMeta(
            label="Qubit spectroscopy probe",
            description="Fill in the qubit drive pulse applied with the Stark tone; the RF sweep sets its carrier.",
        ),
    ]
    readout: Annotated[
        Readout.Cfg,
        FieldMeta(
            label="Readout",
            description="Fill in the resonator readout applied after the spectroscopy probe.",
        ),
    ]
    gain: Annotated[
        SweepCfg,
        FieldMeta(
            label="Stark tone gain",
            unit="a.u.",
            description="Fill in the resonator drive gain endpoints (a.u.) and number of spectra.",
        ),
    ]
    freq: Annotated[
        SweepCfg,
        FieldMeta(
            label="Probe RF frequency",
            unit="MHz",
            description="Fill in the qubit drive RF endpoints (MHz) and points per spectrum.",
        ),
    ]
    earlystop_snr: Annotated[
        float | None,
        FieldMeta(
            label="Early-stop SNR",
            description="Set an SNR target to stop averaging one spectrum; blank uses all rounds.",
        ),
    ] = None

    reps: Annotated[
        int,
        FieldMeta(
            label="Reps", unit="count", description="Number of repetitions per round."
        ),
    ]
    rounds: Annotated[
        int,
        FieldMeta(
            label="Rounds",
            unit="count",
            description="Number of software acquisition rounds.",
        ),
    ] = 1
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="µs",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ]


class AcStarkExp(Experiment[AcStarkResult, AcStarkCfg]):
    Options: ClassVar[type[AcStarkAnalyzeOptions]] = AcStarkAnalyzeOptions

    # gains stored in a.u. -> IDENTITY; freqs stored in Hz on disk -> MHZ_TO_HZ
    AXES_SPEC: ClassVar[AxesSpec[AcStarkResult, AcStarkCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("freqs", label="Frequency", unit="Hz", scale=1e6),
            AxesSpec.Axis("gains", label="Stark Pulse Gain", unit="a.u."),
        ),
        z=AxesSpec.Z("signals", label="Signal", unit="a.u."),
        result_type=AcStarkResult,
        cfg_type=AcStarkCfg,
        cfg_schema_version="4.0",
        tag="twotone/ge/ac_stark",
    )

    def run(
        self,
        cfg: AcStarkCfg,
        *,
        context: RunContext,
    ) -> AcStarkResult:
        """Measure the qubit's AC Stark shift versus resonator drive gain.

        Reset, launch stark_tone without blocking, then play the
        frequency-swept qubit probe probe and read out. Configure the
        pulse timing for the intended overlap. Host gain points are uniform
        in gain squared before hardware rounding; the FPGA sweeps probe
        frequency. Return mean complex IQ shaped (Ngain, Nfreq), with
        frequencies in MHz. ``earlystop_snr`` can end averaging per spectrum.
        """
        # 1. Prepare probe frequencies and gain points spaced by nominal drive power.
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg

        gain_sweep = cfg.gain
        freq_sweep = cfg.freq

        freqs = sweep2array(
            freq_sweep,
            "freq",
            {"soccfg": soccfg, "gen_ch": cfg.probe.ch},
        )
        gains = np.sqrt(
            np.linspace(gain_sweep.start**2, gain_sweep.stop**2, gain_sweep.expts)
        )
        gains = round_zcu_gain(gains, soccfg, cfg.stark_tone.ch)

        current_snr = 0.0

        def update_snr(snr: float) -> None:
            """Record the latest checked SNR for the live plot title."""
            nonlocal current_snr
            current_snr = snr

        signals_buffer = self._live_spectra(
            context.plots, gains, freqs, lambda: current_snr
        )

        # 3. Acquire one frequency trace per Stark gain, retaining its SNR early stop.
        with Schedule(signals_buffer, scope=context.scope) as sched:
            for _, step in sched.scan("gain", gains.tolist()):
                builder = TimelineBuilder(
                    soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
                )
                freq = builder.add_sweep("freq", cfg.freq)
                if cfg.reset is not None:
                    builder.emit(Reset("reset", cfg.reset))
                # The Stark tone does not block the probe or the following readout.
                builder.fork("default", into=["stark", "probe"])
                builder.emit(
                    Pulse("stark_tone", cfg.stark_tone, gain=step.value),
                    timeline="stark",
                )
                builder.emit(Pulse("probe", cfg.probe, freq=freq), timeline="probe")
                builder.join("probe", into="default")
                builder.emit(Readout("readout", cfg.readout))
                program = builder.build()
                step.run_acquire(
                    program,
                    soc,
                    rounds=cfg.rounds,
                    options=RunOptions(
                        stop_condition=snr_checker(
                            signals_buffer[step],
                            cfg.earlystop_snr,
                            lambda x: rotate2real(x).real,
                            after_check=update_snr,
                        )
                    ),
                )

        # 4. Preserve the gain-frequency grid for photon calibration.
        return AcStarkResult(gains, freqs, signals_buffer.array)

    def _live_spectra(
        self,
        plots: Plots,
        gains: NDArray[np.float64],
        freqs: NDArray[np.float64],
        snr_now: Callable[[], float],
    ) -> SignalBuffer:
        """Attach the map and recent spectra to the run's live buffer."""
        # 2. Track the Stark-shift map, recent frequency traces and early-stop SNR.
        fig, _ = plots.subplots(
            1, 2, num="measurement", figsize=(10, 4.8), layout="constrained"
        )
        fig.suptitle("AC Stark spectroscopy · qubit probe · IQ contrast")
        heat_ax, line_ax = fig.axes
        heat_ax.set(xlabel="Stark tone gain (a.u.)", ylabel="Probe RF frequency (MHz)")
        contrast_label = "Row-normalized IQ projection (a.u.)"
        line_ax.set(xlabel="Probe RF frequency (MHz)", ylabel=contrast_label)
        heat_ax.grid(False)
        heatmap = live_heatmap(heat_ax, uniform=False, cmap="viridis")
        (image,) = heatmap.set(gains, freqs, np.full((len(gains), len(freqs)), np.nan))
        fig.colorbar(image, ax=heat_ax, label=contrast_label)
        line_ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        lines = live_line(line_ax, num_lines=2)
        line_ax.lines[0].set(
            label="Previous gain", marker="o", markersize=3, color="0.55"
        )
        line_ax.lines[1].set(label="Latest gain", marker="o", markersize=3, color="C0")
        line_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncols=2)

        @plots.on_plot_thread
        def update(data: NDArray[np.complex128]) -> tuple[Artist, ...]:
            """Refresh the Stark contrast map, two recent traces and SNR title.

            data is complex IQ shaped (gain, frequency); NaN marks unmeasured points.
            Return the changed artists without modifying the input array.
            """
            real = acstark_signal2real(data)
            valid = np.flatnonzero(~np.all(np.isnan(real), axis=1))
            current = int(valid[-1]) if valid.size else -1
            history = np.full((2, len(freqs)), np.nan)
            for offset in range(min(2, current + 1)):
                history[-offset - 1] = real[current - offset]
            artists = heatmap.set(gains, freqs, real)
            artists = (*artists, *lines.set(freqs, history))
            progress = heat_ax.set_title(
                f"Gain rows with data: {len(valid)}/{len(gains)}"
            )
            if current >= 0:
                current_snr = snr_now()
                snr = f" · SNR {current_snr:.1f}" if current_snr else ""
                line_ax.set_title(f"Latest gain: {gains[current]:.4g} a.u.{snr}")
            return (*artists, progress)

        fig.show()
        return SignalBuffer(
            (len(gains), len(freqs)),
            on_update=update,
        )

    def analyze(
        self,
        source: RunRecord[AcStarkCfg, AcStarkResult],
        options: AcStarkAnalyzeOptions,
        *,
        plots: Plots,
    ) -> AcStarkAnalysis:
        """Convert the gain-squared Stark slope to a mean-photon calibration.

        Extract Lorentzian qubit centers and fit frequency against gain
        squared. Supply chi and kappa in MHz; ``deg`` must be 1 or 2, else
        raise ValueError. ``cutoff`` excludes gains at or above that value.
        Return ac_coeff for mean photon number = ac_coeff * gain**2, using
        the linear coefficient and the dispersive correction from chi/kappa.
        Publish ``fit``; source cfg may be None.
        """
        # 1. Track resonance centers below the chosen Stark-gain cutoff.
        result = source.result
        chi, kappa, deg, cutoff = (
            options.chi,
            options.kappa,
            options.deg,
            options.cutoff,
        )

        gains, freqs, signals = result.gains, result.freqs, result.signals

        # apply cutoff if provided
        if cutoff is not None:
            valid_indices = np.where(gains < cutoff)[0]
            gains = gains[valid_indices]
            signals = signals[valid_indices, :]

        amps = acstark_signal2real(signals)
        s_gains, s_freqs = get_resonance_freq(gains, freqs, amps)

        gains2 = gains**2
        s_gains2 = s_gains**2

        # 2. Fit the Stark shift versus gain squared, allowing optional curvature.
        x2_fit = np.linspace(min(gains2), max(gains2), 240)
        if deg == 1:
            b, c = np.polyfit(s_gains2, s_freqs, 1)
            y_fit = b * x2_fit + c
        else:
            a, b, c = np.polyfit(s_gains2, s_freqs, 2)
            y_fit = a * x2_fit**2 + b * x2_fit + c

        # 3. Convert the linear Stark slope to photons using the supplied chi/kappa.
        eta = kappa**2 / (kappa**2 + chi**2)
        ac_coeff = abs(b) / (2 * eta * chi)

        # 4. Overlay the tracked ridge and its gain-squared polynomial on the Run map.
        fig, ax = plots.subplots(
            num="fit", figsize=config.figsize, layout="constrained"
        )
        ax.set(
            xlabel="Stark tone gain (a.u.)",
            ylabel="Probe RF frequency (MHz)",
            title=(
                "AC Stark spectroscopy · qubit probe · photon calibration\n"
                f"Photons per squared gain = {ac_coeff:.3g} a.u.⁻²\n"
                f"Centers used: {len(s_freqs)}/{len(gains)}"
            ),
        )
        ax.grid(False)
        (image,) = live_heatmap(ax, uniform=False, cmap="viridis").set(
            gains, freqs, amps
        )
        fig.colorbar(image, ax=ax, label="Row-normalized IQ projection (a.u.)")
        ax.plot(
            s_gains,
            s_freqs,
            "o",
            color="white",
            markeredgecolor="0.25",
            markersize=4,
            label="Measured ridge centers",
        )
        ax.plot(np.sqrt(x2_fit), y_fit, color="C1", label="Polynomial in gain²")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=2)

        return AcStarkAnalysis(ac_coeff)
