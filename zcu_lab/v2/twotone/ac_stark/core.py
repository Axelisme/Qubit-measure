from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar

import numpy as np
from matplotlib.image import NonUniformImage
from numpy.typing import NDArray
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, config
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.program_builder import AcquireOptions
from zcu_tools.experiment.v2.runtime.schedule import (
    Schedule,
    SignalBuffer,
)
from zcu_tools.experiment.v2.utils.round_zcu import round_zcu_gain, sweep2array
from zcu_tools.experiment.v2.utils.snr import snr_checker
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    GainSweep,
    Pulse,
    Readout,
    Reset,
    SweepCfg,
    sweep2param,
)
from zcu_tools.utils.process import rotate2real

from zcu_lab.v2.support import live_heatmap_with_line
from zcu_lab.v2.support.resonance import get_resonance_freq


@dataclass(frozen=True)
class AcStarkResult:
    """Stark gains, probe frequencies in MHz, and mean IQ shaped (Ngain, Nfreq)."""

    gains: NDArray[np.float64]
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]


@dataclass(frozen=True)
class AcStarkAnalyzeOptions:
    chi: float
    kappa: float
    deg: int = 1
    cutoff: float | None = None


@dataclass(frozen=True)
class AcStarkAnalysis:
    """Mean photon number coefficient: n = ac_coeff * Stark gain squared."""

    ac_coeff: float


def acstark_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
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
    reset: Reset.Cfg | None = None
    stark_pulse1: Annotated[Pulse.Cfg, FieldMeta(label="Stark resonator tone")]
    stark_pulse2: Annotated[Pulse.Cfg, FieldMeta(label="Qubit spectroscopy probe")]
    readout: Readout.Cfg
    gain: Annotated[GainSweep, FieldMeta(label="Stark gain", unit="a.u.")]
    freq: Annotated[SweepCfg, FieldMeta(label="Probe frequency", unit="MHz")]
    earlystop_snr: Annotated[float | None, FieldMeta(label="Early-stop SNR")] = None


class AcStarkExp(Experiment[AcStarkResult, AcStarkCfg]):
    Options: ClassVar[type[AcStarkAnalyzeOptions]] = AcStarkAnalyzeOptions

    # gains stored in a.u. -> IDENTITY; freqs stored in Hz on disk -> MHZ_TO_HZ
    AXES_SPEC: ClassVar[AxesSpec[AcStarkResult, AcStarkCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("freqs", "Frequency", "Hz", scale=1e6),
            AxesSpec.Axis("gains", "Stark Pulse Gain", "a.u."),
        ),
        z=AxesSpec.Z("signals", "Signal", "a.u."),
        result_type=AcStarkResult,
        cfg_type=AcStarkCfg,
        cfg_schema_version="3.0",
        tag="twotone/ge/ac_stark",
    )

    def run(
        self,
        cfg: AcStarkCfg,
        *,
        context: RunContext,
    ) -> AcStarkResult:
        """Measure the qubit's AC Stark shift versus resonator drive gain.

        Reset, launch stark_pulse1 without blocking, then play the
        frequency-swept qubit probe stark_pulse2 and read out. Configure the
        pulse timing for the intended overlap. Host gain points are uniform
        in gain squared before hardware rounding; the FPGA sweeps probe
        frequency. Return mean complex IQ shaped (Ngain, Nfreq), with
        frequencies in MHz. ``earlystop_snr`` can end averaging per spectrum.
        """
        # Prepare probe frequencies and gain points spaced by nominal drive power.
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg

        gain_sweep = cfg.gain
        freq_sweep = cfg.freq

        freqs = sweep2array(
            freq_sweep,
            "freq",
            {"soccfg": soccfg, "gen_ch": cfg.stark_pulse2.ch},
        )
        gains = np.sqrt(
            np.linspace(gain_sweep.start**2, gain_sweep.stop**2, gain_sweep.expts)
        )
        gains = round_zcu_gain(gains, soccfg, cfg.stark_pulse1.ch)

        current_snr = 0.0

        def update_snr(snr: float) -> None:
            nonlocal current_snr
            current_snr = snr

        viewer = live_heatmap_with_line(
            context.plots,
            "measurement",
            "Stark Pulse Gain (a.u.)",
            "Frequency (MHz)",
            line_axis=1,
            num_lines=2,
            uniform=False,
        )
        signals_buffer = SignalBuffer(
            (len(gains), len(freqs)),
            on_update=lambda data: viewer.update(
                gains,
                freqs,
                acstark_signal2real(data),
                title=f"snr = {current_snr:.1f}" if current_snr else None,
            ),
        )

        # Acquire one frequency trace per Stark gain, retaining its SNR early stop.
        with Schedule(cfg, signals_buffer, scope=context.scope) as sched:
            for _, step in sched.scan("resonator gain", gains.tolist()):
                step_cfg = step.cfg

                step_cfg.stark_pulse1.set_param("gain", step.value)
                freq_sweep = step_cfg.freq
                step_cfg.stark_pulse2.set_param("freq", sweep2param("freq", freq_sweep))

                _ = (
                    step.prog_builder(
                        soc,
                        soccfg,
                        reps=step.cfg.reps,
                        relax_delay=step.cfg.relax_delay,
                    )
                    .add(
                        Reset("reset", step_cfg.reset),
                        # Launch the Stark tone without waiting before the probe.
                        Pulse("stark_pulse1", step_cfg.stark_pulse1, block_mode=False),
                        Pulse("stark_pulse2", step_cfg.stark_pulse2),
                        Readout("readout", step_cfg.readout),
                    )
                    .declare_sweep("freq", freq_sweep)
                    .build_and_acquire(
                        options=AcquireOptions(
                            stop_condition=snr_checker(
                                signals_buffer[step],
                                cfg.earlystop_snr,
                                lambda x: rotate2real(x).real,
                                after_check=update_snr,
                            )
                        ),
                        rounds=cfg.rounds,
                    )
                )

        return AcStarkResult(gains, freqs, signals_buffer.array)

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

        # Fit the Stark shift versus gain squared, allowing optional curvature.
        x2_fit = np.linspace(min(gains2), max(gains2), 100)
        if deg == 1:
            b, c = np.polyfit(s_gains2, s_freqs, 1)
            y_fit = b * x2_fit + c
        elif deg == 2:
            a, b, c = np.polyfit(s_gains2, s_freqs, 2)
            y_fit = a * x2_fit**2 + b * x2_fit + c
        else:
            raise ValueError(f"Degree {deg} is not supported.")

        # Convert the linear Stark slope to photons using the supplied chi/kappa.
        eta = kappa**2 / (kappa**2 + chi**2)
        ac_coeff = abs(b) / (2 * eta * chi)

        # plot the data and the fitted polynomial
        avg_n = ac_coeff * gains2

        fig, ax1 = plots.subplots("fit", figsize=config.figsize)

        # Use NonUniformImage for better visualization with gain^2 as x-axis
        im = NonUniformImage(ax1, cmap="viridis", interpolation="nearest")
        im.set_data(avg_n, freqs, amps.T)
        im.set_extent((avg_n[0], avg_n[-1], freqs[0], freqs[-1]))
        ax1.add_image(im)

        # Set proper limits for the plot
        ax1.set_xlim(avg_n[0], avg_n[-1])
        ax1.set_ylim(freqs[0], freqs[-1])

        # Plot the resonance frequencies and fitted curve
        ax1.plot(ac_coeff * s_gains2, s_freqs, ".", c="k")

        # Fit curve in terms of gain^2
        label = r"$\bar n$" + f" = {ac_coeff:.2g} " + r"$gain^2$"
        n_fit = ac_coeff * x2_fit
        ax1.plot(n_fit, y_fit, "-", label=label)

        # Create secondary x-axis for gain^2 (Readout Gain²)
        ax2 = ax1.twiny()

        # main x-axis: avg_n, secondary x-axis: gain^2
        # Average photon number equals ac_coeff times gain squared.
        ax1.set_xticks(ax1.get_xticks())
        ax1.set_xlabel(r"Average Photon Number ($\bar n$)", fontsize=14)

        # Show gain on the upper secondary axis.
        avgn_ticks = ax1.get_xticks()
        gain_ticks = np.sqrt(avgn_ticks / ac_coeff)
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xticks(avgn_ticks)
        ax2.set_xticklabels([f"{gain:.2g}" for gain in gain_ticks])
        ax2.set_xlabel("Readout Gain (a.u.)", fontsize=14)

        ax1.set_ylabel("Qubit Frequency (MHz)", fontsize=14)
        ax1.legend(fontsize="x-large")
        ax1.tick_params(axis="both", which="major", labelsize=12)

        fig.tight_layout()

        return AcStarkAnalysis(ac_coeff)
