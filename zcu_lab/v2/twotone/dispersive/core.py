from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Annotated, Any, ClassVar

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from numpy.typing import NDArray
from zcu_tools.analysis.fitting.resonance import (
    HangerModel,
    find_edelay_branch,
    fit_edelay,
    get_proper_model,
    normalize_signal,
)
from zcu_tools.analysis.fitting.resonance.base import remove_background
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, ProgramExpCfg
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Branch,
    Pulse,
    PulseReadout,
    Reset,
    SweepCfg,
    sweep2param,
)


@dataclass(frozen=True)
class DispersiveResult:
    """Readout frequencies in MHz and mean IQ shaped (2, Nfreq).

    ``ge`` labels branches 0/1 without/with the qubit preparation pulse;
    these labels do not certify pure ground/excited populations.
    """

    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]
    ge: NDArray[np.int64] = field(default_factory=lambda: np.array([0, 1]))


def dispersive_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(signals)


class DispersiveModuleCfg(ConfigBase):
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    init_pulse: Annotated[Pulse.Cfg | None, FieldMeta(label="Initial pulse")] = None
    qub_pulse: Annotated[Pulse.Cfg, FieldMeta(label="Qubit preparation pulse")]
    readout: Annotated[PulseReadout.Cfg, FieldMeta(label="Readout")]


class DispersiveSweepCfg(ConfigBase):
    freq: Annotated[SweepCfg, FieldMeta(label="Readout freq", unit="MHz")]


class DispersiveCfg(ProgramExpCfg):
    modules: DispersiveModuleCfg
    sweep: DispersiveSweepCfg


@dataclass(frozen=True)
class DispersiveAnalyzeOptions:
    fit_bg_amp_slope: bool = False


@dataclass(frozen=True)
class DispersiveAnalysis:
    chi: float
    avg_fwhm: float


def _plot_circle_fit(
    ax: Axes,
    freqs: NDArray[np.float64],
    signals: NDArray[np.complex128],
    params_dict: dict[str, Any],
    color: str,
    label: str,
) -> None:
    """Plot the background-corrected normalized IQ trace and fitted circle."""
    corrected = remove_background(
        freqs,
        signals,
        freq=params_dict["freq"],
        edelay=params_dict["edelay"],
        bg_amp_slope=params_dict["bg_amp_slope"],
    )
    norm_signals, norm_circle_params = normalize_signal(
        corrected, params_dict["circle_params"], params_dict["a0"]
    )
    norm_xc, norm_yc, norm_r0 = norm_circle_params

    ax.plot(
        norm_signals.real,
        norm_signals.imag,
        color=color,
        marker=".",
        markersize=1,
        label=label,
    )
    ax.add_patch(Circle((norm_xc, norm_yc), norm_r0, fill=False, color=color))
    ax.plot([norm_xc, 1], [norm_yc, 0], "kx--")
    ax.axhline(0, color="k", linestyle="--")
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_ylabel(r"$Im[S_{21}]$", fontsize=14)
    ax.yaxis.set_label_position("right")
    ax.legend()


class DispersiveExp(Experiment[DispersiveResult, DispersiveCfg]):
    Options: ClassVar[type[DispersiveAnalyzeOptions]] = DispersiveAnalyzeOptions

    # inner freqs stores MHz on disk (disk Hz) -> scale=1e6; outer ge index -> IDENTITY
    AXES_SPEC: ClassVar[AxesSpec[DispersiveResult, DispersiveCfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("freqs", "Frequency", "Hz", scale=1e6),
            AxesSpec.Axis("ge", "Amplitude", "None", dtype=np.int64),
        ),
        z=AxesSpec.Z("signals", "Signal", "a.u."),
        result_type=DispersiveResult,
        cfg_type=DispersiveCfg,
        cfg_schema_version="2.0",
        tag="twotone/ge/dispersive",
    )

    def run(
        self,
        cfg: DispersiveCfg,
        *,
        context: RunContext,
    ) -> DispersiveResult:
        """Compare resonator spectra without and with a qubit preparation pulse.

        Reset and apply the optional init pulse, then omit or apply the
        configured qubit pulse (nominally pi) before frequency-swept readout.
        Return mean complex IQ for both branches; analysis infers the
        dispersive shift from their resonance separation.
        """
        # Prepare a common frequency axis and separate IQ traces for both branches.
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg

        modules = cfg.modules

        freq_sweep = cfg.sweep.freq

        freqs = sweep2array(
            freq_sweep,
            "freq",
            {"soccfg": soccfg, "gen_ch": modules.qub_pulse.ch},
        )

        viewer = context.plots.liveplot_1d(
            "measurement", "Frequency (MHz)", "Amplitude", num_lines=2
        )
        signals_buffer = SignalBuffer(
            (2, len(freqs)),
            on_update=lambda data: viewer.update(freqs, dispersive_signal2real(data)),
        )

        # Sweep readout frequency for both preparations in the same FPGA program.
        with Schedule(cfg, signals_buffer, scope=context.scope) as sched:
            cfg = sched.cfg
            modules = cfg.modules
            freq_sweep = cfg.sweep.freq
            modules.readout.set_param("freq", sweep2param("freq", freq_sweep))

            _ = (
                sched.prog_builder(soc, soccfg)
                .add(
                    Reset("reset", modules.reset),
                    Pulse("init_pulse", modules.init_pulse),
                    Branch("ge", [], Pulse("qub_pulse", modules.qub_pulse)),
                    PulseReadout("readout", modules.readout),
                )
                .declare_sweep("ge", 2)
                .declare_sweep("freq", freq_sweep)
                .build_and_acquire()
            )

        return DispersiveResult(freqs=freqs, signals=signals_buffer.array)

    def analyze(
        self,
        source: RunRecord[DispersiveCfg, DispersiveResult],
        options: DispersiveAnalyzeOptions,
        *,
        plots: Plots,
    ) -> DispersiveAnalysis:
        """Fit the two resonator IQ traces with a shared electrical delay.

        Return half the absolute resonance separation (chi) and the average
        FWHM in MHz. ``fit_bg_amp_slope`` allows a sloped amplitude background;
        the ``fit`` figure shows spectra and normalized IQ circles. The
        source cfg may be None; the preparation labels alone do not establish
        state purity or a readout fidelity.
        """
        result = source.result

        freqs = result.freqs
        signals = result.signals
        g_signals, e_signals = signals[0, :], signals[1, :]
        g_amps, e_amps = np.abs(g_signals), np.abs(e_signals)

        branch_seed = find_edelay_branch(freqs, np.stack((g_signals, e_signals)))
        g_edelay = fit_edelay(freqs, g_signals, branch_seed=branch_seed)
        e_edelay = fit_edelay(freqs, e_signals, branch_seed=branch_seed)
        edelay = 0.5 * (g_edelay + e_edelay)

        model = get_proper_model(freqs, g_signals)
        # Keep each concrete model and its fitted parameter types in the same branch.
        if isinstance(model, HangerModel):
            g_params = model.fit(
                freqs,
                g_signals,
                edelay=edelay,
                fit_bg_amp_slope=options.fit_bg_amp_slope,
            )
            e_params = model.fit(
                freqs,
                e_signals,
                edelay=edelay,
                fit_bg_amp_slope=options.fit_bg_amp_slope,
            )
            g_fit = np.abs(model.calc_signals(freqs, **g_params))
            e_fit = np.abs(model.calc_signals(freqs, **e_params))
        else:
            g_params = model.fit(
                freqs,
                g_signals,
                edelay=edelay,
                fit_bg_amp_slope=options.fit_bg_amp_slope,
            )
            e_params = model.fit(
                freqs,
                e_signals,
                edelay=edelay,
                fit_bg_amp_slope=options.fit_bg_amp_slope,
            )
            g_fit = np.abs(model.calc_signals(freqs, **g_params))
            e_fit = np.abs(model.calc_signals(freqs, **e_params))

        g_freq, g_fwhm = g_params["freq"], g_params["fwhm"]
        e_freq, e_fwhm = e_params["freq"], e_params["fwhm"]

        # Calculate dispersive shift and average linewidth
        chi = abs(g_freq - e_freq) / 2  # dispersive shift χ/2π
        avg_fwhm = (g_fwhm + e_fwhm) / 2  # average linewidth κ/2π

        fig = Figure(figsize=(8, 4))
        plots.adopt("fit", fig)
        spec = fig.add_gridspec(2, 3, wspace=0.2)
        ax_main = fig.add_subplot(spec[:, :2])
        ax_g = fig.add_subplot(spec[0, 2])
        ax_e = fig.add_subplot(spec[1, 2])

        fig.suptitle(
            f"Dispersive shift χ/2π = {chi:.3f} MHz, κ/2π = {avg_fwhm:.1f} MHz"
        )

        # Plot data and fits
        label_g = f"Ground: {g_freq:.1f} MHz, κ = {g_fwhm:.1f} MHz"
        label_e = f"Excited: {e_freq:.1f} MHz, κ = {e_fwhm:.1f} MHz"
        ax_main.scatter(freqs, g_amps, marker=".", c="b")
        ax_main.scatter(freqs, e_amps, marker=".", c="r")
        ax_main.plot(freqs, g_fit, "b-", alpha=0.7, label=label_g)
        ax_main.plot(freqs, e_fit, "r-", alpha=0.7, label=label_e)

        # Mark resonance frequencies
        ax_main.axvline(g_freq, color="b", ls="--", alpha=0.7)
        ax_main.axvline(e_freq, color="r", ls="--", alpha=0.7)
        ax_main.set_xlabel("Frequency (MHz)", fontsize=14)
        ax_main.set_ylabel("Signal Amplitude (a.u.)", fontsize=14)

        ax_main.legend(loc="upper right")
        ax_main.grid(True)

        # Plot individual circle fit
        _plot_circle_fit(ax_g, freqs, g_signals, dict(g_params), "b", "Ground")
        _plot_circle_fit(ax_e, freqs, e_signals, dict(e_params), "r", "Excited")
        ax_e.set_xlabel(r"$Re[S_{21}]$", fontsize=14)

        return DispersiveAnalysis(chi, avg_fwhm)
