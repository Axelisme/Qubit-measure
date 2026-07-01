from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from numpy.typing import NDArray

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    IDENTITY,
    MHZ_TO_HZ,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
    record_result,
    retrieve_result,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runner import MeasureSession, TaskState
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.program.v2 import (
    Branch,
    ModularProgramV2,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    PulseReadout,
    PulseReadoutCfg,
    Reset,
    ResetCfg,
    SweepCfg,
    sweep2param,
)
from zcu_tools.utils.fitting.resonance import (
    fit_edelay,
    get_proper_model,
    normalize_signal,
    remove_edelay,
)


@dataclass(frozen=True)
class DispersiveResult:
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]
    ge: NDArray[np.int64] = field(default_factory=lambda: np.array([0, 1]))
    cfg_snapshot: DispersiveCfg | None = None


def dispersive_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(signals)


class DispersiveModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    init_pulse: PulseCfg | None = None
    qub_pulse: PulseCfg
    readout: PulseReadoutCfg


class DispersiveSweepCfg(ConfigBase):
    freq: SweepCfg


class DispersiveCfg(ProgramV2Cfg, ExpCfgModel):
    modules: DispersiveModuleCfg
    sweep: DispersiveSweepCfg


class DispersiveExp(PersistableExperiment[DispersiveResult, DispersiveCfg]):
    # inner freqs stores MHz on disk (disk Hz) -> scale=MHZ_TO_HZ; outer ge index -> IDENTITY
    AXES_SPEC = AxesSpec(
        axes=(
            Axis("freqs", "Frequency", "Hz", scale=MHZ_TO_HZ),
            Axis("ge", "Amplitude", "None", scale=IDENTITY, dtype=np.int64),
        ),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=DispersiveResult,
        cfg_type=DispersiveCfg,
        tag="twotone/ge/dispersive",
    )

    @record_result
    def run(
        self,
        soc,
        soccfg,
        cfg: DispersiveCfg,
        *,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> DispersiveResult:
        orig_cfg = deepcopy(cfg)

        setup_devices(cfg, progress=True)
        modules = cfg.modules

        freq_sweep = cfg.sweep.freq

        freqs = sweep2array(
            freq_sweep,
            "freq",
            {"soccfg": soccfg, "gen_ch": modules.qub_pulse.ch},
        )

        def measure_fn(ctx: TaskState, update_hook: Callable | None):
            cfg = ctx.cfg
            freq_param = sweep2param("freq", cfg.sweep.freq)
            cfg.modules.readout.set_param("freq", freq_param)

            return ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    Reset("reset", cfg.modules.reset),
                    Pulse("init_pulse", cfg.modules.init_pulse),
                    Branch(
                        "ge",
                        [],
                        Pulse("qub_pulse", cfg.modules.qub_pulse),
                    ),
                    PulseReadout("readout", cfg.modules.readout),
                ],
                sweep=[
                    ("ge", 2),
                    ("freq", cfg.sweep.freq),
                ],
            ).acquire(
                soc,
                progress=False,
                round_hook=update_hook,
                stop_checkers=[ctx.is_stop],
                **(acquire_kwargs or {}),
            )

        with LivePlot1D(
            "Frequency (MHz)", "Amplitude", segment_kwargs=dict(num_lines=2)
        ) as viewer:
            with MeasureSession(cfg) as run:
                signals_buffer = run.buffer(
                    (2, len(freqs)),
                    dtype=np.complex128,
                    on_update=lambda data: viewer.update(
                        freqs, dispersive_signal2real(data)
                    ),
                )
                signals_buffer.measure(measure_fn, pbar_n=run.cfg.rounds)
                signals = signals_buffer.array

        return DispersiveResult(freqs=freqs, signals=signals, cfg_snapshot=orig_cfg)

    @retrieve_result
    def analyze(
        self, result: DispersiveResult | None = None, fit_bg_slope: bool = False
    ) -> tuple[float, float, Figure]:
        assert result is not None, "no result found"

        freqs = result.freqs
        signals = result.signals
        g_signals, e_signals = signals[0, :], signals[1, :]
        g_amps, e_amps = np.abs(g_signals), np.abs(e_signals)

        g_edelay = fit_edelay(freqs, g_signals)
        e_edelay = fit_edelay(freqs, e_signals)
        edelay = 0.5 * (g_edelay + e_edelay)

        model = get_proper_model(freqs, g_signals)
        g_params = model.fit(freqs, g_signals, edelay=edelay, fit_bg_slope=fit_bg_slope)
        e_params = model.fit(freqs, e_signals, edelay=edelay, fit_bg_slope=fit_bg_slope)

        g_freq, g_fwhm = g_params["freq"], g_params["fwhm"]
        e_freq, e_fwhm = e_params["freq"], e_params["fwhm"]

        g_fit = np.abs(model.calc_signals(freqs, **g_params))  # type: ignore
        e_fit = np.abs(model.calc_signals(freqs, **e_params))  # type: ignore

        # Calculate dispersive shift and average linewidth
        chi = abs(g_freq - e_freq) / 2  # dispersive shift χ/2π
        avg_fwhm = (g_fwhm + e_fwhm) / 2  # average linewidth κ/2π

        fig = plt.figure(figsize=(8, 4))
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

        def _plot_circle_fit(
            ax: Axes,
            signals: NDArray[np.complex128],
            params_dict: dict[str, Any],
            color: str,
            label: str,
        ) -> None:
            rot_signals = remove_edelay(freqs, signals, edelay)
            norm_signals, norm_circle_params = normalize_signal(
                rot_signals, params_dict["circle_params"], params_dict["a0"]
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
            # ax.set_xlabel(r"$Re[S_{21}]$", fontsize=14)
            ax.set_ylabel(r"$Im[S_{21}]$", fontsize=14)
            ax.yaxis.set_label_position("right")
            ax.legend()

        # Plot individual circle fit
        _plot_circle_fit(ax_g, g_signals, dict(g_params), "b", "Ground")
        _plot_circle_fit(ax_e, e_signals, dict(e_params), "r", "Excited")
        ax_e.set_xlabel(r"$Re[S_{21}]$", fontsize=14)

        # fig.tight_layout()

        return chi, avg_fwhm, fig
