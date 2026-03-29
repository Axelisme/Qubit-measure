from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from numpy.typing import NDArray
from typeguard import check_type
from typing_extensions import Any, Optional, TypeAlias

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import format_sweep1D
from zcu_tools.experiment.v2.runner import Task, TaskCfg, run_task
from zcu_tools.experiment.v2.utils import make_ge_sweep, sweep2array
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import (
    Pulse,
    Readout,
    TwoToneCfg,
    TwoToneProgram,
    sweep2param,
)
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.fitting.resonance import (
    fit_edelay,
    get_proper_model,
    normalize_signal,
    remove_edelay,
)

DispersiveResult: TypeAlias = tuple[NDArray[np.float64], NDArray[np.complex128]]


def dispersive_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(signals)


class DispersiveCfg(TwoToneCfg, TaskCfg):
    sweep: dict[str, SweepCfg]


class DispersiveExp(AbsExperiment[DispersiveResult, DispersiveCfg]):
    def run(self, soc, soccfg, cfg: dict[str, Any]) -> DispersiveResult:
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "freq")
        _cfg = check_type(deepcopy(cfg), DispersiveCfg)
        modules = _cfg["modules"]

        ge_sweep = make_ge_sweep()
        freq_sweep = _cfg["sweep"]["freq"]

        freqs = sweep2array(
            freq_sweep,
            "freq",
            {"soccfg": soccfg, "gen_ch": modules["qub_pulse"]["ch"]},
        )

        ge_param = sweep2param("ge", ge_sweep)
        freq_param = sweep2param("freq", freq_sweep)
        Pulse.set_param(modules["qub_pulse"], "on/off", ge_param)
        Readout.set_param(modules["readout"], "freq", freq_param)

        with LivePlotter1D(
            "Frequency (MHz)", "Amplitude", segment_kwargs=dict(num_lines=2)
        ) as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=lambda ctx, update_hook: TwoToneProgram(
                        soccfg,
                        ctx.cfg,
                        sweep=[
                            ("ge", ge_sweep),
                            ("freq", ctx.cfg["sweep"]["freq"]),
                        ],
                    ).acquire(soc, progress=False, callback=update_hook),
                    result_shape=(2, len(freqs)),
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    freqs, dispersive_signal2real(ctx.root_data)
                ),
            )

        # Cache results
        self.last_cfg = _cfg
        self.last_result = (freqs, signals)

        return freqs, signals

    def analyze(
        self, result: Optional[DispersiveResult] = None
    ) -> tuple[float, float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        freqs, signals = result
        g_signals, e_signals = signals[0, :], signals[1, :]
        g_amps, e_amps = np.abs(g_signals), np.abs(e_signals)

        g_edelay = fit_edelay(freqs, g_signals)
        e_edelay = fit_edelay(freqs, e_signals)
        edelay = 0.5 * (g_edelay + e_edelay)

        model = get_proper_model(freqs, g_signals)
        g_params = model.fit(freqs, g_signals, edelay=edelay)
        e_params = model.fit(freqs, e_signals, edelay=edelay)

        g_freq, g_kappa = g_params["freq"], g_params["kappa"]
        e_freq, e_kappa = e_params["freq"], e_params["kappa"]

        g_fit = np.abs(model.calc_signals(freqs, **g_params))  # type: ignore
        e_fit = np.abs(model.calc_signals(freqs, **e_params))  # type: ignore

        # Calculate dispersive shift and average linewidth
        chi = abs(g_freq - e_freq) / 2  # dispersive shift χ/2π
        avg_kappa = (g_kappa + e_kappa) / 2  # average linewidth κ/2π

        fig = plt.figure(figsize=(8, 4))
        spec = fig.add_gridspec(2, 3, wspace=0.2)
        ax_main = fig.add_subplot(spec[:, :2])
        ax_g = fig.add_subplot(spec[0, 2])
        ax_e = fig.add_subplot(spec[1, 2])

        fig.suptitle(
            f"Dispersive shift χ/2π = {chi:.3f} MHz, κ/2π = {avg_kappa:.1f} MHz"
        )

        # Plot data and fits
        label_g = f"Ground: {g_freq:.1f} MHz, κ = {g_kappa:.1f} MHz"
        label_e = f"Excited: {e_freq:.1f} MHz, κ = {e_kappa:.1f} MHz"
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

        return chi, avg_kappa, fig

    def save(
        self,
        filepath: str,
        result: Optional[DispersiveResult] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/dispersive",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        freqs, signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "Frequency", "unit": "Hz", "values": freqs * 1e6},
            y_info={"name": "Amplitude", "unit": "None", "values": np.array([0, 1])},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> DispersiveResult:
        signals, freqs, _ = load_data(filepath, **kwargs)
        assert len(freqs.shape) == 1
        assert signals.shape == (len(freqs), 2)

        freqs = freqs * 1e-6  # Hz -> MHz
        signals = signals.T  # transpose back

        freqs = freqs.astype(np.float64)
        signals = signals.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (freqs, signals)

        return freqs, signals
