from copy import deepcopy
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from numpy.typing import NDArray

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import format_sweep1D, make_ge_sweep, sweep2array
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program.v2 import (
    Pulse,
    Readout,
    TwoToneProgram,
    TwoToneProgramCfg,
    sweep2param,
)
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.fitting.resonance import (
    fit_edelay,
    get_proper_model,
    normalize_signal,
    remove_edelay,
)

from ..runner import HardTask, TaskConfig, run_task

DispersiveResultType = Tuple[NDArray[np.float64], NDArray[np.complex128]]


def dispersive_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(signals)


class DispersiveTaskConfig(TaskConfig, TwoToneProgramCfg): ...


class DispersiveExp(AbsExperiment[DispersiveResultType, DispersiveTaskConfig]):
    def run(self, soc, soccfg, cfg: DispersiveTaskConfig) -> DispersiveResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        # Canonicalise sweep section to single-axis form
        assert "sweep" in cfg
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "freq")
        cfg["sweep"] = {"ge": make_ge_sweep(), "freq": cfg["sweep"]["freq"]}

        fpts = sweep2array(cfg["sweep"]["freq"])  # predicted frequency points

        # Set with/without π gain for qubit pulse
        Pulse.set_param(
            cfg["qub_pulse"], "on/off", sweep2param("ge", cfg["sweep"]["ge"])
        )
        Readout.set_param(
            cfg["readout"], "freq", sweep2param("freq", cfg["sweep"]["freq"])
        )

        with LivePlotter1D(
            "Frequency (MHz)", "Amplitude", segment_kwargs=dict(num_lines=2)
        ) as viewer:
            signals = run_task(
                task=HardTask(
                    measure_fn=lambda ctx, update_hook: (
                        TwoToneProgram(soccfg, ctx.cfg).acquire(
                            soc, progress=False, callback=update_hook
                        )
                    ),
                    result_shape=(2, len(fpts)),
                ),
                init_cfg=cfg,
                update_hook=lambda ctx: viewer.update(
                    fpts, dispersive_signal2real(ctx.data)
                ),
            )

        # Cache results
        self.last_cfg = cfg
        self.last_result = (fpts, signals)

        return fpts, signals

    def analyze(
        self, result: Optional[DispersiveResultType] = None
    ) -> Tuple[float, float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        fpts, signals = result
        g_signals, e_signals = signals[0, :], signals[1, :]
        g_amps, e_amps = np.abs(g_signals), np.abs(e_signals)

        g_edelay = fit_edelay(fpts, g_signals)
        e_edelay = fit_edelay(fpts, e_signals)
        edelay = 0.5 * (g_edelay + e_edelay)

        model = get_proper_model(fpts, g_signals)
        g_params = model.fit(fpts, g_signals, edelay=edelay)
        e_params = model.fit(fpts, e_signals, edelay=edelay)

        g_freq, g_kappa = g_params["freq"], g_params["kappa"]
        e_freq, e_kappa = e_params["freq"], e_params["kappa"]

        g_fit = np.abs(model.calc_signals(fpts, **g_params))  # type: ignore
        e_fit = np.abs(model.calc_signals(fpts, **e_params))  # type: ignore

        # Calculate dispersive shift and average linewidth
        chi = abs(g_freq - e_freq) / 2  # dispersive shift χ/2π
        avg_kappa = (g_kappa + e_kappa) / 2  # average linewidth κ/2π

        fig = plt.figure(figsize=(10, 4))
        spec = fig.add_gridspec(2, 3)
        ax_main = fig.add_subplot(spec[:, :2])
        ax_g = fig.add_subplot(spec[0, 2])
        ax_e = fig.add_subplot(spec[1, 2])

        # Plot data and fits
        ax_main.plot(fpts, g_amps, marker=".", c="b", label="Ground state")
        ax_main.plot(fpts, e_amps, marker=".", c="r", label="Excited state")
        ax_main.plot(fpts, g_fit, "b-", alpha=0.7)
        ax_main.plot(fpts, e_fit, "r-", alpha=0.7)

        # Mark resonance frequencies
        label_g = f"Ground: {g_freq:.1f} MHz, κ = {g_kappa:.1f} MHz"
        label_e = f"Excited: {e_freq:.1f} MHz, κ = {e_kappa:.1f} MHz"
        ax_main.axvline(g_freq, color="b", ls="--", alpha=0.7, label=label_g)
        ax_main.axvline(e_freq, color="r", ls="--", alpha=0.7, label=label_e)

        ax_main.set_xlabel("Frequency (MHz)")
        ax_main.set_ylabel("Amplitude (a.u.)")
        ax_main.set_title(
            f"Dispersive shift χ/2π = {chi:.3f} MHz, κ/2π = {avg_kappa:.1f} MHz"
        )
        ax_main.legend()
        ax_main.grid(True)

        def _plot_circle_fit(ax: Axes, signals: NDArray, params_dict: dict, color: str):
            rot_signals = remove_edelay(fpts, signals, edelay)
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
            )
            ax.add_patch(Circle((norm_xc, norm_yc), norm_r0, fill=False, color=color))
            ax.plot([norm_xc, 1], [norm_yc, 0], "kx--")
            ax.axhline(0, color="k", linestyle="--")
            ax.set_aspect("equal")
            ax.grid(True)
            ax.set_xlabel(r"$Re(S_{21})$")
            ax.set_ylabel(r"$Im(S_{21})$")

        # Plot individual circle fit
        _plot_circle_fit(ax_g, g_signals, dict(g_params), "b")
        _plot_circle_fit(ax_e, e_signals, dict(e_params), "r")
        ax_g.set_title("Circle fit (Ground)")
        ax_e.set_title("Circle fit (Excited)")

        fig.tight_layout()

        return chi, avg_kappa, fig

    def save(
        self,
        filepath: str,
        result: Optional[DispersiveResultType] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/dispersive",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        fpts, signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
            y_info={"name": "Amplitude", "unit": "None", "values": np.array([0, 1])},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> DispersiveResultType:
        signals, fpts, _ = load_data(filepath, **kwargs)
        assert len(fpts.shape) == 1
        assert signals.shape == (len(fpts), 2)

        fpts = fpts * 1e-6  # Hz -> MHz
        signals = signals.T  # transpose back

        fpts = fpts.astype(np.float64)
        signals = signals.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (fpts, signals)

        return fpts, signals
