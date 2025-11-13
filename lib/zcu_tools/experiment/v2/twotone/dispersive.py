from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program.v2 import Pulse, Readout, TwoToneProgram, sweep2param
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.fitting.resonance import fit_edelay, get_proper_model

from ..runner import HardTask, Runner

DispersiveResultType = Tuple[np.ndarray, np.ndarray]


def dispersive_signal2real(signals: np.ndarray) -> np.ndarray:
    return np.abs(signals)


class DispersiveExperiment(AbsExperiment[DispersiveResultType]):
    def run(
        self, soc, soccfg, cfg: Dict[str, Any], *, progress: bool = True
    ) -> DispersiveResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        # Canonicalise sweep section to single-axis form
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "freq")
        cfg["sweep"] = {
            "ge": {"start": 0, "stop": 1.0, "expts": 2},
            "freq": cfg["sweep"]["freq"],
        }

        fpts = sweep2array(cfg["sweep"]["freq"])  # predicted frequency points

        # Set with/without π gain for qubit pulse
        Pulse.set_param(
            cfg["qub_pulse"], "on/off", sweep2param("ge", cfg["sweep"]["ge"])
        )
        Readout.set_param(
            cfg["readout"], "freq", sweep2param("freq", cfg["sweep"]["freq"])
        )

        with LivePlotter1D(
            "Frequency (MHz)",
            "Amplitude",
            segment_kwargs=dict(num_lines=2),
            disable=not progress,
        ) as viewer:
            signals = Runner(
                task=HardTask(
                    measure_fn=lambda ctx, update_hook: (
                        TwoToneProgram(soccfg, ctx.cfg).acquire(
                            soc, progress=False, callback=update_hook
                        )
                    ),
                    result_shape=(2, len(fpts)),
                ),
                update_hook=lambda ctx: viewer.update(
                    fpts, dispersive_signal2real(np.asarray(ctx.get_data()))
                ),
            ).run(cfg)
            signals = np.asarray(signals)

        # Cache results
        self.last_cfg = cfg
        self.last_result = (fpts, signals)

        return fpts, signals

    def analyze(
        self, result: Optional[DispersiveResultType] = None
    ) -> Tuple[float, float, plt.Figure]:
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

        g_fit = np.abs(model.calc_signals(fpts, **g_params))
        e_fit = np.abs(model.calc_signals(fpts, **e_params))

        # Calculate dispersive shift and average linewidth
        chi = abs(g_freq - e_freq) / 2  # dispersive shift χ/2π
        avg_kappa = (g_kappa + e_kappa) / 2  # average linewidth κ/2π

        fig, ax = plt.subplots(figsize=config.figsize)
        assert isinstance(fig, plt.Figure)

        # Plot data and fits
        ax.plot(fpts, g_amps, marker=".", c="b", label="Ground state")
        ax.plot(fpts, e_amps, marker=".", c="r", label="Excited state")
        ax.plot(fpts, g_fit, "b-", alpha=0.7)
        ax.plot(fpts, e_fit, "r-", alpha=0.7)

        # Mark resonance frequencies
        label_g = f"Ground: {g_freq:.1f} MHz, κ = {g_kappa:.1f} MHz"
        label_e = f"Excited: {e_freq:.1f} MHz, κ = {e_kappa:.1f} MHz"
        ax.axvline(g_freq, color="b", ls="--", alpha=0.7, label=label_g)
        ax.axvline(e_freq, color="r", ls="--", alpha=0.7, label=label_e)

        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Amplitude (a.u.)")
        ax.set_title(
            f"Dispersive shift χ/2π = {chi:.3f} MHz, κ/2π = {avg_kappa:.1f} MHz"
        )
        ax.legend()
        ax.grid(True)

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
