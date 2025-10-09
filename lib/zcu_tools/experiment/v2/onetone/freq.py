from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program.v2 import OneToneProgram, set_readout_cfg, sweep2param
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.fitting import fit_resonence_freq

from ..runner import HardTask, Runner

FreqResultType = Tuple[np.ndarray, np.ndarray]


def freq_signal2real(signals: np.ndarray) -> np.ndarray:
    return np.abs(signals)


class FreqExperiment(AbsExperiment[FreqResultType]):
    def run(
        self, soc, soccfg, cfg: Dict[str, Any], *, progress: bool = True
    ) -> FreqResultType:
        cfg = deepcopy(cfg)

        # Ensure the sweep section is in canonical single-axis form.
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "freq")

        # Predicted frequency points (before mapping to ADC domain)
        fpts = sweep2array(cfg["sweep"]["freq"])  # MHz

        # set readout frequency as sweep param
        set_readout_cfg(
            cfg["readout"], "freq", sweep2param("freq", cfg["sweep"]["freq"])
        )

        # run experiment
        with LivePlotter1D(
            "Frequency (MHz)", "Amplitude", disable=not progress
        ) as viewer:
            signals = Runner(
                task=HardTask(
                    measure_fn=lambda ctx, update_hook: (
                        OneToneProgram(soccfg, ctx.cfg).acquire(
                            soc, progress=progress, callback=update_hook
                        )
                    ),
                    result_shape=(len(fpts),),
                ),
                update_hook=lambda ctx: viewer.update(
                    fpts, freq_signal2real(np.asarray(ctx.get_data()))
                ),
            ).run(cfg)
            signals = np.asarray(signals)

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (fpts, signals)

        return fpts, signals

    def analyze_by_abcd(
        self,
        fpts: np.ndarray,
        signals: np.ndarray,
        solve_type: str = "hm",
        fit_edelay: bool = True,
    ) -> Dict[str, float]:
        try:
            from abcd_rf_fit import analyze
        except ImportError:
            print(
                "cannot import abcd_rf_fit, do you have it installed? please check: <https://github.com/UlysseREGLADE/abcd_rf_fit.git>"
            )
            raise

        fit = analyze(1e6 * fpts, signals, solve_type, fit_edelay=fit_edelay)
        fit.plot()
        param = fit.tolist()
        return {
            "freq": round(param[0] * 1e-6, 7),  # MHz
            "kappa": round(param[1] * 1e-6, 4),  # MHz
            "Qi": round((param[0] / (param[1] - param[2]))),
            "absQc": round(param[0] / param[2]),
            "Ql": round(param[0] / param[1]),
        }

    def analyze_wo_abcd(
        self,
        fpts: np.ndarray,
        signals: np.ndarray,
        *,
        type: Literal["lor", "sinc"] = "lor",
        asym: bool = False,
    ) -> Dict[str, float]:
        val_mask = ~np.isnan(signals)
        fpts = fpts[val_mask]
        signals = signals[val_mask]

        amps = np.abs(signals)

        freq, freq_err, kappa, _, y_fit, _ = fit_resonence_freq(fpts, amps, type, asym)

        plt.figure(figsize=config.figsize)
        plt.tight_layout()
        plt.plot(fpts, amps, label="signal", marker="o", markersize=3)
        plt.plot(fpts, y_fit, label=f"fit, $kappa$={kappa:.1g} MHz")
        label = f"$f_res$ = {freq:.7g} +/- {freq_err:.1g} MHz"
        plt.axvline(freq, color="r", ls="--", label=label)
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Magnitude (a.u.)")
        plt.legend()
        plt.show()

        return dict(freq=freq, kappa=kappa, freq_err=freq_err)

    def analyze(
        self,
        result: Optional[FreqResultType] = None,
        *,
        use_abcd: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        fpts, signals = result

        if use_abcd:
            params = self.analyze_by_abcd(fpts=fpts, signals=signals, **kwargs)
        else:
            params = self.analyze_wo_abcd(fpts=fpts, signals=signals, **kwargs)

        return params["freq"], params["kappa"], params

    def save(
        self,
        filepath: str,
        result: Optional[FreqResultType] = None,
        comment: Optional[str] = None,
        tag: str = "onetone/freq",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        fpts, signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "Frequency", "unit": "MHz", "values": fpts},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )
