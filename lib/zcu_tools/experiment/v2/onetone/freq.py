from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program.v2 import OneToneProgram, Readout, sweep2param
from zcu_tools.utils.datasaver import save_data
from zcu_tools.utils.fitting import HangerModel, TransmissionModel, get_proper_model

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
        Readout.set_param(
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
                            soc, progress=False, callback=update_hook
                        )
                    ),
                    result_shape=(len(fpts),),
                ),
                update_hook=lambda ctx: viewer.update(
                    fpts,
                    freq_signal2real(np.asarray(ctx.get_data())),  # type: ignore
                ),
            ).run(cfg)
            signals = np.asarray(signals)  # type: ignore

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (fpts, signals)

        return fpts, signals

    def analyze(
        self,
        result: Optional[FreqResultType] = None,
        *,
        model_type: Literal["hm", "t", "auto"] = "auto",
        edelay: Optional[float] = None,
        plot: bool = True,
    ) -> Tuple[float, float, Dict[str, Any], Optional[plt.Figure]]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        fpts, signals = result

        # remove first and last point, sometimes they have problems
        fpts = fpts[1:-1]
        signals = signals[1:-1]

        # discard NaNs (possible early abort)
        val_mask = ~np.isnan(signals)  # type: ignore
        fpts = fpts[val_mask]
        signals = signals[val_mask]

        if model_type == "hm":
            model = HangerModel()
        elif model_type == "t":
            model = TransmissionModel()
        elif model_type == "auto":
            model = get_proper_model(fpts, signals)
        else:
            raise ValueError(f"Invalid model type: {model_type}")

        param_dict = model.fit(fpts, signals, edelay)
        if plot:
            fig = model.visualize_fit(fpts, signals, param_dict)
        else:
            fig = None

        return float(param_dict["freq"]), float(param_dict["kappa"]), param_dict, fig

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
            x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e-6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )
