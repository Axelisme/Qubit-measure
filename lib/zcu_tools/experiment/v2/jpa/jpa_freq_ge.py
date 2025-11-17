from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import numpy as np

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import (
    format_sweep1D,
    make_ge_sweep,
    set_freq_in_dev_cfg,
    sweep2array,
)
from zcu_tools.experiment.v2.runner import HardTask, Runner, SoftTask
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program.v2 import ModularProgramV2, Pulse, Readout, Reset, sweep2param
from zcu_tools.utils.datasaver import save_data

JPAFreqGEResultType = Tuple[np.ndarray, np.ndarray]


def jpa_freq_ge_signal2real(signals: np.ndarray) -> np.ndarray:
    # signals: (freq, ge)
    return np.abs(signals[..., 0] - signals[..., 1])  # (freq, )


class JPAFreqGEExperiment(AbsExperiment[JPAFreqGEResultType]):
    def run(self, soc, soccfg, cfg: Dict[str, Any]) -> JPAFreqGEResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        cfg["sweep"] = format_sweep1D(cfg["sweep"], "jpa_freq")

        jpa_freqs = sweep2array(cfg["sweep"]["jpa_freq"], allow_array=True)

        cfg["sweep"] = {"ge": make_ge_sweep()}
        Pulse.set_param(
            cfg["pi_pulse"], "on/off", sweep2param("ge", cfg["sweep"]["ge"])
        )

        with LivePlotter1D("JPA Frequency (MHz)", "Signal Difference") as viewer:
            signals = Runner(
                task=SoftTask(
                    sweep_name="JPA Frequency",
                    sweep_values=jpa_freqs,
                    update_cfg_fn=lambda i, ctx, freq: set_freq_in_dev_cfg(
                        ctx.cfg["dev"], freq * 1e6, label="jpa_rf_dev"
                    ),
                    sub_task=HardTask(
                        measure_fn=lambda ctx, update_hook: (
                            ModularProgramV2(
                                soccfg,
                                ctx.cfg,
                                modules=[
                                    Reset(
                                        "reset", ctx.cfg.get("reset", {"type": "none"})
                                    ),
                                    Pulse("pi_pulse", ctx.cfg["pi_pulse"]),
                                    Readout("readout", ctx.cfg["readout"]),
                                ],
                            )
                        ),
                        result_shape=(2,),
                    ),
                ),
                update_hook=lambda ctx: viewer.update(
                    jpa_freqs, jpa_freq_ge_signal2real(np.asarray(ctx.get_data()))
                ),
            ).run(cfg)
            signals = np.asarray(signals)  # type: ignore

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (jpa_freqs, signals)

        return jpa_freqs, signals

    def analyze(self, result: Optional[JPAFreqGEResultType] = None) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        jpa_freqs, signals = result

        raise NotImplementedError("analysis not yet implemented")

    def save(
        self,
        filepath: str,
        result: Optional[JPAFreqGEResultType] = None,
        comment: Optional[str] = None,
        tag: str = "jpa/freq_ge",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        jpa_freqs, signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "JPA Frequency", "unit": "Hz", "values": jpa_freqs * 1e6},
            y_info={"name": "GE", "unit": "None", "values": np.array([0, 1])},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )
