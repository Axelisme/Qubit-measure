from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import numpy as np

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import sweep2array, set_freq_in_dev_cfg, format_sweep1D
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program.v2 import OneToneProgram
from zcu_tools.utils.datasaver import save_data

from ..runner import HardTask, Runner, SoftTask

JPAFreqResultType = Tuple[np.ndarray, np.ndarray]


def jpa_freq_signal2real(signals: np.ndarray) -> np.ndarray:
    return np.abs(signals)


class JPAFreqExperiment(AbsExperiment[JPAFreqResultType]):
    def run(self, soc, soccfg, cfg: Dict[str, Any]) -> JPAFreqResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        cfg["sweep"] = format_sweep1D(cfg["sweep"], "jpa_freq")
        jpa_freq_sweep = cfg["sweep"]["jpa_freq"]

        # remove sweep dict
        del cfg["sweep"]

        jpa_freqs = sweep2array(jpa_freq_sweep, allow_array=True)

        with LivePlotter1D("JPA Frequency (MHz)", "Magnitude") as viewer:
            signals = Runner(
                task=SoftTask(
                    sweep_name="JPA Frequency",
                    sweep_values=jpa_freqs,
                    update_cfg_fn=lambda i, ctx, freq: set_freq_in_dev_cfg(
                        ctx.cfg["dev"], freq * 1e6, label="jpa_rf_dev"
                    ),
                    sub_task=HardTask(
                        measure_fn=lambda ctx, update_hook: (
                            OneToneProgram(soccfg, ctx.cfg).acquire(
                                soc, progress=False, callback=update_hook
                            )
                        ),
                    ),
                ),
                update_hook=lambda ctx: viewer.update(
                    jpa_freqs, jpa_freq_signal2real(np.asarray(ctx.get_data()))
                ),
            ).run(cfg)
            signals = np.asarray(signals)  # type: ignore

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (jpa_freqs, signals)

        return jpa_freqs, signals

    def analyze(self, result: Optional[JPAFreqResultType] = None) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        jpa_freqs, signals = result

        raise NotImplementedError("analysis not yet implemented")

    def save(
        self,
        filepath: str,
        result: Optional[JPAFreqResultType] = None,
        comment: Optional[str] = None,
        tag: str = "jpa/freq",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        jpa_freqs, signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "JPA Frequency", "unit": "Hz", "values": jpa_freqs * 1e6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )
