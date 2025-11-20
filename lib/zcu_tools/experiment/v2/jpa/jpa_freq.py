from __future__ import annotations

from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import format_sweep1D, set_freq_in_dev_cfg, sweep2array
from zcu_tools.experiment.v2.runner import HardTask, SoftTask, TaskConfig, run_task
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program.v2 import OneToneProgram, OneToneProgramCfg
from zcu_tools.utils.datasaver import save_data

JPAFreqResultType = Tuple[NDArray[np.float64], NDArray[np.complex128]]


def jpa_freq_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(signals)


class JPAFreqTaskConfig(TaskConfig, OneToneProgramCfg): ...


class JPAFreqExperiment(AbsExperiment):
    def run(self, soc, soccfg, cfg: JPAFreqTaskConfig) -> JPAFreqResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        assert "sweep" in cfg
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "jpa_freq")
        jpa_freq_sweep = cfg["sweep"]["jpa_freq"]

        # remove sweep dict
        del cfg["sweep"]

        jpa_freqs = sweep2array(jpa_freq_sweep, allow_array=True)

        with LivePlotter1D("JPA Frequency (MHz)", "Magnitude") as viewer:
            signals = run_task(
                task=SoftTask(
                    sweep_name="JPA Frequency",
                    sweep_values=jpa_freqs.tolist(),
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
                init_cfg=cfg,
                update_hook=lambda ctx: viewer.update(
                    jpa_freqs, jpa_freq_signal2real(np.asarray(ctx.data))
                ),
            )
            signals = np.asarray(signals)

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
