from __future__ import annotations

from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from typing_extensions import NotRequired

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import (
    format_sweep1D,
    make_ge_sweep,
    set_freq_in_dev_cfg,
    sweep2array,
)
from zcu_tools.experiment.v2.runner import HardTask, SoftTask, TaskConfig, run_task
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program.v2 import (
    ModularProgramCfg,
    ModularProgramV2,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Reset,
    ResetCfg,
    sweep2param,
)
from zcu_tools.utils.datasaver import save_data

JPAFreqGEResultType = Tuple[NDArray[np.float64], NDArray[np.complex128]]


def jpa_freq_ge_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(signals[..., 0] - signals[..., 1])  # (freq, )


class JPAFreqGETaskConfig(TaskConfig, ModularProgramCfg):
    reset: NotRequired[ResetCfg]
    pi_pulse: PulseCfg
    readout: ReadoutCfg


class JPAFreqGEExperiment(AbsExperiment):
    def run(self, soc, soccfg, cfg: JPAFreqGETaskConfig) -> JPAFreqGEResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        assert "sweep" in cfg
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "jpa_freq")

        jpa_freqs = sweep2array(cfg["sweep"]["jpa_freq"], allow_array=True)

        cfg["sweep"] = {"ge": make_ge_sweep()}
        Pulse.set_param(
            cfg["pi_pulse"], "on/off", sweep2param("ge", cfg["sweep"]["ge"])
        )

        with LivePlotter1D("JPA Frequency (MHz)", "Signal Difference") as viewer:
            signals = run_task(
                task=SoftTask(
                    sweep_name="JPA Frequency",
                    sweep_values=jpa_freqs.tolist(),
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
                            ).acquire(soc, progress=False, callback=update_hook)
                        ),
                        result_shape=(2,),
                    ),
                ),
                init_cfg=cfg,
                update_hook=lambda ctx: viewer.update(
                    jpa_freqs, jpa_freq_ge_signal2real(np.asarray(ctx.data))
                ),
            )
            signals = np.asarray(signals)

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
