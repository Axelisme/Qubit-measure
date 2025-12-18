from __future__ import annotations

from copy import deepcopy
from typing import Mapping, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from zcu_tools.device import DeviceInfo
from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import (
    format_sweep1D,
    set_output_in_dev_cfg,
    sweep2array,
)
from zcu_tools.experiment.v2.runner import HardTask, SoftTask, TaskConfig, run_task
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program.v2 import OneToneProgram, OneToneProgramCfg, Readout, sweep2param
from zcu_tools.utils.datasaver import load_data, save_data

JPACheckResultType = Tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


def jpa_check_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(signals)


class JPACheckTaskConfig(TaskConfig, OneToneProgramCfg):
    dev: Mapping[str, DeviceInfo]


class JPACheckExperiment(AbsExperiment):
    OUTPUT_MAP = {0: "off", 1: "on"}

    def run(self, soc, soccfg, cfg: JPACheckTaskConfig) -> JPACheckResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        assert "sweep" in cfg
        cfg["sweep"] = format_sweep1D(cfg["sweep"], "freq")

        fpts = sweep2array(cfg["sweep"]["freq"])  # predicted frequency points

        Readout.set_param(
            cfg["readout"], "freq", sweep2param("freq", cfg["sweep"]["freq"])
        )

        outputs = np.array([0, 1])

        with LivePlotter1D(
            "Frequency (MHz)", "Magnitude", segment_kwargs=dict(num_lines=2)
        ) as viewer:
            signals = run_task(
                task=SoftTask(
                    sweep_name="JPA on/off",
                    sweep_values=outputs.tolist(),
                    update_cfg_fn=lambda i, ctx, output: set_output_in_dev_cfg(
                        ctx.cfg["dev"],
                        self.OUTPUT_MAP[output],  # type: ignore
                        label="jpa_rf_dev",
                    ),
                    sub_task=HardTask(
                        measure_fn=lambda ctx, update_hook: (
                            OneToneProgram(soccfg, ctx.cfg).acquire(
                                soc, progress=False, callback=update_hook
                            )
                        ),
                        result_shape=(len(fpts),),
                    ),
                ),
                init_cfg=cfg,
                update_hook=lambda ctx: viewer.update(
                    fpts, jpa_check_signal2real(np.asarray(ctx.data))
                ),
            )
            signals = np.asarray(signals)

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (outputs, fpts, signals)

        return outputs, fpts, signals

    def analyze(self, result: Optional[JPACheckResultType] = None) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        outputs, fpts, signals2D = result

        real_signals = jpa_check_signal2real(signals2D)

        fig, ax = plt.subplots(figsize=config.figsize)
        for i, output in enumerate(outputs):
            ax.plot(
                fpts,
                real_signals[i, :],
                label=f"JPA {self.OUTPUT_MAP[output]}",
                marker="o",
                markersize=4,
                linestyle="-",
            )

        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Signal Magnitude (a.u.)")
        ax.legend()
        ax.grid(True)

        return fig

    def save(
        self,
        filepath: str,
        result: Optional[JPACheckResultType] = None,
        comment: Optional[str] = None,
        tag: str = "jpa/check",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        outputs, fpts, signals2D = result

        save_data(
            filepath=filepath,
            x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
            y_info={"name": "JPA Output", "unit": "a.u.", "values": outputs},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals2D},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> JPACheckResultType:
        signals2D, fpts, outputs = load_data(filepath, **kwargs)
        assert fpts is not None and outputs is not None
        assert len(fpts.shape) == 1 and len(outputs.shape) == 1
        assert signals2D.shape == (len(outputs), len(fpts))

        fpts = fpts * 1e-6  # Hz -> MHz

        outputs = outputs.astype(np.float64)
        fpts = fpts.astype(np.float64)
        signals2D = signals2D.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (outputs, fpts, signals2D)

        return outputs, fpts, signals2D
