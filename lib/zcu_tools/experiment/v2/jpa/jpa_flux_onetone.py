from __future__ import annotations

from copy import deepcopy
from typing import Mapping, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from zcu_tools.device import DeviceInfo
from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import set_flux_in_dev_cfg, sweep2array
from zcu_tools.experiment.v2.runner import HardTask, SoftTask, TaskConfig, run_task
from zcu_tools.liveplot import LivePlotter2DwithLine
from zcu_tools.program.v2 import OneToneProgram, OneToneProgramCfg, Readout, sweep2param
from zcu_tools.utils.datasaver import load_data, save_data

JPAFluxResultType = Tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


class JPAFluxByOneToneTaskConfig(TaskConfig, OneToneProgramCfg):
    dev: Mapping[str, DeviceInfo]


class JPAFluxByOneToneExperiment(AbsExperiment):
    def run(self, soc, soccfg, cfg: JPAFluxByOneToneTaskConfig) -> JPAFluxResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        assert "sweep" in cfg
        jpa_flux_sweep = cfg["sweep"]["jpa_flux"]
        cfg["sweep"] = {"freq": cfg["sweep"]["freq"]}

        jpa_flxs = sweep2array(jpa_flux_sweep, allow_array=True)
        fpts = sweep2array(cfg["sweep"]["freq"], allow_array=True)

        Readout.set_param(
            cfg["readout"], "freq", sweep2param("freq", cfg["sweep"]["freq"])
        )

        with LivePlotter2DwithLine(
            "JPA Flux value (a.u.)",
            "Readout frequency (MHz)",
            line_axis=1,
            num_lines=5,
        ) as viewer:
            signals = run_task(
                task=SoftTask(
                    sweep_name="JPA Flux value",
                    sweep_values=jpa_flxs.tolist(),
                    update_cfg_fn=lambda i, ctx, flx: set_flux_in_dev_cfg(
                        ctx.cfg["dev"], flx, label="jpa_flux_dev"
                    ),
                    sub_task=HardTask(
                        measure_fn=lambda ctx, update_hook: (
                            OneToneProgram(soccfg, ctx.cfg).acquire(
                                soc,
                                progress=False,
                                callback=update_hook,
                                record_stderr=True,
                            )
                        ),
                        result_shape=(len(fpts),),
                    ),
                ),
                init_cfg=cfg,
                update_hook=lambda ctx: viewer.update(
                    jpa_flxs, fpts, np.abs(np.asarray(ctx.data))
                ),
            )
            signals = np.asarray(signals)

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (jpa_flxs, fpts, signals)

        return jpa_flxs, fpts, signals

    def analyze(self, result: Optional[JPAFluxResultType] = None) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        jpa_flxs, fpts, signals = result

        raise NotImplementedError("analysis not implemented yet")

    def save(
        self,
        filepath: str,
        result: Optional[JPAFluxResultType] = None,
        comment: Optional[str] = None,
        tag: str = "jpa/flux_onetone",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        jpa_flxs, fpts, signals = result

        save_data(
            filepath=filepath,
            x_info={"name": "JPA Flux value", "unit": "a.u.", "values": jpa_flxs},
            y_info={"name": "Readout frequency", "unit": "Hz", "values": fpts * 1e6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> JPAFluxResultType:
        signals, jpa_flxs, fpts = load_data(filepath, **kwargs)
        assert jpa_flxs is not None and fpts is not None
        assert len(jpa_flxs.shape) == 1 and len(fpts.shape) == 1
        assert signals.shape == (len(fpts), len(jpa_flxs))

        fpts = fpts * 1e-6  # Hz -> MHz
        signals = signals.T  # transpose back

        jpa_flxs = jpa_flxs.astype(np.float64)
        fpts = fpts.astype(np.float64)
        signals = signals.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (jpa_flxs, fpts, signals)

        return jpa_flxs, fpts, signals
