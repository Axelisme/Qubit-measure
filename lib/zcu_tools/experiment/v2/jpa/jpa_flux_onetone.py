from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from typeguard import check_type

from zcu_tools.device import DeviceInfo
from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import set_flux_in_dev_cfg, sweep2array
from zcu_tools.experiment.v2.runner import Task, TaskCfg, run_task
from zcu_tools.liveplot import LivePlotter2DwithLine
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import OneToneCfg, OneToneProgram, Readout, sweep2param
from zcu_tools.utils.datasaver import load_data, save_data

JPAFluxByOneToneResult = Tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


class JPAFluxByOneToneCfg(OneToneCfg, TaskCfg):
    dev: Mapping[str, DeviceInfo]
    sweep: Dict[str, SweepCfg]


class JPAFluxByOneToneExp(AbsExperiment[JPAFluxByOneToneResult, JPAFluxByOneToneCfg]):
    def run(self, soc, soccfg, cfg: Dict[str, Any]) -> JPAFluxByOneToneResult:
        _cfg = check_type(deepcopy(cfg), JPAFluxByOneToneCfg)

        jpa_flux_sweep = _cfg["sweep"]["jpa_flux"]
        _cfg["sweep"] = {"freq": _cfg["sweep"]["freq"]}

        modules = _cfg["modules"]
        jpa_flxs = sweep2array(jpa_flux_sweep, allow_array=True)
        fpts = sweep2array(_cfg["sweep"]["freq"], allow_array=True)

        Readout.set_param(
            modules["readout"], "freq", sweep2param("freq", _cfg["sweep"]["freq"])
        )

        with LivePlotter2DwithLine(
            "JPA Flux value (a.u.)",
            "Readout frequency (MHz)",
            line_axis=1,
            num_lines=5,
        ) as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=lambda ctx, update_hook: OneToneProgram(
                        soccfg, ctx.cfg
                    ).acquire(soc, progress=False, callback=update_hook),
                    result_shape=(len(fpts),),
                ).scan(
                    "JPA Flux value",
                    jpa_flxs.tolist(),
                    before_each=lambda i, ctx, flx: set_flux_in_dev_cfg(
                        ctx.cfg["dev"], flx, label="jpa_flux_dev"
                    ),
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    jpa_flxs, fpts, np.abs(np.asarray(ctx.root_data))
                ),
            )
            signals = np.asarray(signals)

        # record last cfg and result
        self.last_cfg = _cfg
        self.last_result = (jpa_flxs, fpts, signals)

        return jpa_flxs, fpts, signals

    def analyze(self, result: Optional[JPAFluxByOneToneResult] = None) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        jpa_flxs, fpts, signals = result

        raise NotImplementedError("analysis not implemented yet")

    def save(
        self,
        filepath: str,
        result: Optional[JPAFluxByOneToneResult] = None,
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

    def load(self, filepath: str, **kwargs) -> JPAFluxByOneToneResult:
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
