from __future__ import annotations

from copy import deepcopy

import numpy as np
from numpy.typing import NDArray
from typeguard import check_type
from typing_extensions import Any, Mapping, Optional, TypeAlias

from zcu_tools.device import DeviceInfo
from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import set_flux_in_dev_cfg, sweep2array
from zcu_tools.experiment.v2.runner import Task, TaskCfg, run_task
from zcu_tools.liveplot import LivePlotter2DwithLine
from zcu_tools.notebook.analysis.fluxdep.interactive import InteractiveLines
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import OneToneCfg, OneToneProgram, Readout, sweep2param
from zcu_tools.utils.datasaver import load_data, save_data

FluxDepResult: TypeAlias = tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


def fluxdep_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(signals)


class FluxDepCfg(OneToneCfg, TaskCfg):
    dev: Mapping[str, DeviceInfo]
    sweep: dict[str, SweepCfg]


class FluxDepExp(AbsExperiment[FluxDepResult, FluxDepCfg]):
    def run(self, soc, soccfg, cfg: dict[str, Any]) -> FluxDepResult:
        _cfg = check_type(deepcopy(cfg), FluxDepCfg)

        fpt_sweep = _cfg["sweep"]["freq"]
        flx_sweep = _cfg["sweep"]["flux"]

        # remove flux from sweep dict, will be handled by soft loop
        _cfg["sweep"] = {"freq": fpt_sweep}

        dev_values: NDArray[np.float64] = sweep2array(flx_sweep, allow_array=True)
        fpts = sweep2array(fpt_sweep)  # predicted frequency points

        modules = _cfg["modules"]
        Readout.set_param(modules["readout"], "freq", sweep2param("freq", fpt_sweep))

        with LivePlotter2DwithLine(
            "Flux device value", "Frequency (MHz)", line_axis=1, num_lines=10
        ) as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=lambda ctx, update_hook: OneToneProgram(
                        soccfg, ctx.cfg
                    ).acquire(soc, progress=False, callback=update_hook),
                    result_shape=(len(fpts),),
                ).scan(
                    "flux",
                    dev_values.tolist(),
                    before_each=lambda i, ctx, flx: set_flux_in_dev_cfg(
                        ctx.cfg["dev"], flx
                    ),
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    dev_values,
                    fpts,
                    fluxdep_signal2real(np.asarray(ctx.root_data)),
                ),
            )
            signals = np.asarray(signals)

        # record last cfg and result
        self.last_cfg = _cfg
        self.last_result = (dev_values, fpts, signals)

        return dev_values, fpts, signals

    def analyze(
        self,
        result: Optional[FluxDepResult] = None,
        mA_c: Optional[float] = None,
        mA_e: Optional[float] = None,
    ) -> InteractiveLines:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        values, fpts, signals2D = result

        actline = InteractiveLines(
            signals2D,
            mAs=values,
            fpts=fpts,
            mA_c=mA_c,
            mA_e=mA_e,
        )

        return actline

    def save(
        self,
        filepath: str,
        result: Optional[FluxDepResult] = None,
        comment: Optional[str] = None,
        tag: str = "onetone/flux_dep",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        values, fpts, signals2D = result

        save_data(
            filepath=filepath,
            x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
            y_info={"name": "Flux device value", "unit": "a.u.", "values": values},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals2D},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> FluxDepResult:
        signals2D, fpts, values = load_data(filepath, **kwargs)
        assert fpts is not None and values is not None
        assert len(fpts.shape) == 1 and len(values.shape) == 1
        assert signals2D.shape == (len(fpts), len(values))

        fpts = fpts * 1e-6  # Hz -> MHz
        signals2D = signals2D.T  # transpose back

        values = values.astype(np.float64)
        fpts = fpts.astype(np.float64)
        signals2D = signals2D.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (values, fpts, signals2D)

        return values, fpts, signals2D
