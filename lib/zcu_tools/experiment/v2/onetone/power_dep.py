from __future__ import annotations

from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import sweep2array
from zcu_tools.liveplot import LivePlotter2DwithLine
from zcu_tools.program.v2 import OneToneProgram, OneToneProgramCfg, Readout, sweep2param
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.process import minus_background, rescale

from ..runner import HardTask, SoftTask, TaskConfig, run_task
from ..utils import wrap_earlystop_check

PowerDepResultType = Tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


def pdrdep_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rescale(minus_background(np.abs(signals), axis=1), axis=1)


class PowerDepTaskConfig(TaskConfig, OneToneProgramCfg): ...


class PowerDepExp(AbsExperiment[PowerDepResultType, PowerDepTaskConfig]):
    def run(
        self,
        soc,
        soccfg,
        cfg: PowerDepTaskConfig,
        *,
        earlystop_snr: Optional[float] = None,
    ) -> PowerDepResultType:
        cfg = deepcopy(cfg)  # prevent in-place modification

        assert "sweep" in cfg
        assert isinstance(cfg["sweep"], dict)
        pdr_sweep = cfg["sweep"].pop("gain")

        pdrs = sweep2array(pdr_sweep, allow_array=True)
        fpts = sweep2array(cfg["sweep"]["freq"])  # predicted frequency points

        Readout.set_param(
            cfg["readout"], "freq", sweep2param("freq", cfg["sweep"]["freq"])
        )

        # run experiment
        with LivePlotter2DwithLine(
            "Power (a.u.)", "Frequency (MHz)", line_axis=1, num_lines=10
        ) as viewer:
            ax1d = viewer.get_ax("1d")

            signals = run_task(
                task=SoftTask(
                    sweep_name="gain",
                    sweep_values=pdrs.tolist(),
                    update_cfg_fn=lambda _, ctx, pdr: (
                        Readout.set_param(ctx.cfg["readout"], "gain", pdr)
                    ),
                    sub_task=HardTask(
                        measure_fn=lambda ctx, update_hook: (
                            prog := OneToneProgram(soccfg, ctx.cfg)
                        ).acquire(
                            soc,
                            progress=False,
                            callback=wrap_earlystop_check(
                                prog,
                                update_hook,
                                earlystop_snr,
                                signal2real_fn=np.abs,
                                snr_hook=lambda snr: ax1d.set_title(f"snr = {snr:.1f}"),
                            ),
                        ),
                        result_shape=(len(fpts),),
                    ),
                ),
                init_cfg=cfg,
                update_hook=lambda ctx: viewer.update(
                    pdrs, fpts, pdrdep_signal2real(np.asarray(ctx.data))
                ),
            )
            signals = np.asarray(signals)

        # record last cfg and result
        self.last_cfg = cfg
        self.last_result = (pdrs, fpts, signals)

        return pdrs, fpts, signals

    def analyze(
        self,
        result: Optional[PowerDepResultType] = None,
    ) -> None:
        raise NotImplementedError("Not implemented")

    def save(
        self,
        filepath: str,
        result: Optional[PowerDepResultType] = None,
        comment: Optional[str] = None,
        tag: str = "onetone/power_dep",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        pdrs, fpts, signals2D = result

        save_data(
            filepath=filepath,
            x_info={"name": "Frequency", "unit": "Hz", "values": fpts * 1e6},
            y_info={"name": "Power", "unit": "a.u.", "values": pdrs},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals2D},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> PowerDepResultType:
        signals2D, fpts, pdrs = load_data(filepath, **kwargs)
        assert fpts is not None and pdrs is not None
        assert len(fpts.shape) == 1 and len(pdrs.shape) == 1
        assert signals2D.shape == (len(pdrs), len(fpts))

        fpts = fpts * 1e-6  # Hz -> MHz

        pdrs = pdrs.astype(np.float64)
        fpts = fpts.astype(np.float64)
        signals2D = signals2D.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (pdrs, fpts, signals2D)

        return pdrs, fpts, signals2D
