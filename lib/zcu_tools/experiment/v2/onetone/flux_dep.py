from __future__ import annotations

from copy import deepcopy

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Callable, Mapping, Optional, TypeAlias, cast

from zcu_tools.config import ConfigBase
from zcu_tools.device import DeviceInfo
from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import (
    make_comment,
    parse_comment,
    set_flux_in_dev_cfg,
    setup_devices,
)
from zcu_tools.experiment.v2.runner import Task, TaskState, run_task
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot2DwithLine
from zcu_tools.notebook.analysis.fluxdep.interactive import InteractiveLines
from zcu_tools.program.v2 import (
    ModularProgramV2,
    ProgramV2Cfg,
    PulseReadout,
    PulseReadoutCfg,
    Reset,
    ResetCfg,
    SweepCfg,
    sweep2param,
)
from zcu_tools.utils.datasaver import load_data, save_data

FluxDepResult: TypeAlias = tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


def fluxdep_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(signals)


class FluxDepModuleCfg(ConfigBase):
    reset: Optional[ResetCfg] = None
    readout: PulseReadoutCfg


class FluxDepSweepCfg(ConfigBase):
    freq: SweepCfg
    flux: SweepCfg


class FluxDepCfg(ProgramV2Cfg, ExpCfgModel):
    modules: FluxDepModuleCfg
    dev: Mapping[str, DeviceInfo] = ...
    sweep: FluxDepSweepCfg


class FluxDepExp(AbsExperiment[FluxDepResult, FluxDepCfg]):
    def run(self, soc, soccfg, cfg: FluxDepCfg) -> FluxDepResult:
        modules = cfg.modules
        freq_sweep = cfg.sweep.freq
        flux_sweep = cfg.sweep.flux

        dev_values = sweep2array(flux_sweep, allow_array=True)
        freqs = sweep2array(
            freq_sweep,
            "freq",
            {
                "soccfg": soccfg,
                "gen_ch": modules.readout.pulse_cfg.ch,
                "ro_ch": modules.readout.ro_cfg.ro_ch,
            },
        )

        def measure_fn(
            ctx: TaskState, update_hook: Optional[Callable]
        ) -> list[NDArray[np.float64]]:
            cfg = cast(FluxDepCfg, ctx.cfg)
            setup_devices(cfg, progress=False)
            modules = cfg.modules

            freq_sweep = cfg.sweep.freq
            freq_param = sweep2param("freq", freq_sweep)
            modules.readout.set_param("freq", freq_param)

            return ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    Reset("reset", modules.reset),
                    PulseReadout("readout", modules.readout),
                ],
                sweep=[("freq", freq_sweep)],
            ).acquire(soc, progress=False, round_hook=update_hook)

        with LivePlot2DwithLine(
            "Flux device value", "Frequency (MHz)", line_axis=1, num_lines=10
        ) as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    result_shape=(len(freqs),),
                    pbar_n=cfg.rounds,
                ).scan(
                    "flux",
                    dev_values.tolist(),
                    before_each=lambda i, ctx, flux: set_flux_in_dev_cfg(
                        ctx.cfg.dev, flux
                    ),
                ),
                init_cfg=cfg,
                on_update=lambda ctx: viewer.update(
                    dev_values,
                    freqs,
                    fluxdep_signal2real(np.asarray(ctx.root_data)),
                ),
            )
            signals = np.asarray(signals)

        # record last cfg and result
        self.last_cfg = deepcopy(cfg)
        self.last_result = (dev_values, freqs, signals)

        return dev_values, freqs, signals

    def analyze(
        self,
        result: Optional[FluxDepResult] = None,
        flux_half: Optional[float] = None,
        flux_int: Optional[float] = None,
    ) -> InteractiveLines:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        values, freqs, signals2D = result

        actline = InteractiveLines(
            signals2D,
            dev_values=values,
            freqs=freqs,
            flux_half=flux_half,
            flux_int=flux_int,
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

        values, freqs, signals2D = result

        cfg = self.last_cfg
        assert cfg is not None
        comment = make_comment(cfg, comment)

        save_data(
            filepath=filepath,
            x_info={"name": "Flux device value", "unit": "a.u.", "values": values},
            y_info={"name": "Frequency", "unit": "Hz", "values": freqs * 1e6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals2D.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> FluxDepResult:
        signals2D, values, freqs, comment = load_data(
            filepath, return_comment=True, **kwargs
        )
        assert freqs is not None
        assert len(freqs.shape) == 1 and len(values.shape) == 1
        assert signals2D.shape == (len(values), len(freqs))

        freqs = freqs * 1e-6  # Hz -> MHz

        values = values.astype(np.float64)
        freqs = freqs.astype(np.float64)
        signals2D = signals2D.astype(np.complex128)

        if comment is not None:

            cfg, _, _ = parse_comment(comment)

            if cfg is not None:

                self.last_cfg = FluxDepCfg.validate_or_warn(cfg, source=filepath)
        self.last_result = (values, freqs, signals2D)

        return values, freqs, signals2D
