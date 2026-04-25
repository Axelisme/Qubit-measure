from __future__ import annotations

from copy import deepcopy

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Any, Callable, Mapping, Optional, TypeAlias

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
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import (
    ModularProgramV2,
    ProgramV2Cfg,
    PulseReadout,
    PulseReadoutCfg,
    Reset,
    ResetCfg,
    sweep2param,
)
from zcu_tools.utils.datasaver import load_data, save_data

OneToneFluxResult: TypeAlias = tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


class OneToneFluxModuleCfg(ConfigBase):
    reset: Optional[ResetCfg] = None
    readout: PulseReadoutCfg


class OneToneFluxSweepCfg(ConfigBase):
    jpa_flux: SweepCfg
    freq: SweepCfg


class OneToneFluxCfg(ProgramV2Cfg, ExpCfgModel):
    modules: OneToneFluxModuleCfg
    dev: Mapping[str, DeviceInfo] = ...
    sweep: OneToneFluxSweepCfg


class OneToneFluxExp(AbsExperiment[OneToneFluxResult, OneToneFluxCfg]):
    def run(self, soc, soccfg, cfg: OneToneFluxCfg) -> OneToneFluxResult:
        modules = cfg.modules
        jpa_flux_sweep = cfg.sweep.jpa_flux

        jpa_fluxs = sweep2array(jpa_flux_sweep, allow_array=True)
        freqs = sweep2array(
            cfg.sweep.freq,
            "freq",
            {"soccfg": soccfg, "gen_ch": modules.readout.pulse_cfg.ch},
            allow_array=True,
        )

        def measure_fn(
            ctx: TaskState[NDArray[np.complex128], Any, OneToneFluxCfg],
            update_hook: Optional[Callable[[int, list[NDArray[np.float64]]], None]],
        ) -> list[NDArray[np.float64]]:
            cfg = ctx.cfg
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
            "JPA Flux value (a.u.)",
            "Readout frequency (MHz)",
            line_axis=1,
            num_lines=5,
        ) as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    result_shape=(len(freqs),),
                    pbar_n=cfg.rounds,
                ).scan(
                    "JPA Flux value",
                    jpa_fluxs.tolist(),
                    before_each=lambda _, ctx, flux: (
                        (dev := ctx.cfg.dev) is not None
                        and set_flux_in_dev_cfg(dev, flux, label="jpa_flux_dev")
                    ),
                ),
                init_cfg=cfg,
                on_update=lambda ctx: viewer.update(
                    jpa_fluxs, freqs, np.abs(np.asarray(ctx.root_data))
                ),
            )
            signals = np.asarray(signals)

        self.last_cfg = deepcopy(cfg)
        self.last_result = (jpa_fluxs, freqs, signals)
        return jpa_fluxs, freqs, signals

    def analyze(self, result: Optional[OneToneFluxResult] = None) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"
        raise NotImplementedError("analysis not implemented yet")

    def save(
        self,
        filepath: str,
        result: Optional[OneToneFluxResult] = None,
        comment: Optional[str] = None,
        tag: str = "jpa/flux_onetone",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        jpa_fluxs, freqs, signals = result

        cfg = self.last_cfg
        assert cfg is not None
        comment = make_comment(cfg, comment)

        save_data(
            filepath=filepath,
            x_info={"name": "JPA Flux value", "unit": "a.u.", "values": jpa_fluxs},
            y_info={"name": "Readout frequency", "unit": "Hz", "values": freqs * 1e6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> OneToneFluxResult:
        signals, jpa_fluxs, freqs, comment = load_data(
            filepath, return_comment=True, **kwargs
        )
        assert jpa_fluxs is not None and freqs is not None
        assert len(jpa_fluxs.shape) == 1 and len(freqs.shape) == 1
        assert signals.shape == (len(freqs), len(jpa_fluxs))

        freqs = freqs * 1e-6  # Hz -> MHz
        signals = signals.T  # transpose back

        jpa_fluxs = jpa_fluxs.astype(np.float64)
        freqs = freqs.astype(np.float64)
        signals = signals.astype(np.complex128)

        if comment is not None:
            cfg, _, _ = parse_comment(comment)

            if cfg is not None:
                self.last_cfg = OneToneFluxCfg.validate_or_warn(cfg, source=filepath)
        self.last_result = (jpa_fluxs, freqs, signals)
        return jpa_fluxs, freqs, signals
