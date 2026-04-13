from __future__ import annotations

from copy import deepcopy

import numpy as np
from numpy.typing import NDArray
from typeguard import check_type
from typing_extensions import (
    Any,
    Callable,
    Mapping,
    NotRequired,
    Optional,
    TypeAlias,
    TypedDict,
    cast,
)

from zcu_tools.device import DeviceInfo
from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import set_flux_in_dev_cfg
from zcu_tools.experiment.v2.runner import Task, TaskCfg, TaskState, run_task
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot2DwithLine
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import (
    ModularProgramCfg,
    ModularProgramV2,
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


class OneToneFluxModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    readout: PulseReadoutCfg


class OneToneFluxCfg(ModularProgramCfg, TaskCfg):
    modules: OneToneFluxModuleCfg
    dev: Mapping[str, DeviceInfo]
    sweep: dict[str, SweepCfg]


class OneToneFluxExp(AbsExperiment[OneToneFluxResult, OneToneFluxCfg]):
    def run(self, soc, soccfg, cfg: dict[str, Any]) -> OneToneFluxResult:
        _cfg = check_type(deepcopy(cfg), OneToneFluxCfg)
        modules = _cfg["modules"]

        jpa_flux_sweep = _cfg["sweep"]["jpa_flux"]

        jpa_fluxs = sweep2array(jpa_flux_sweep, allow_array=True)
        freqs = sweep2array(
            _cfg["sweep"]["freq"],
            "freq",
            {"soccfg": soccfg, "gen_ch": modules["readout"].pulse_cfg.ch},
            allow_array=True,
        )

        def measure_fn(
            ctx: TaskState[NDArray[np.complex128], Any],
            update_hook: Optional[Callable[[int, list[NDArray[np.float64]]], None]],
        ) -> list[NDArray[np.float64]]:
            cfg: OneToneFluxCfg = cast(OneToneFluxCfg, ctx.cfg)
            modules = cfg["modules"]

            freq_sweep = cfg["sweep"]["freq"]
            freq_param = sweep2param("freq", freq_sweep)
            modules["readout"].set_param("freq", freq_param)

            return ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    Reset("reset", modules.get("reset")),
                    PulseReadout("readout", modules["readout"]),
                ],
                sweep=[("freq", freq_sweep)],
            ).acquire(ctx.env["soc"], progress=False, callback=update_hook)

        with LivePlot2DwithLine(
            "JPA Flux value (a.u.)",
            "Readout frequency (MHz)",
            line_axis=1,
            num_lines=5,
        ) as viewer:
            signals = run_task(
                task=Task(measure_fn=measure_fn, result_shape=(len(freqs),)).scan(
                    "JPA Flux value",
                    jpa_fluxs.tolist(),
                    before_each=lambda i, ctx, flux: set_flux_in_dev_cfg(
                        ctx.cfg["dev"], flux, label="jpa_flux_dev"
                    ),
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    jpa_fluxs, freqs, np.abs(np.asarray(ctx.root_data))
                ),
            )
            signals = np.asarray(signals)

        # record last cfg and result
        self.last_cfg = _cfg
        self.last_result = (jpa_fluxs, freqs, signals)

        return jpa_fluxs, freqs, signals

    def analyze(self, result: Optional[OneToneFluxResult] = None) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        jpa_fluxs, freqs, signals = result

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
        signals, jpa_fluxs, freqs = load_data(filepath, **kwargs)
        assert jpa_fluxs is not None and freqs is not None
        assert len(jpa_fluxs.shape) == 1 and len(freqs.shape) == 1
        assert signals.shape == (len(freqs), len(jpa_fluxs))

        freqs = freqs * 1e-6  # Hz -> MHz
        signals = signals.T  # transpose back

        jpa_fluxs = jpa_fluxs.astype(np.float64)
        freqs = freqs.astype(np.float64)
        signals = signals.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (jpa_fluxs, freqs, signals)

        return jpa_fluxs, freqs, signals
