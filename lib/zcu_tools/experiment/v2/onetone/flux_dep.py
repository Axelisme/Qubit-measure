from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from zcu_tools.cfg_model import ConfigBase
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


@dataclass(frozen=True)
class FluxDepResult:
    values: NDArray[np.float64]
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]
    cfg_snapshot: FluxDepCfg | None = None


def fluxdep_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(signals)


class FluxDepModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    readout: PulseReadoutCfg


class FluxDepSweepCfg(ConfigBase):
    freq: SweepCfg
    flux: SweepCfg


class FluxDepCfg(ProgramV2Cfg, ExpCfgModel):
    modules: FluxDepModuleCfg
    # Field(...) makes dev required in this subclass, overriding the Optional
    # default from ExpCfgModel — intentional Pydantic pattern (type: ignore[override]).
    dev: Mapping[str, DeviceInfo] = Field(...)  # type: ignore[override]
    sweep: FluxDepSweepCfg


class FluxDepExp(AbsExperiment[FluxDepResult, FluxDepCfg]):
    def run(self, soc, soccfg, cfg: FluxDepCfg) -> FluxDepResult:
        orig_cfg = deepcopy(cfg)
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

        set_flux_in_dev_cfg(cfg.dev, dev_values[0])
        setup_devices(cfg, progress=True)

        def measure_fn(
            ctx: TaskState, update_hook: Callable | None
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
            ).acquire(
                soc,
                progress=False,
                round_hook=update_hook,
                stop_checkers=[ctx.is_stop],
            )

        with LivePlot2DwithLine(
            "Flux device value", "Frequency (MHz)", line_axis=1, num_lines=10
        ) as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=measure_fn, result_shape=(len(freqs),), pbar_n=cfg.rounds
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

        # record result
        self.last_result = FluxDepResult(
            values=dev_values, freqs=freqs, signals=signals, cfg_snapshot=orig_cfg
        )

        return self.last_result

    def analyze(
        self,
        result: FluxDepResult | None = None,
        flux_half: float | None = None,
        flux_int: float | None = None,
    ) -> InteractiveLines:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        values = result.values
        freqs = result.freqs
        signals2D = result.signals

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
        result: FluxDepResult | None = None,
        comment: str | None = None,
        tag: str = "onetone/flux_dep",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        values = result.values
        freqs = result.freqs
        signals2D = result.signals

        cfg = result.cfg_snapshot
        if cfg is None:
            raise ValueError("cfg_snapshot is None")
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

        cfg_snapshot = None
        if comment is not None:
            cfg, _, _ = parse_comment(comment)

            if cfg is not None:
                cfg_snapshot = FluxDepCfg.validate_or_warn(cfg, source=filepath)
        self.last_result = FluxDepResult(
            values=values, freqs=freqs, signals=signals2D, cfg_snapshot=cfg_snapshot
        )

        return self.last_result
