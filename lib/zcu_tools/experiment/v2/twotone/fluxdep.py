from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Any, Mapping, Optional

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
from zcu_tools.notebook.analysis.fluxdep.interactive import (
    InteractiveFindPoints,
    InteractiveLines,
)
from zcu_tools.program.v2 import SweepCfg, TwoToneCfg, TwoToneProgram, sweep2param
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.process import minus_background


@dataclass(frozen=True)
class FreqFluxResult:
    values: NDArray[np.float64]
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]
    cfg_snapshot: Optional[FreqFluxCfg] = None


def freqflux_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(minus_background(signals, axis=1))


class FreqFluxSweepCfg(ConfigBase):
    flux: SweepCfg
    freq: SweepCfg


class FreqFluxCfg(TwoToneCfg, ExpCfgModel):
    dev: Mapping[str, DeviceInfo]  # type: ignore[dict-item]
    sweep: FreqFluxSweepCfg


class FreqFluxExp(AbsExperiment[FreqFluxResult, FreqFluxCfg]):
    def run(
        self,
        soc,
        soccfg,
        cfg: FreqFluxCfg,
        *,
        fail_retry: int = 0,
        acquire_kwargs: Optional[dict[str, Any]] = None,
    ) -> FreqFluxResult:
        orig_cfg = deepcopy(cfg)
        modules = cfg.modules

        value_sweep = cfg.sweep.flux
        freq_sweep = cfg.sweep.freq

        dev_values = sweep2array(value_sweep, allow_array=True)
        freqs = sweep2array(
            freq_sweep, "freq", {"soccfg": soccfg, "gen_ch": modules.qub_pulse.ch}
        )

        def measure_fn(
            ctx: TaskState[NDArray[np.complex128], Any, FreqFluxCfg], update_hook
        ):
            cfg = ctx.cfg
            modules = cfg.modules
            setup_devices(cfg, progress=False)

            # Frequency is swept by FPGA (hard sweep)
            freq_sweep = cfg.sweep.freq
            freq_param = sweep2param("freq", freq_sweep)
            modules.qub_pulse.set_param("freq", freq_param)

            return TwoToneProgram(soccfg, cfg, sweep=[("freq", freq_sweep)]).acquire(
                soc,
                progress=False,
                round_hook=update_hook,
                stop_checkers=[ctx.is_stop],
                **(acquire_kwargs or {}),
            )

        with LivePlot2DwithLine(
            "Flux device value", "Frequency (MHz)", line_axis=1, num_lines=2
        ) as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=measure_fn, result_shape=(len(freqs),), pbar_n=cfg.rounds
                )
                .auto_retry(max_retries=fail_retry)
                .scan(
                    "flux",
                    dev_values.tolist(),
                    before_each=lambda _, ctx, flux: set_flux_in_dev_cfg(
                        ctx.cfg.dev, flux
                    ),
                ),
                init_cfg=cfg,
                on_update=lambda ctx: viewer.update(
                    dev_values, freqs, freqflux_signal2real(np.asarray(ctx.root_data))
                ),
            )
            signals = np.asarray(signals)

        # record result
        self.last_result = FreqFluxResult(
            values=dev_values, freqs=freqs, signals=signals, cfg_snapshot=orig_cfg
        )

        return self.last_result

    def analyze(
        self,
        result: Optional[FreqFluxResult] = None,
        flux_half: Optional[float] = None,
        flux_int: Optional[float] = None,
    ) -> InteractiveLines:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        values = result.values
        freqs = result.freqs
        signals2D = result.signals

        signals2D = minus_background(signals2D, axis=1)

        actline = InteractiveLines(
            signals2D, values, freqs, flux_half=flux_half, flux_int=flux_int
        )

        return actline

    def extract_points(
        self,
        result: Optional[FreqFluxResult] = None,
    ) -> InteractiveFindPoints:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        values = result.values
        freqs = result.freqs
        signals2D = result.signals

        point_selector = InteractiveFindPoints(signals2D, values, freqs)

        return point_selector

    def save(
        self,
        filepath: str,
        result: Optional[FreqFluxResult] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/flux_dep/freq",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        cfg = result.cfg_snapshot
        if cfg is None:
            raise ValueError("cfg_snapshot is None")

        values = result.values
        freqs = result.freqs
        signals2D = result.signals
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

    def load(self, filepath: str, **kwargs) -> FreqFluxResult:
        signals2D, values, freqs, comment = load_data(
            filepath, return_comment=True, **kwargs
        )
        assert values is not None and freqs is not None
        assert len(values.shape) == 1 and len(freqs.shape) == 1
        assert signals2D.shape == (len(values), len(freqs))

        freqs = freqs * 1e-6  # Hz -> MHz

        values = values.astype(np.float64)
        freqs = freqs.astype(np.float64)
        signals2D = signals2D.astype(np.complex128)

        cfg_snapshot = None
        if comment is not None:
            cfg, _, _ = parse_comment(comment)

            if cfg is not None:
                cfg_snapshot = FreqFluxCfg.validate_or_warn(cfg, source=filepath)
        self.last_result = FreqFluxResult(
            values=values, freqs=freqs, signals=signals2D, cfg_snapshot=cfg_snapshot
        )

        return self.last_result
