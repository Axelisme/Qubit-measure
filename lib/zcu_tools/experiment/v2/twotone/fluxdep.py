from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.device import DeviceInfo
from zcu_tools.experiment import (
    IDENTITY,
    MHZ_TO_HZ,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
    record_result,
    retrieve_result,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import set_flux_in_dev_cfg, setup_devices
from zcu_tools.experiment.v2.runner import MeasureSession, TaskState
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot2DwithLine
from zcu_tools.notebook.analysis.fluxdep.interactive import (
    InteractiveFindPoints,
    InteractiveLines,
)
from zcu_tools.program.v2 import SweepCfg, TwoToneCfg, TwoToneProgram, sweep2param
from zcu_tools.utils.process import minus_background


@dataclass(frozen=True)
class FreqFluxResult:
    values: NDArray[np.float64]
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]
    cfg_snapshot: FreqFluxCfg | None = None


def freqflux_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.abs(minus_background(signals, axis=1))


class FreqFluxSweepCfg(ConfigBase):
    flux: SweepCfg
    freq: SweepCfg


class FreqFluxCfg(TwoToneCfg, ExpCfgModel):
    dev: Mapping[str, DeviceInfo]  # type: ignore[dict-item]
    sweep: FreqFluxSweepCfg


class FreqFluxExp(PersistableExperiment[FreqFluxResult, FreqFluxCfg]):
    # inner freqs stores MHz on disk -> scale=MHZ_TO_HZ; outer values raw -> IDENTITY
    AXES_SPEC = AxesSpec(
        axes=(
            Axis("freqs", "Frequency", "Hz", scale=MHZ_TO_HZ),
            Axis("values", "Flux device value", "a.u.", scale=IDENTITY),
        ),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=FreqFluxResult,
        cfg_type=FreqFluxCfg,
        tag="twotone/flux_dep/freq",
    )

    @record_result
    def run(
        self,
        soc,
        soccfg,
        cfg: FreqFluxCfg,
        *,
        fail_retry: int = 0,
        acquire_kwargs: dict[str, Any] | None = None,
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
            with MeasureSession(cfg) as run:
                signals_buffer = run.buffer(
                    (len(dev_values), len(freqs)),
                    dtype=np.complex128,
                    on_update=lambda data: viewer.update(
                        dev_values, freqs, freqflux_signal2real(data)
                    ),
                )
                for step in run.scan("flux", dev_values.tolist()):
                    set_flux_in_dev_cfg(step.cfg.dev, step.value)
                    signals_buffer[step].measure(
                        measure_fn,
                        pbar_n=step.cfg.rounds,
                        retry=fail_retry,
                    )
                signals = signals_buffer.array

        return FreqFluxResult(
            values=dev_values, freqs=freqs, signals=signals, cfg_snapshot=orig_cfg
        )

    @retrieve_result
    def analyze(
        self,
        result: FreqFluxResult | None = None,
        flux_half: float | None = None,
        flux_int: float | None = None,
    ) -> InteractiveLines:
        assert result is not None, "no result found"

        values = result.values
        freqs = result.freqs
        signals2D = result.signals

        signals2D = minus_background(signals2D, axis=1)

        actline = InteractiveLines(
            signals2D, values, freqs, flux_half=flux_half, flux_int=flux_int
        )

        return actline

    @retrieve_result
    def extract_points(
        self,
        result: FreqFluxResult | None = None,
    ) -> InteractiveFindPoints:
        assert result is not None, "no result found"

        values = result.values
        freqs = result.freqs
        signals2D = result.signals

        point_selector = InteractiveFindPoints(signals2D, values, freqs)

        return point_selector
