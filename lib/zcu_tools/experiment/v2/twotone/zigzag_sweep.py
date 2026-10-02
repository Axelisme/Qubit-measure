from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    IDENTITY,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runtime import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    LoadValue,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Repeat,
    Reset,
    ResetCfg,
    SweepCfg,
    sweep2param,
)
from zcu_tools.utils.process import rotate2real


@dataclass(frozen=True)
class ZigZagScanResult:
    times: NDArray[np.int64]
    values: NDArray[np.float64]
    signals: NDArray[np.complex128]


def zigzag_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class ZigZagScanModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    X90_pulse: PulseCfg
    X180_pulse: PulseCfg | None = None
    readout: ReadoutCfg


class ZigZagScanSweepCfg(ConfigBase):
    gain: SweepCfg | None = None
    freq: SweepCfg | None = None


class ZigZagScanCfg(ProgramV2Cfg, ExpCfgModel):
    modules: ZigZagScanModuleCfg
    sweep: ZigZagScanSweepCfg
    n_times: int
    repeat_on: Literal["X90_pulse", "X180_pulse"] = "X180_pulse"


@dataclass(frozen=True)
class ZigZagScanAnalyzeOptions:
    find_range: tuple[float | None, float | None] = (None, None)


@dataclass(frozen=True)
class ZigZagScanAnalysis:
    min_value: float


class ZigZagScanExp(PersistableExperiment[ZigZagScanResult, ZigZagScanCfg]):
    Options: ClassVar[type[ZigZagScanAnalyzeOptions]] = ZigZagScanAnalyzeOptions

    SWEEP_MAP = {
        "gain": {"name": "Gain (a.u.)", "param_key": "gain"},
        "freq": {"name": "Frequency (MHz)", "param_key": "freq"},
    }

    # inner-first: times (fastest-varying, int64) then values; both a.u. -> IDENTITY
    AXES_SPEC = AxesSpec(
        axes=(
            Axis("times", "Times", "a.u.", IDENTITY, np.int64),
            Axis("values", "Sweep value", "a.u."),
        ),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=ZigZagScanResult,
        cfg_type=ZigZagScanCfg,
        tag="twotone/ge/zigzag_scan",
    )

    def run(self, cfg: ZigZagScanCfg, *, context: RunContext) -> ZigZagScanResult:
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg
        repeat_on = cfg.repeat_on

        setup_devices(
            cfg,
            context.devices,
            progress=True,
            cancel_signal=context.cancel_signal,
        )
        modules = cfg.modules

        times = np.arange(0, cfg.n_times + 1)

        if cfg.sweep.freq is not None:
            x_key = "freq"
            x_sweep = cfg.sweep.freq
        elif cfg.sweep.gain is not None:
            x_key = "gain"
            x_sweep = cfg.sweep.gain
        else:
            raise ValueError("No sweep parameter found in cfg")
        x_info = ZigZagScanExp.SWEEP_MAP[x_key]

        repeat_pulse = getattr(modules, repeat_on)
        if repeat_pulse is None:
            raise ValueError(f"Repeat on pulse {repeat_on} not found")

        values = sweep2array(
            x_sweep,
            x_key,
            {"soccfg": soccfg, "gen_ch": repeat_pulse.ch},
        )

        viewer = context.plots.liveplot_2d("measurement", "Times", x_info["name"])
        signals_buffer = SignalBuffer(
            (len(times), len(values)),
            on_update=lambda data: viewer.update(
                times.astype(np.float64),
                values,
                zigzag_signal2real(data),
            ),
        )
        with Schedule(cfg, signals_buffer, stop=context.cancel_signal) as sched:
            modules = sched.cfg.modules

            X90_pulse = deepcopy(modules.X90_pulse)
            repeat_pulse = getattr(modules, repeat_on)
            if repeat_pulse is None:
                raise ValueError(f"Repeat on pulse {repeat_on} not found")

            x_sweep = getattr(sched.cfg.sweep, x_key)
            assert x_sweep is not None
            x_param = sweep2param(x_info["param_key"], x_sweep)
            repeat_pulse.set_param(x_info["param_key"], x_param)

            # Convert to plain int list: LoadValue.values expects Sequence[int],
            # and numpy 2.x scalar types (int_) are not considered int by pyright.
            loop_n: list[int] = [
                int(x) for x in (2 * times if repeat_on == "X90_pulse" else times)
            ]

            _ = (
                sched.prog_builder(soc, soccfg)
                .add(
                    LoadValue(
                        "load_repeat_count",
                        values=loop_n,
                        idx_reg="times",
                        val_reg="repeat_count",
                    ),
                    Reset("reset", cfg=modules.reset),
                    Pulse("X90_pulse", cfg=X90_pulse),
                    Repeat(
                        "zigzag_loop",
                        n="repeat_count",
                        # int() cast: numpy scalar types are not plain int to pyright.
                        range_hint=(int(min(times)), int(max(times))),
                    ).add_content(Pulse(f"loop_{repeat_on}", cfg=repeat_pulse)),
                    Readout("readout", cfg=modules.readout),
                )
                .declare_sweep("times", len(times))
                .declare_sweep(x_key, x_sweep)
                .build_and_acquire()
            )

        return ZigZagScanResult(
            times=times, values=values, signals=signals_buffer.array
        )

    def analyze(
        self,
        source: RunRecord[ZigZagScanCfg, ZigZagScanResult],
        options: ZigZagScanAnalyzeOptions,
        *,
        plots: Plots,
    ) -> ZigZagScanAnalysis:
        result = source.result
        find_range = options.find_range

        times = result.times
        values = result.values
        signals = result.signals

        real_signals = zigzag_signal2real(signals)  # (times , values)
        valid_cutoff = np.min(np.sum(~np.isnan(real_signals), axis=0))

        if valid_cutoff < 2:
            raise ValueError("Not enough valid data points for analysis")

        times = times[:valid_cutoff]
        real_signals = real_signals[:valid_cutoff]

        val_diff = np.sum(np.abs(np.diff(real_signals, axis=0)), axis=0)
        loss = gaussian_filter1d(val_diff, sigma=1)

        if find_range[0] is not None:
            loss[values < find_range[0]] = np.nan
        if find_range[1] is not None:
            loss[values > find_range[1]] = np.nan
        min_value = values[np.nanargmin(loss)]

        fig, _ = plots.subplots("fit", nrows=2, ncols=1, sharex=True)
        ax1, ax2 = fig.axes

        dx = (values[1] - values[0]) * 0.5
        dy = (times[1] - times[0]) * 0.5
        ax1.imshow(
            zigzag_signal2real(signals),
            aspect="auto",
            extent=(values[0] - dx, values[-1] + dx, times[0] - dy, times[-1] + dy),
            origin="lower",
            interpolation="none",
        )
        ax1.set_ylabel("Number of gate")
        ax2.plot(values, loss, marker=".")
        ax2.axvline(
            x=min_value, color="red", linestyle="--", label=f"x = {min_value:.3f}"
        )
        ax2.grid(True)
        ax2.legend()
        ax2.set_xlabel("Sweep value (a.u.)")
        ax2.set_ylabel("Loss (a.u.)")

        return ZigZagScanAnalysis(min_value=float(min_value))
