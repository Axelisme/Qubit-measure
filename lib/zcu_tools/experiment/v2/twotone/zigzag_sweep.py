from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import make_comment, parse_comment, setup_devices
from zcu_tools.experiment.v2.runner import Task, TaskState, run_task
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot2D
from zcu_tools.program.v2 import (
    LoadValue,
    ModularProgramV2,
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
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.process import rotate2real


@dataclass(frozen=True)
class ZigZagScanResult:
    times: NDArray[np.int64]
    values: NDArray[np.float64]
    signals: NDArray[np.complex128]
    cfg_snapshot: ZigZagScanCfg | None = None


def zigzag_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real  # type: ignore


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


class ZigZagScanExp(AbsExperiment[ZigZagScanResult, ZigZagScanCfg]):
    SWEEP_MAP = {
        "gain": {"name": "Gain (a.u.)", "param_key": "gain"},
        "freq": {"name": "Frequency (MHz)", "param_key": "freq"},
    }

    def run(
        self,
        soc,
        soccfg,
        cfg: ZigZagScanCfg,
        *,
        repeat_on: Literal["X90_pulse", "X180_pulse"] = "X180_pulse",
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> ZigZagScanResult:
        orig_cfg = deepcopy(cfg)
        setup_devices(cfg, progress=True)
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
            x_key,  # type: ignore
            {"soccfg": soccfg, "gen_ch": repeat_pulse.ch},
        )

        def measure_fn(
            ctx: TaskState[NDArray[np.complex128], Any, ZigZagScanCfg],
            update_hook: Callable | None,
        ) -> list[NDArray[np.float64]]:
            cfg = ctx.cfg
            modules = cfg.modules

            X90_pulse = deepcopy(modules.X90_pulse)
            repeat_pulse = getattr(modules, repeat_on)
            if repeat_pulse is None:
                raise ValueError(f"Repeat on pulse {repeat_on} not found")

            x_sweep = getattr(cfg.sweep, x_key)
            assert x_sweep is not None
            x_param = sweep2param(x_info["param_key"], x_sweep)
            repeat_pulse.set_param(x_info["param_key"], x_param)

            # Convert to plain int list: LoadValue.values expects Sequence[int],
            # and numpy 2.x scalar types (int_) are not considered int by pyright.
            loop_n: list[int] = [
                int(x) for x in (2 * times if repeat_on == "X90_pulse" else times)
            ]

            return ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    LoadValue(
                        "load_repeat_count",
                        values=loop_n,
                        idx_reg="times",
                        val_reg="repeat_count",
                    ),
                    Reset("reset", modules.reset),
                    Pulse("X90_pulse", X90_pulse),
                    Repeat(
                        "zigzag_loop",
                        n="repeat_count",
                        # int() cast: numpy scalar types are not plain int to pyright.
                        range_hint=(int(min(times)), int(max(times))),
                    ).add_content(Pulse(f"loop_{repeat_on}", repeat_pulse)),
                    Readout("readout", modules.readout),
                ],
                sweep=[("times", len(times)), (x_key, x_sweep)],
            ).acquire(
                soc,
                progress=False,
                round_hook=update_hook,
                stop_checkers=[ctx.is_stop],
                **(acquire_kwargs or {}),
            )

        with LivePlot2D("Times", x_info["name"]) as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    result_shape=(len(times), len(values)),
                    pbar_n=cfg.rounds,
                ),
                init_cfg=cfg,
                on_update=lambda ctx: viewer.update(
                    times.astype(np.float64),
                    values,
                    zigzag_signal2real(np.asarray(ctx.root_data)),
                ),
            )
            signals = np.asarray(signals, dtype=np.complex128)

        # record result
        self.last_result = ZigZagScanResult(
            times=times, values=values, signals=signals, cfg_snapshot=orig_cfg
        )

        return self.last_result

    def analyze(
        self,
        result: ZigZagScanResult | None = None,
        find_range: tuple[float | None, float | None] = (None, None),
    ) -> tuple[float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

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

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        dx = (values[1] - values[0]) * 0.5
        dy = (times[1] - times[0]) * 0.5
        ax1.imshow(
            zigzag_signal2real(signals),
            aspect="auto",
            extent=[values[0] - dx, values[-1] + dx, times[0] - dy, times[-1] + dy],
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

        return min_value, fig

    def save(
        self,
        filepath: str,
        result: ZigZagScanResult | None = None,
        comment: str | None = None,
        tag: str = "twotone/ge/zigzag_scan",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        cfg = result.cfg_snapshot
        if cfg is None:
            raise ValueError("cfg_snapshot is None")

        times = result.times.astype(np.float64)
        values = result.values
        signals = result.signals
        comment = make_comment(cfg, comment)

        save_data(
            filepath=filepath,
            x_info={"name": "Times", "unit": "a.u.", "values": times},
            y_info={"name": "Sweep value", "unit": "a.u.", "values": values},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals.T},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> ZigZagScanResult:
        signals, times, values, comment = load_data(
            filepath, return_comment=True, **kwargs
        )
        assert times is not None and values is not None
        assert len(times.shape) == 1 and len(values.shape) == 1
        assert signals.shape == (len(values), len(times))

        signals = signals.T  # transpose back

        times = times.astype(np.int64)
        values = values.astype(np.float64)
        signals = signals.astype(np.complex128)

        cfg_snapshot = None
        if comment is not None:
            cfg, _, _ = parse_comment(comment)

            if cfg is not None:
                cfg_snapshot = ZigZagScanCfg.validate_or_warn(cfg, source=filepath)
        self.last_result = ZigZagScanResult(
            times=times, values=values, signals=signals, cfg_snapshot=cfg_snapshot
        )

        return self.last_result
