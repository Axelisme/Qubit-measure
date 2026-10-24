from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, PersistableExperiment, ProgramExpCfg
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    LoadValue,
    Pulse,
    Readout,
    Repeat,
    Reset,
    SweepCfg,
    sweep2param,
)
from zcu_tools.utils.process import rotate2real


@dataclass(frozen=True)
class ZigZagScanResult:
    """Mean complex IQ with shape (repetition count, sweep value).

    times counts pi rotations, with two X90 pulses per count when selected.
    values is the hardware-rounded gain in a.u. or frequency in MHz.
    """

    times: NDArray[np.int64]
    values: NDArray[np.float64]
    signals: NDArray[np.complex128]


def zigzag_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class ZigZagScanModuleCfg(ConfigBase):
    reset: Annotated[Reset.Cfg | None, FieldMeta(label="Reset")] = None
    X90_pulse: Annotated[Pulse.Cfg, FieldMeta(label="X90 Pulse")]
    X180_pulse: Annotated[Pulse.Cfg | None, FieldMeta(label="X180 Pulse")] = None
    readout: Annotated[Readout.Cfg, FieldMeta(label="Readout")]


class ZigZagScanSweepCfg(ConfigBase):
    gain: Annotated[SweepCfg | None, FieldMeta(label="Gain", unit="a.u.")] = None
    freq: Annotated[SweepCfg | None, FieldMeta(label="Freq", unit="MHz")] = None


class ZigZagScanCfg(ProgramExpCfg):
    modules: ZigZagScanModuleCfg
    sweep: ZigZagScanSweepCfg
    n_times: Annotated[
        int,
        FieldMeta(
            label="Max repetitions",
            description="Repetition counts run from zero to this value.",
        ),
    ]
    repeat_on: Annotated[
        Literal["X90_pulse", "X180_pulse"],
        FieldMeta(
            label="Repeat pulse",
            description="X90 repeats in pairs; each count adds one pi rotation.",
        ),
    ] = "X180_pulse"


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

    # inner-first: signals are (times, values), so values is inner and times
    # (int64) outer; both a.u. -> IDENTITY
    AXES_SPEC = AxesSpec(
        axes=(
            AxesSpec.Axis("values", "Sweep value", "a.u."),
            AxesSpec.Axis("times", "Times", "a.u.", dtype=np.int64),
        ),
        z=AxesSpec.Z("signals", "Signal", "a.u."),
        result_type=ZigZagScanResult,
        cfg_type=ZigZagScanCfg,
        cfg_schema_version="2.0",
        tag="twotone/ge/zigzag_scan",
    )

    def run(self, cfg: ZigZagScanCfg, *, context: RunContext) -> ZigZagScanResult:
        """Scan repeated-gate errors versus pulse gain or frequency.

        After optional reset and a fixed X90 preparation, repeat the selected
        pulse and read out. Each count adds one X180 or two X90 pulses, from
        zero through n_times. Only the repeated pulse uses the swept parameter;
        the preparation keeps its original settings.

        cfg must provide the selected repeat_on pulse and a gain or frequency
        sweep, or ValueError is raised. Frequency takes precedence if both
        sweeps are present. Return counts, rounded sweep values and mean IQ.
        analyze selects the value with least signal variation across counts.
        """
        # 1. Select the repeated pulse and round its gain or frequency sweep.
        cfg = deepcopy(cfg)
        soc, soccfg = context.soc, context.soccfg
        repeat_on = cfg.repeat_on

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

        if x_key == "freq":
            values = sweep2array(
                x_sweep,
                "freq",
                {
                    "soccfg": soccfg,
                    "gen_ch": repeat_pulse.ch,
                    "mixer_freq": repeat_pulse.mixer_freq,
                },
            )
        else:
            values = sweep2array(
                x_sweep, "gain", {"soccfg": soccfg, "gen_ch": repeat_pulse.ch}
            )

        # 2. Track mean IQ on the count-by-parameter grid.
        viewer = context.plots.liveplot_2d("measurement", "Times", x_info["name"])
        signals_buffer = SignalBuffer(
            (len(times), len(values)),
            on_update=lambda data: viewer.update(
                times.astype(np.float64),
                values,
                zigzag_signal2real(data),
            ),
        )
        with Schedule(cfg, signals_buffer, scope=context.scope) as sched:
            modules = sched.cfg.modules

            # Preserve the preparation pulse before changing the repeated pulse.
            X90_pulse = deepcopy(modules.X90_pulse)
            repeat_pulse = getattr(modules, repeat_on)
            if repeat_pulse is None:
                raise ValueError(f"Repeat on pulse {repeat_on} not found")

            x_sweep = getattr(sched.cfg.sweep, x_key)
            assert x_sweep is not None
            x_param = sweep2param(x_info["param_key"], x_sweep)
            repeat_pulse.set_param(x_info["param_key"], x_param)

            # X90 repeats in pairs so both choices add one pi rotation per count.
            # LoadValue needs plain Python ints rather than NumPy scalar types.
            loop_n: list[int] = [
                int(x) for x in (2 * times if repeat_on == "X90_pulse" else times)
            ]

            # 3. Load the count on the board, prepare X90, repeat, then read out.
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
        """Choose the swept value with least count-dependent IQ variation.

        Sum absolute changes between adjacent counts after projecting IQ onto
        its signal axis, then smooth that loss over sweep values. find_range
        optionally bounds the search in the same units as result.values.
        Return the minimum-loss value and plot the map and loss. This is an
        error-amplification tuning criterion, not a gate-fidelity estimate.
        Fewer than two valid counts or an all-NaN search range raises ValueError.
        """
        result = source.result
        find_range = options.find_range

        times = result.times
        values = result.values
        signals = result.signals

        # Compare every sweep value over the same acquired count range.
        real_signals = zigzag_signal2real(signals)  # (times , values)
        valid_cutoff = np.min(np.sum(~np.isnan(real_signals), axis=0))

        if valid_cutoff < 2:
            raise ValueError("Not enough valid data points for analysis")

        times = times[:valid_cutoff]
        real_signals = real_signals[:valid_cutoff]

        # Minimize smoothed count-to-count variation within the requested bounds.
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
