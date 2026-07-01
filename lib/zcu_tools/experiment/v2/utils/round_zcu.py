from __future__ import annotations

import math
from collections.abc import Callable
from numbers import Number
from typing import Any, Literal, TypeVar

import numpy as np
from numpy.typing import NDArray
from qick import QickConfig

from zcu_tools.program.v2 import SweepCfg

T_Value = TypeVar("T_Value", NDArray[np.float64], float)


def round_zcu_time(
    us: T_Value,
    soccfg: QickConfig,
    gen_ch: int | None = None,
    ro_ch: int | None = None,
    scaler: float = 1.0,
) -> T_Value:
    one_cycle = soccfg.cycles2us(1, gen_ch=gen_ch, ro_ch=ro_ch)

    def _convert_time(t: float) -> float:
        # QICK scalar us2cycles rounds, while sweep step conversion truncates.
        # Shift by half a cycle so scalar preview follows the sweep grid.
        return (
            soccfg.cycles2us(
                soccfg.us2cycles(
                    scaler * t - 0.5 * one_cycle, gen_ch=gen_ch, ro_ch=ro_ch
                ),
                gen_ch=gen_ch,
                ro_ch=ro_ch,
            )
            / scaler
        )

    if isinstance(us, (Number, float)):
        return _convert_time(float(us))
    else:
        return np.vectorize(_convert_time)(us)


def round_zcu_freq(
    freq: T_Value,
    soccfg: QickConfig,
    gen_ch: int,
    ro_ch: int | None = None,
    scaler: float = 1.0,
) -> T_Value:
    one_reg = soccfg.reg2freq(1, gen_ch=gen_ch) - soccfg.reg2freq(0, gen_ch=gen_ch)

    def _convert_freq(f: float) -> float:
        return (
            soccfg.reg2freq(
                soccfg.freq2reg(scaler * f - 0.5 * one_reg, gen_ch=gen_ch, ro_ch=ro_ch),
                gen_ch=gen_ch,
            )
            / scaler
        )

    if isinstance(freq, (Number, float)):
        return _convert_freq(float(freq))
    else:
        return np.vectorize(_convert_freq)(freq)


def round_zcu_phase(
    phase: T_Value,
    soccfg: QickConfig,
    gen_ch: int,
    ro_ch: int | None = None,
    scaler: float = 1.0,
) -> T_Value:
    one_gain = soccfg.reg2deg(1, gen_ch=gen_ch, ro_ch=ro_ch) - soccfg.reg2deg(
        0, gen_ch=gen_ch, ro_ch=ro_ch
    )

    def _convert_phase(p: float) -> float:
        num_2pi = int(scaler * p / 360.0)
        return (
            360.0 * num_2pi  # restore the wrapped-around phase to the original value
            + soccfg.reg2deg(
                soccfg.deg2reg(scaler * p - 0.5 * one_gain, gen_ch=gen_ch, ro_ch=ro_ch),
                gen_ch=gen_ch,
                ro_ch=ro_ch,
            )
            / scaler
        )

    if isinstance(phase, (Number, float)):
        return _convert_phase(float(phase))
    else:
        return np.vectorize(_convert_phase)(phase)


def round_zcu_gain(
    gain: T_Value, soccfg: QickConfig, gen_ch: int, scaler: float = 1.0
) -> T_Value:
    maxv = soccfg.get_maxv(gen_ch)

    def _convert_gain(g: float) -> float:
        # qick didn't provide gain2reg function, so implement it manually
        return int(np.trunc(scaler * g * maxv)) / (scaler * maxv)

    if isinstance(gain, (Number, float)):
        return _convert_gain(float(gain))
    else:
        return np.vectorize(_convert_gain)(gain)


def apply_round(
    val: T_Value,
    round_type: Literal["none", "time", "freq", "phase", "gain"],
    round_info: dict[str, Any],
) -> T_Value:
    ROUND_FN_MAP: dict[str, Callable] = {
        "none": lambda v: v,
        "time": round_zcu_time,
        "freq": round_zcu_freq,
        "phase": round_zcu_phase,
        "gain": round_zcu_gain,
    }

    if round_fn := ROUND_FN_MAP.get(round_type):
        return round_fn(val, **round_info)
    else:
        raise ValueError(f"Invalid round type: {round_type}")


def _format_zero_step_error(
    sweep: SweepCfg,
    round_type: str,
    requested_step: float,
    rounded_step: float,
) -> str:
    reason = (
        "sweep step is effectively zero"
        if round_type == "none"
        else "sweep step becomes zero after hardware quantization"
    )
    return (
        f"{reason}: "
        f"round_type={round_type!r}, start={sweep.start}, stop={sweep.stop}, "
        f"expts={sweep.expts}, requested_step={requested_step}, "
        f"rounded_step={rounded_step}. Increase the sweep span or reduce expts so "
        "adjacent points remain distinct on the hardware grid."
    )


def round_sweep_dict(
    sweep: SweepCfg,
    round_type: Literal["none", "time", "freq", "phase", "gain"] = "none",
    round_info: dict[str, Any] | None = None,
) -> SweepCfg:
    if round_info is None:
        round_info = {}

    expts = sweep.expts
    span = sweep.stop - sweep.start
    round_start = apply_round(sweep.start, round_type, round_info)
    if expts == 1:
        return SweepCfg(start=round_start, stop=round_start, expts=expts, step=0.0)

    requested_step = span / (expts - 1)
    round_step = apply_round(requested_step, round_type, round_info)
    if math.isclose(float(round_step), 0.0, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(
            _format_zero_step_error(
                sweep, round_type, requested_step, float(round_step)
            )
        )
    round_span = (expts - 1) * round_step
    round_stop = round_start + round_span

    return SweepCfg(start=round_start, stop=round_stop, expts=expts, step=round_step)


def sweep2array(
    sweep: SweepCfg | list | NDArray,
    round_type: Literal["none", "time", "freq", "phase", "gain"] = "none",
    round_info: dict[str, Any] | None = None,
    allow_array: bool = False,
) -> NDArray[np.float64]:
    if round_info is None:
        round_info = {}

    if isinstance(sweep, SweepCfg):
        round_sweep = round_sweep_dict(sweep, round_type, round_info)

        return round_sweep.start + np.linspace(
            0, round_sweep.stop - round_sweep.start, round_sweep.expts
        )
    elif isinstance(sweep, list) or isinstance(sweep, np.ndarray):
        if not allow_array:
            raise ValueError(f"Custom sweep is not allowed: {sweep}")
        sweep_array = np.asarray(sweep, dtype=np.float64)
        return apply_round(sweep_array, round_type, round_info)
    else:
        raise ValueError(f"Invalid sweep format: {sweep}")
