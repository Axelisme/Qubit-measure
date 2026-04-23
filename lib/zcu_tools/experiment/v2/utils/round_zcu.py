from __future__ import annotations

from numbers import Number

import numpy as np
from numpy.typing import NDArray
from qick import QickConfig
from typing_extensions import Any, Callable, Literal, Optional, Sequence, TypeVar, Union

from zcu_tools.program import SweepCfg

T_Value = TypeVar("T_Value", NDArray[np.float64], float)


def round_zcu_time(
    us: T_Value,
    soccfg: QickConfig,
    gen_ch: Optional[int] = None,
    ro_ch: Optional[int] = None,
    scaler: float = 1.0,
) -> T_Value:
    one_cycle = soccfg.cycles2us(1, gen_ch=gen_ch, ro_ch=ro_ch)

    def _convert_time(t: float) -> float:
        # us2cycles use np.round to convert time to cycles
        # but qick implementation of sweep  use np.trunc to convert step size
        # pre substract 0.5 cycle to perform truncation instead of rounding
        # TODO: is there better way to handle the rounding issue
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
    ro_ch: Optional[int] = None,
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
    ro_ch: Optional[int] = None,
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


def round_sweep_dict(sweep: SweepCfg, *args, **kwargs) -> SweepCfg:
    expts = sweep["expts"]
    span = sweep["stop"] - sweep["start"]
    round_start = apply_round(sweep["start"], *args, **kwargs)
    round_step = apply_round(span / (expts - 1), *args, **kwargs)
    round_span = (expts - 1) * round_step
    round_stop = round_start + round_span

    return SweepCfg(start=round_start, stop=round_stop, expts=expts, step=round_step)


def sweep2array(
    sweep: Union[SweepCfg, Sequence, NDArray],
    round_type: Literal["none", "time", "freq", "phase", "gain"] = "none",
    round_info: Optional[dict[str, Any]] = None,
    allow_array: bool = False,
) -> NDArray[np.float64]:
    if round_info is None:
        round_info = {}

    if isinstance(sweep, dict):
        round_sweep = round_sweep_dict(sweep, round_type, round_info)

        return round_sweep["start"] + np.linspace(
            0, round_sweep["stop"] - round_sweep["start"], round_sweep["expts"]
        )
    elif isinstance(sweep, list) or isinstance(sweep, np.ndarray):
        if not allow_array:
            raise ValueError("Custom sweep is not allowed")
        sweep_array = np.asarray(sweep, dtype=np.float64)
        return apply_round(sweep_array, round_type, round_info)
    else:
        raise ValueError("Invalid sweep format")
