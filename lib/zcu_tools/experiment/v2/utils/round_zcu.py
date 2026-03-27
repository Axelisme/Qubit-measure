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
    def _convert_time(t: float) -> float:
        return (
            soccfg.cycles2us(
                soccfg.us2cycles(scaler * t, gen_ch=gen_ch, ro_ch=ro_ch),
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
    def _convert_freq(f: float) -> float:
        return (
            soccfg.reg2freq(
                soccfg.freq2reg(scaler * f, gen_ch=gen_ch, ro_ch=ro_ch), gen_ch=gen_ch
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
    def _convert_phase(p: float) -> float:
        num_2pi = int(scaler * p / 360.0)
        return (
            360.0 * num_2pi  # restore the wrapped-around phase to the original value
            + soccfg.reg2deg(
                soccfg.deg2reg(scaler * p, gen_ch=gen_ch, ro_ch=ro_ch),
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
        return int(np.round(scaler * g * maxv)) / (scaler * maxv)

    if isinstance(gain, (Number, float)):
        return _convert_gain(float(gain))
    else:
        return np.vectorize(_convert_gain)(gain)


def sweep2array(
    sweep: Union[SweepCfg, Sequence, NDArray],
    round_type: Literal["none", "time", "freq", "phase", "gain"] = "none",
    round_info: Optional[dict[str, Any]] = None,
    allow_array: bool = False,
) -> NDArray[np.float64]:
    if round_info is None:
        round_info = {}

    ROUND_FN_MAP: dict[str, Callable] = {
        "none": lambda v: v,
        "time": round_zcu_time,
        "freq": round_zcu_freq,
        "phase": round_zcu_phase,
        "gain": round_zcu_gain,
    }

    def apply_round(val: T_Value) -> T_Value:

        if round_fn := ROUND_FN_MAP.get(round_type):
            return round_fn(val, **round_info)
        else:
            raise ValueError(f"Invalid round type: {round_type}")

    if isinstance(sweep, dict):
        expts = sweep["expts"]
        round_start = apply_round(sweep["start"])
        round_span = expts * apply_round((sweep["stop"] - sweep["start"]) / expts)

        return round_start + np.linspace(0, round_span, sweep["expts"])
    elif isinstance(sweep, list) or isinstance(sweep, np.ndarray):
        if not allow_array:
            raise ValueError("Custom sweep is not allowed")
        return apply_round(np.asarray(sweep, dtype=np.float64))
    else:
        raise ValueError("Invalid sweep format")
