from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from qick import QickConfig
from typing_extensions import Any, Literal, Optional, Sequence, Union

from zcu_tools.program import SweepCfg


def round_zcu_time(
    us: NDArray[np.float64], soccfg: QickConfig, gen_ch: Optional[int] = None
) -> NDArray[np.float64]:
    @np.vectorize
    def _convert_time(t: float) -> float:
        return soccfg.cycles2us(soccfg.us2cycles(t, gen_ch=gen_ch), gen_ch=gen_ch)

    return _convert_time(us)


def round_zcu_freq(
    freqs: NDArray[np.float64],
    soccfg: QickConfig,
    gen_ch: int,
    ro_ch: Optional[int] = None,
) -> NDArray[np.float64]:
    @np.vectorize
    def _convert_freq(f: float) -> float:
        return soccfg.reg2freq(
            soccfg.freq2reg(f, gen_ch=gen_ch, ro_ch=ro_ch), gen_ch=gen_ch
        )

    return _convert_freq(freqs)


def round_zcu_phase(
    phases: NDArray[np.float64],
    soccfg: QickConfig,
    gen_ch: int,
    ro_ch: Optional[int] = None,
) -> NDArray[np.float64]:
    @np.vectorize
    def _convert_phase(p: float) -> float:
        return soccfg.reg2deg(
            soccfg.deg2reg(p, gen_ch=gen_ch, ro_ch=ro_ch), gen_ch=gen_ch, ro_ch=ro_ch
        )

    return _convert_phase(phases)

def round_zcu_gain(
    gains: NDArray[np.float64], soccfg: QickConfig, gen_ch: int
) -> NDArray[np.float64]:
    maxv = soccfg.get_maxv(gen_ch)

    @np.vectorize
    def _convert_gain(g: float) -> float:
        # qick didn't provide gain2reg function, so implement it manually
        return int(np.round(g * maxv)) / maxv

    return _convert_gain(gains)


def sweep2array(
    sweep: Union[SweepCfg, Sequence, NDArray],
    allow_array: bool = False,
    round_type: Literal["none", "time", "freq", "phase", "gain"] = "none",
    round_info: Optional[dict[str, Any]] = None,
) -> NDArray:
    if isinstance(sweep, dict):
        return sweep["start"] + np.arange(sweep["expts"]) * sweep["step"]
    elif isinstance(sweep, list) or isinstance(sweep, np.ndarray):
        if not allow_array:
            raise ValueError("Custom sweep is not allowed")
        return np.asarray(sweep)
    else:
        raise ValueError("Invalid sweep format")