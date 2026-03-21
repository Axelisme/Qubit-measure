from __future__ import annotations

import warnings

import numpy as np
from qick.asm_v2 import QickProgramV2, QickParam
from typing_extensions import Optional, Union

from ..utils import param2str


def get_fclk(
    prog: QickProgramV2, gen_ch: Optional[int] = None, ro_ch: Optional[int] = None
):
    # TODO: a better way to get fclk for tproc?
    if gen_ch is not None and ro_ch is not None:
        raise RuntimeError("can't specify both gen_ch and ro_ch!")
    if gen_ch is not None:
        return prog.soccfg["gens"][gen_ch]["f_fabric"]
    elif ro_ch is not None:
        return prog.soccfg["readouts"][ro_ch]["f_output"]
    else:
        return prog.soccfg["tprocs"][0]["f_time"]


def round_timestamp(
    prog: QickProgramV2,
    t: Union[float, QickParam],
    gen_ch: Optional[int] = None,
    ro_ch: Optional[int] = None,
    take_ceil: bool = True,
) -> Union[float, QickParam]:
    fclk = get_fclk(prog, gen_ch=gen_ch, ro_ch=ro_ch)

    round_fn = np.ceil if take_ceil else np.floor

    # TODO: non-hacky way to round QickParam timestamps?
    cycles_t = t * fclk
    if isinstance(cycles_t, QickParam):
        cycles_t.start = int(round_fn(cycles_t.start))
        cycles_t.spans = {k: int(round_fn(v)) for k, v in cycles_t.spans.items()}
    else:
        cycles_t = int(round_fn(cycles_t))

    return cycles_t / fclk


def calc_max_length(
    length1: Union[float, QickParam], length2: Union[float, QickParam]
) -> Union[float, QickParam]:
    if length1 > length2:
        return length1
    elif length1 < length2:
        return length2

    warnings.warn(
        f"Detected overlap between {param2str(length1)} and {param2str(length2)}; "
        "using the maximum length for calculation. "
        "Note: this approach may not always be correct; it is only correct for some loop iterations and may be inaccurate for others."
    )

    if isinstance(length1, QickParam):
        length1 = length1.maxval()
    if isinstance(length2, QickParam):
        length2 = length2.maxval()

    return max(length1, length2)
