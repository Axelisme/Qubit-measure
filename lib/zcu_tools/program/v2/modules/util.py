import warnings
from typing import Union, Optional

from qick.asm_v2 import QickParam, AbsQickProgram

from ..utils import param2str


def round_timestamp(
    prog: AbsQickProgram,
    t: Union[float, QickParam],
    gen_ch: Optional[int] = None,
    ro_ch: Optional[int] = None,
    take_ceil: bool = True,
) -> float:
    cycles_t = prog.us2cycles(t, gen_ch=gen_ch, ro_ch=ro_ch, as_float=True)
    if take_ceil:
        cycles_t = 0.99 + cycles_t
    return prog.cycles2us(cycles_t, gen_ch=gen_ch, ro_ch=ro_ch)


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
