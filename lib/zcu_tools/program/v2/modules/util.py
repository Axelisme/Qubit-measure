from __future__ import annotations

import warnings

import numpy as np
from qick.asm_v2 import QickParam, QickProgramV2

from ..utils import param2str


def get_fclk(prog: QickProgramV2, gen_ch: int | None = None, ro_ch: int | None = None):
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
    t: float | QickParam,
    gen_ch: int | None = None,
    ro_ch: int | None = None,
    take_ceil: bool = True,
) -> float | QickParam:
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
    length1: float | QickParam, length2: float | QickParam
) -> float | QickParam:
    if length1 > length2:
        return length1
    elif length1 < length2:
        return length2

    # Equal lengths. For two plain floats this is genuinely unambiguous -- both
    # operands are the same constant, so returning either is exactly correct and
    # no warning is warranted. The "may not always be correct" caveat only applies
    # when at least one operand is a sweepable QickParam: equality at this single
    # evaluation does not imply equality across every loop iteration.
    if not isinstance(length1, QickParam) and not isinstance(length2, QickParam):
        return length1

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


def merge_max_length(*args: float | QickParam) -> float | QickParam:
    merge_list = list(args)

    if len(merge_list) == 0:
        raise ValueError("at least one length must be provided")

    def try_reduce(
        length1: float | QickParam, length2: float | QickParam
    ) -> float | QickParam | None:
        if length1 > length2:
            return length1
        elif length1 < length2:
            return length2
        # Equal: collapse two equal plain floats unambiguously (see
        # calc_max_length); keep both only when a QickParam makes the
        # cross-iteration relationship ambiguous.
        elif not isinstance(length1, QickParam) and not isinstance(length2, QickParam):
            return length1
        else:
            return None

    while True:
        prev_num = len(merge_list)
        i = 0
        while i < len(merge_list):
            j = i + 1
            while j < len(merge_list):
                result = try_reduce(merge_list[i], merge_list[j])
                if result is not None:
                    merge_list[i] = result
                    merge_list.pop(j)
                else:
                    j += 1
            i += 1

        # no more reduction possible
        if len(merge_list) == prev_num:
            break

    if len(merge_list) == 1:
        return merge_list[0]

    warnings.warn(
        f"Detected multiple overlapping lengths: {[param2str(m) for m in merge_list]}. "
        "Using the maximum length among them for calculation. "
        "Note: this approach may not always be correct; it is only correct for some loop iterations and may be inaccurate for others."
    )

    merge_list = [m.maxval() if isinstance(m, QickParam) else m for m in merge_list]

    return max(merge_list)
