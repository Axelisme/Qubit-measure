from __future__ import annotations

import numpy as np
from qick.asm_v2 import QickParam, QickProgramV2


def get_fclk(prog: QickProgramV2, gen_ch: int | None = None, ro_ch: int | None = None):
    # QICK stores timing clocks under different soccfg sections by channel type.
    if gen_ch is not None and ro_ch is not None:
        raise RuntimeError("can't specify both gen_ch and ro_ch!")
    if gen_ch is not None:
        return prog.soccfg["gens"][gen_ch]["f_fabric"]
    if ro_ch is not None:
        return prog.soccfg["readouts"][ro_ch]["f_output"]
    return prog.soccfg["tprocs"][0]["f_time"]


def round_timestamp(
    prog: QickProgramV2,
    t: float | QickParam,
    gen_ch: int | None = None,
    ro_ch: int | None = None,
    *,
    take_ceil: bool = True,
) -> float | QickParam:
    fclk = get_fclk(prog, gen_ch=gen_ch, ro_ch=ro_ch)

    round_fn = np.ceil if take_ceil else np.floor

    # QickParam has no public rounding hook; round start/spans in cycle units.
    cycles_t = t * fclk
    if isinstance(cycles_t, QickParam):
        cycles_t.start = int(round_fn(cycles_t.start))
        cycles_t.spans = {k: int(round_fn(v)) for k, v in cycles_t.spans.items()}
    else:
        cycles_t = int(round_fn(cycles_t))

    return cycles_t / fclk
