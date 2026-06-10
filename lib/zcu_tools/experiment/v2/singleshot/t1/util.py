from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import Any

import numpy as np
from numpy.typing import NDArray

from zcu_tools.experiment.v2.runner import TaskState
from zcu_tools.program.v2 import MyProgramV2


def measure_with_sweep(
    ctx: TaskState,
    prog_maker: Callable[[Any, float], MyProgramV2],
    values: Sequence[float],
    sweep_shape: tuple[int, ...],
    **acquire_kwargs,
) -> list[NDArray[np.float64]]:
    cfg = deepcopy(ctx.cfg)
    rounds = cfg.pop("rounds", 1)
    cfg["rounds"] = 1

    callback: Callable = acquire_kwargs.pop("callback")
    # Stop both granularities: each acquire's internal rounds check is_stop, and
    # the manual outer round loop below breaks on it (a QICK-level check cannot
    # stop this Python loop).
    acquire_kwargs.setdefault("stop_checkers", [ctx.is_stop])

    # Prepare programs only once
    progs = [prog_maker(cfg, val) for val in values]

    acc_populations = np.zeros((1, len(values), *sweep_shape, 2), dtype=np.float64)
    completed_rounds = 0
    for ir in range(1, rounds + 1):
        if ctx.is_stop():
            break
        for j, prog in enumerate(progs):
            raw_j = prog.acquire(**acquire_kwargs)

            acc_populations[0, j] += raw_j[0][0]

        completed_rounds = ir
        if callback is not None:
            callback(ir, [acc_populations / ir])

    # Normalize by rounds actually completed (avoid dividing a partial sum by the
    # full target when stopped early).
    return [acc_populations / max(completed_rounds, 1)]
