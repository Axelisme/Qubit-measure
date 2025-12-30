from copy import deepcopy
from typing import Callable, Sequence, Tuple, List

import numpy as np
from numpy.typing import NDArray

from zcu_tools.experiment.v2.runner import TaskContextView
from zcu_tools.program.v2 import MyProgramV2


def measure_with_sweep(
    ctx: TaskContextView,
    prog_maker: Callable[[dict, float], MyProgramV2],
    values: Sequence[float],
    sweep_shape: Tuple[int, ...],
    **acquire_kwargs,
) -> List[NDArray[np.float64]]:
    cfg = deepcopy(ctx.cfg)
    rounds = cfg.pop("rounds", 1)
    cfg["rounds"] = 1

    callback: Callable = acquire_kwargs.pop("callback")

    # Prepare programs only once
    progs = [prog_maker(cfg, val) for val in values]

    acc_populations = np.zeros((1, len(values), *sweep_shape, 2), dtype=np.float64)
    for ir in range(1, rounds + 1):
        for j, prog in enumerate(progs):
            raw_j = prog.acquire(**acquire_kwargs)

            acc_populations[0, j] += raw_j[0][0]

        if callback is not None:
            callback(ir, [acc_populations / ir])

    return [acc_populations / rounds]
