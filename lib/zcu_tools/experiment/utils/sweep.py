from __future__ import annotations

from zcu_tools.program.v2 import SweepCfg


def make_sweep(
    start: int | float,
    stop: int | float | None = None,
    expts: int | None = None,
    step: int | float | None = None,
    *,
    force_int: bool = False,
) -> SweepCfg:
    """Build a regular sweep from endpoints, count, or increment.

    Numeric values use the caller's units. Provide stop and step to infer expts,
    or expts and stop to infer step. A single point (expts=1) may omit both.
    When expts and step are supplied, stop is ignored and recomputed.

    Count inference truncates (stop - start) / step + 1 with int(); the returned
    stop always equals start + step * (expts - 1). Negative steps are allowed.
    With force_int=True, infer missing values first, then truncate start, step,
    and expts toward zero before recomputing stop.

    Raise ValueError for insufficient inputs, nonpositive counts, a nonzero
    single-point step, a zero multi-point step, or SweepCfg validation failure.
    The returned SweepCfg is fresh; this function does not access hardware.
    """
    if expts is None:
        if stop is None or step is None:
            raise ValueError("Not enough information to define a sweep.")
        expts = _infer_sweep_count(start, stop, step)
    elif step is None:
        step = _infer_sweep_step(start, stop, expts)

    if force_int:
        start = int(start)
        step = int(step)
        expts = int(expts)

    if expts <= 0:
        raise ValueError(f"expts must be greater than 0, but got {expts}")
    if expts == 1 and step != 0:
        raise ValueError(f"for expts == 1, step must be 0, but got {step}")
    if expts > 1 and step == 0:
        raise ValueError(f"step must not be zero when expts > 1, but got {step}")

    stop = start + step * (expts - 1)
    return SweepCfg(start=start, stop=stop, expts=expts, step=step)


def _infer_sweep_count(start: int | float, stop: int | float, step: int | float) -> int:
    if step == 0:
        if stop != start:
            raise ValueError(
                f"stop must equal start when step is 0, got start={start}, stop={stop}"
            )
        return 1
    return int((stop - start) / step + 1)


def _infer_sweep_step(
    start: int | float, stop: int | float | None, expts: int
) -> int | float:
    if expts == 1:
        if stop is not None and stop != start:
            raise ValueError(
                f"for expts == 1, stop must equal start, got start={start}, stop={stop}"
            )
        return 0
    if stop is None:
        raise ValueError("Not enough information to define a sweep.")
    return (stop - start) / (expts - 1)
