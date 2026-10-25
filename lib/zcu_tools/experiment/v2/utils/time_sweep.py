"""Plan regular time sweeps before QICK compilation."""

import math

from qick import QickConfig

from zcu_tools.program.v2 import SweepCfg

from .round_zcu import round_zcu_time


def plan_time_sweep(
    sweep: SweepCfg,
    soccfg: QickConfig,
    gen_ch: int | None = None,
    ro_ch: int | None = None,
    scaler: float = 1.0,
) -> SweepCfg:
    """Return a clock-aligned sweep for compilation, without acquiring hardware.

    start/stop are microseconds before multiplication by finite nonzero scaler.
    gen_ch/ro_ch select the clock, as in round_zcu_time. The start rounds to the
    nearest cycle. The signed step magnitude is ceil(requested span / (expts-1))
    in cycles. Keep floor(available span / planned step)+1 points, never extending
    beyond the requested stop. Thus the endpoint differs by less than one planned
    step, not necessarily one cycle. Point count may decrease; input is unchanged.
    Pass the returned SweepCfg to both the compiler and coordinate preview.

    Raise ValueError for nonfinite inputs/scaler, fewer than two requested points,
    or a span below one cycle / fewer than two available points. Timing errors
    report the requested span, cycle and maximum available point count.
    """
    if (
        not all(math.isfinite(x) for x in (sweep.start, sweep.stop, scaler))
        or scaler == 0
    ):
        raise ValueError(
            "Time sweep endpoints and scaler must be finite; scaler must be nonzero"
        )
    span = abs(sweep.stop - sweep.start)
    cycle = abs(soccfg.cycles2us(1, gen_ch=gen_ch, ro_ch=ro_ch) / scaler)
    start = round_zcu_time(sweep.start, soccfg, gen_ch, ro_ch, scaler)
    direction = 1 if sweep.stop > sweep.start else -1
    max_points = max(0, math.floor(direction * (sweep.stop - start) / cycle) + 1)
    error = (
        f"Time sweep cannot supply two distinct points: span={span} us, "
        f"cycle={cycle} us, max_available_points={max_points}"
    )
    if sweep.expts < 2 or span < cycle:
        raise ValueError(error)
    step_cycles = math.ceil(span / cycle / (sweep.expts - 1))
    step = direction * step_cycles * cycle
    available = direction * (sweep.stop - start) / abs(step)
    # Recover exact integer grids from floating-point round trips through us.
    nearest = round(available)
    if math.isclose(available, nearest, rel_tol=0, abs_tol=1e-10):
        available = float(nearest)
    expts = math.floor(available) + 1
    if expts < 2:
        raise ValueError(error)
    return SweepCfg(
        start=start, stop=start + step * (expts - 1), expts=expts, step=step
    )
