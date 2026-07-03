from __future__ import annotations

import threading
from collections.abc import Callable

import numpy as np

from zcu_tools.progress_bar import make_pbar


def ramp_linear(
    *,
    start: float,
    target: float,
    step: float,
    apply_value: Callable[[float], None],
    progress: bool,
    desc: str,
    unit: str | None = None,
    progress_scale: float = 1.0,
    progress_decimals: int = 6,
    stop_event: threading.Event | None = None,
    include_start: bool = False,
) -> None:
    if step <= 0:
        raise ValueError(f"ramp step must be positive, got {step}")
    if progress_decimals < 0:
        raise ValueError(
            f"progress_decimals must be non-negative, got {progress_decimals}"
        )
    if start == target:
        return

    distance = abs(target - start)
    total = round(progress_scale * distance, progress_decimals)

    pbar_kwargs: dict[str, object] = {
        "total": total,
        "desc": desc,
        "leave": False,
        "disable": not progress,
    }
    if unit is not None:
        pbar_kwargs["unit"] = unit

    pbar = make_pbar(**pbar_kwargs)
    steps = max(1, round(distance / step))
    targets = np.linspace(start, target, num=steps + 1, endpoint=True)
    ramp_targets = targets if include_start else targets[1:]

    try:
        for raw_value in ramp_targets:
            if stop_event is not None and stop_event.is_set():
                break

            value = float(raw_value)
            apply_value(value)

            # Anchor progress to absolute distance so rounded increments do not drift.
            covered = min(
                round(progress_scale * abs(value - start), progress_decimals),
                total,
            )
            pbar.set_progress(covered)
    finally:
        pbar.close()
