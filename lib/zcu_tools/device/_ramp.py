from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from zcu_tools.progress_bar import make_pbar


@dataclass(frozen=True)
class RampProgress:
    """Display one ramp without changing its physical stepping.

    enabled controls visibility; desc and optional unit label the bar. scale
    converts physical distance to displayed distance; decimals rounds that value
    (default scale 1 and six decimals).
    """

    enabled: bool
    desc: str
    unit: str | None = None
    scale: float = 1.0
    decimals: int = 6


# tolerated: Stepping endpoints, writer, stop token and start-point policy remain independent.
def ramp_linear(  # noqa: PLR0913
    *,
    start: float,
    target: float,
    step: float,
    apply_value: Callable[[float], None],
    progress: RampProgress,
    stop_event: threading.Event | None = None,
    include_start: bool = False,
) -> None:
    progress_decimals = progress.decimals
    progress_scale = progress.scale
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
        "desc": progress.desc,
        "leave": False,
        "disable": not progress.enabled,
    }
    if progress.unit is not None:
        pbar_kwargs["unit"] = progress.unit

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
