"""Flux-Dependence Analysis shared kernel.

The public API here is intentionally small and notebook-neutral. Notebook and Qt
adapters should translate their UI events into these functions/classes instead of
owning duplicated domain rules.
"""

from __future__ import annotations

from .line_picker import (
    TwoLinePicker,
    find_best_mirror_position,
    fold_initial_lines,
)
from .line_state import (
    FluxPickInputs,
    FluxPickState,
    align_lines,
    mirror_loss_at,
    move_line,
    swap_lines,
)
from .onetone import (
    detect_peaks,
    max_dispersion_freq_index,
    onetone_peak_points,
    smoothed_slice,
)
from .processing import (
    cast2real_and_norm,
    diff_mirror,
    downsample_points,
    spectrum2d_findpoint,
)
from .selection import (
    points_in_normalized_brush,
    toggle_near_mask,
)

__all__ = [
    "FluxPickInputs",
    "FluxPickState",
    "TwoLinePicker",
    "align_lines",
    "cast2real_and_norm",
    "detect_peaks",
    "diff_mirror",
    "downsample_points",
    "find_best_mirror_position",
    "fold_initial_lines",
    "max_dispersion_freq_index",
    "mirror_loss_at",
    "move_line",
    "onetone_peak_points",
    "points_in_normalized_brush",
    "smoothed_slice",
    "spectrum2d_findpoint",
    "swap_lines",
    "toggle_near_mask",
]
