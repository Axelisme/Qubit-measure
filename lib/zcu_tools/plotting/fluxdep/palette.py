"""Point colours shared by every fluxdep view: Qt pickers, previews and PNGs.

The colours come from the Okabe–Ito palette, which stays distinguishable
under red–green colour blindness. Added and removed points also differ by
marker shape (hollow circle, cross), so they never rely on colour alone.
Half/integer flux lines keep the line picker's colours, defined with
``TwoLinePicker`` in ``analysis.fluxdep.line_picker``.
"""

from __future__ import annotations

# Kept, current or sampled points: orange stands out on the dark resonance
# lines of a gray_r background and on its white background.
KEPT_COLOR = "#E69F00"
# Points a change adds: blue hollow circles.
ADDED_COLOR = "#0072B2"
ADDED_MARKER = "o"
# Points a change removes: reddish-purple crosses.
REMOVED_COLOR = "#CC79A7"
REMOVED_MARKER = "x"
# Points the Filter drops: grey.
DROPPED_COLOR = "#707070"
# Brush path and the band the brush sweeps along it: sky blue.
BRUSH_COLOR = "#56B4E9"

# The eight Okabe–Ito colours, ordered for model transitions on a gray_r
# background: the strongest contrasts first, yellow and black (weakest on
# white and on dark resonance lines) last.
TRANSITION_COLORS = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#E69F00",  # orange
    "#F0E442",  # yellow
    "#000000",  # black
)


def transition_color(index: int) -> str:
    """Colour of the index-th compiled transition, cycling TRANSITION_COLORS.

    index is a nonnegative position in the fit's compiled transition order;
    the Show lines and the points attributed to that transition share it.
    """
    return TRANSITION_COLORS[index % len(TRANSITION_COLORS)]
