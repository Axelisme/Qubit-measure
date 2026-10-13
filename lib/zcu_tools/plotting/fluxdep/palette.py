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
