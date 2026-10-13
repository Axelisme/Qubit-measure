"""The region a brush sweeps along its whole stroke path, drawn on caller axes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import PathPatch
from matplotlib.path import Path

from zcu_tools.analysis.fluxdep.stroke import BrushPoint

from .palette import BRUSH_COLOR

# Corners of the polygon that approximates each brush circle.
_CIRCLE_SEGMENTS = 32


class BrushBand:
    """A translucent band plus a dashed centre line along one brush stroke.

    The brush is a circle of normalized radius ``width``: each axis is scaled
    by its span, as in ``analysis.fluxdep.stroke``. The band is the union of
    that circle at every vertex and the strips joining consecutive vertices,
    so it covers the whole swept path, not only its two ends. The artists
    live on the caller's axes; this class never draws the canvas.
    """

    def __init__(
        self, axes: Axes, *, x_span: float, y_span: float, zorder: float
    ) -> None:
        """Add hidden band and line artists to axes.

        x_span and y_span are the positive data spans one normalized unit
        means on each axis. zorder places the band; the line sits just above.
        """
        self._x_span = x_span
        self._y_span = y_span
        self._band = PathPatch(
            Path(np.zeros((1, 2))),
            facecolor=BRUSH_COLOR,
            edgecolor="none",
            alpha=0.3,
            zorder=zorder,
            visible=False,
        )
        axes.add_patch(self._band)
        (self._line,) = axes.plot(
            [], [], color=BRUSH_COLOR, linewidth=1, linestyle="--", zorder=zorder
        )

    def show(self, vertices: Sequence[BrushPoint], width: float) -> None:
        """Show the band of normalized radius width along vertices.

        vertices are in the axes' data units. An empty sequence hides both
        artists; width <= 0 shows only the centre line.
        """
        self._line.set_data([p.x for p in vertices], [p.y for p in vertices])
        if not vertices or width <= 0:
            self._band.set_visible(False)
            return
        self._band.set_path(
            _band_path(vertices, width * self._x_span, width * self._y_span)
        )
        self._band.set_visible(True)


def _band_path(vertices: Sequence[BrushPoint], rx: float, ry: float) -> Path:
    """One compound path of vertex circles and segment strips.

    Work in coordinates scaled by the radii, where the brush is a unit
    circle. Every subpath runs counter-clockwise, so the nonzero fill rule
    paints their union once, with no darker overlaps.
    """
    scaled = np.array([(p.x / rx, p.y / ry) for p in vertices], dtype=np.float64)
    angles = np.linspace(0.0, 2.0 * np.pi, _CIRCLE_SEGMENTS, endpoint=False)
    circle = np.column_stack((np.cos(angles), np.sin(angles)))
    polygons = [centre + circle for centre in scaled]
    for start, end in zip(scaled[:-1], scaled[1:], strict=True):
        length = float(np.hypot(*(end - start)))
        if length == 0.0:
            continue
        direction = (end - start) / length
        normal = np.array((-direction[1], direction[0]))
        polygons.append(
            np.array((start + normal, start - normal, end - normal, end + normal))
        )
    points: list[np.ndarray] = []
    codes: list[np.uint8] = []
    for polygon in polygons:
        points.extend((*polygon, polygon[0]))
        codes.extend(
            (Path.MOVETO, *([Path.LINETO] * (len(polygon) - 1)), Path.CLOSEPOLY)
        )
    return Path(np.array(points) * (rx, ry), codes)
