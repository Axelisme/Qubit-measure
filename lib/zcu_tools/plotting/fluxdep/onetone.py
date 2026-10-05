"""Native one-tone peak-selection rendering, without GUI state ownership."""

from __future__ import annotations

from matplotlib.figure import Figure

from zcu_tools.analysis.fluxdep.onetone import OneToneInputs, OneTonePickState


class OneTonePickPlot:
    """Own image/curve artists and project committed peak indices.

    Uses Device value/GHz axes and normalized-amplitude curve, ticks and grids.
    Does not own analysis state, Qt canvas, presentation, or a pyplot manager.
    """

    def __init__(
        self,
        figure: Figure,
        inputs: OneToneInputs,
        *,
        flux_half: float | None = None,
        flux_int: float | None = None,
    ) -> None:
        """Build two panels in the supplied empty figure without switching canvas.

        inputs provides read-only spectrum and curve preprocessing.
        Optional finite flux_half/int are native device-position reference lines.
        ValueError rejects nonfinite reference positions or nonempty figure.
        """
        raise NotImplementedError

    def show_state(self, state: OneTonePickState) -> None:
        """Update existing peak artists from committed state without analysis writes.

        ValueError rejects out-of-bounds indices. Caller chooses when to redraw.
        """
        raise NotImplementedError


def make_onetone_pick_figure(
    inputs: OneToneInputs,
    state: OneTonePickState,
    *,
    flux_half: float | None = None,
    flux_int: float | None = None,
) -> Figure:
    """Return a separate Agg Figure projecting native device/GHz peak selection.

    No pyplot registration, presentation or ownership transfer. Reference line
    and index validation follows OneTonePickPlot; caller owns figure lifetime.
    """
    raise NotImplementedError
