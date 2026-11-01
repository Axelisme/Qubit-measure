"""Shared in-memory PNG encoding policy for run and analysis stages."""

from collections.abc import Iterable
from io import BytesIO
from urllib.parse import quote

from matplotlib import rc_context
from matplotlib.figure import Figure


def encode_pngs(
    figures: Iterable[tuple[str, Figure]], *, prefix: str
) -> dict[str, bytes]:
    """Encode named Agg figures as PNG bytes without filesystem or presentation I/O.

    figures supplies unique names and stage-owned figures with Agg canvases.
    prefix is prepended verbatim (run uses "live-", analysis uses "analysis-").
    Return keys of prefix + URL-escaped name + ".png", escaping slashes too.
    Preserve each figure's current size and layout policy, rendering at 100 dpi
    with savefig.bbox=None even when ambient rcParams request tight cropping.
    The caller owns layout choices and the finish/release lifecycle. Rendering
    errors propagate unchanged. No files or temporary directories are created.
    """
    pngs: dict[str, bytes] = {}
    for name, figure in figures:
        with BytesIO() as buffer, rc_context({"savefig.bbox": None}):
            figure.savefig(buffer, format="png", dpi=100)
            pngs[f"{prefix}{quote(name, safe='')}.png"] = buffer.getvalue()
    return pngs
