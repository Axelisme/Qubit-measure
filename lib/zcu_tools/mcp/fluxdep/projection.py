"""Stateless agent summaries of authoritative Fluxdep RPC replies."""

import base64
from collections.abc import Sequence
from typing import Literal, NotRequired, TypedDict

from zcu_tools.gui.app.fluxdep.remote.dto import AxisSnapshot
from zcu_tools.mcp.core.images import validated_png
from zcu_tools.mcp.core.reply import ToolReply


class SpectrumSummary(TypedDict):
    """Loaded identity, published stage/count and unfinished draft kind."""

    name: str
    type: Literal["OneTone", "TwoTone"]
    stage: Literal["loaded", "aligned", "sampled"]
    point_count: int
    pending: Literal["align", "sample"] | None


class FitSummary(TypedDict):
    """GUI result availability and freshness, with reasons only when stale."""

    has_result: bool
    stale: bool
    stale_reasons: NotRequired[list[str]]


class AxisSummary(TypedDict):
    """Axis length, three-decimal extents (None when empty) and native unit."""

    count: int
    min: float | None
    max: float | None
    unit: str


def _stage(
    *, aligned: bool, completed: bool
) -> Literal["loaded", "aligned", "sampled"]:
    if not aligned:
        return "loaded"
    return "sampled" if completed else "aligned"


def spectrum_summaries(reply: ToolReply) -> list[SpectrumSummary]:
    """Project spectrum.list into published stages and GUI draft facts only."""
    return [
        SpectrumSummary(
            name=row["name"],
            type=row["spec_type"],
            stage=_stage(aligned=row["aligned"], completed=row["points_completed"]),
            point_count=row["point_count"],
            pending=row["pending"],
        )
        for row in reply.data["spectrums"]
    ]


def fit_summary(reply: ToolReply) -> FitSummary:
    """Keep availability/freshness from fit.result, omitting detailed fit data."""
    result = FitSummary(has_result=reply.data["has_result"], stale=reply.data["stale"])
    if result["stale"]:
        result["stale_reasons"] = reply.data["stale_reasons"]
    return result


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


def _axis(axis: AxisSnapshot) -> AxisSummary:
    return AxisSummary(
        count=axis["count"],
        min=_rounded(axis["minimum"]),
        max=_rounded(axis["maximum"]),
        unit=axis["unit"],
    )


def _range(values: Sequence[float]) -> list[float] | None:
    return [round(min(values), 3), round(max(values), 3)] if values else None


def spectrum_summary(reply: ToolReply) -> ToolReply:
    """Summarize a complete spectrum.snapshot without losing GUI provenance.

    Absent snapshots retain their native {name, exists:false} receipt. Published
    point arrays become count/ranges; raw axes become rounded extents. This does
    not read GUI state, infer inheritance, validate business data or retain it.
    """
    data = reply.data
    if not data["exists"]:
        return ToolReply(dict(data))
    return ToolReply(
        {
            "name": data["name"],
            "type": data["spec_type"],
            "stage": _stage(
                aligned=data["aligned"], completed=data["points_completed"]
            ),
            "pending": data["pending"],
            "axes": {
                "dev": _axis(data["raw_axes"]["dev_values"]),
                "freq": _axis(data["raw_axes"]["freqs"]),
            },
            "alignment": {
                "flux_half": _rounded(data["flux_half"]),
                "flux_int": _rounded(data["flux_int"]),
                "flux_period": _rounded(data["flux_period"]),
                "inherited_from": data["alignment"]["inherited_from"],
            },
            "points": {
                "count": len(data["points"]["freqs"]),
                "flux_range": _range(data["points"]["fluxs"]),
                "freq_range": _range(data["points"]["freqs"]),
            },
        }
    )


def with_png(summary: ToolReply, rendered: ToolReply) -> ToolReply:
    """Attach the producer PNG to summary content, without base64 text or files.

    Decode/integrity errors propagate. Accepted GUI work is never retried or
    rolled back when evidence delivery fails.
    """
    png = validated_png(base64.b64decode(rendered.data["png_b64"], validate=True))
    return ToolReply(
        summary.data, images=(*summary.images, png), is_error=summary.is_error
    )
