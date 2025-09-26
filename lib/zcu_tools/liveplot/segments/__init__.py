from .base import AbsSegment
from .histogram import HistogramSegment
from .plot1d import Plot1DSegment
from .plot2d import Plot2DSegment, PlotNonUniform2DSegment

__all__ = [
    "AbsSegment",
    "Plot1DSegment",
    "Plot2DSegment",
    "PlotNonUniform2DSegment",
    "HistogramSegment",
]
