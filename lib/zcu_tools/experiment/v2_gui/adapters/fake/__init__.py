"""No-hardware fake / simulation adapters (grouped under a ``fake/`` submenu).

These are GUI-only stand-ins for trying the framework without a board — kept in
their own package so they don't clutter the real ``onetone`` / ``twotone``
experiment menus (the dropdown groups by registry key, so they register under
``fake/*``).
"""

from .freq import FakeFreqAdapter, FakeFreqAnalyzeResult, FakeFreqRunResult
from .stub import FakeAdapter, FakeAnalyzeParams

__all__ = [
    "FakeAdapter",
    "FakeAnalyzeParams",
    "FakeFreqAdapter",
    "FakeFreqAnalyzeResult",
    "FakeFreqRunResult",
]
