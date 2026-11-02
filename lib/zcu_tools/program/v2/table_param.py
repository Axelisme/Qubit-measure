"""Pointwise QICK conversion for nonuniform top-level sweeps."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from qick.asm_v2 import QickRawParam
from qick.helpers import to_int


class TableParam:
    """Physical values belonging to one top-level loop.

    loop names the owning loop. values preserves point order, including repeats.
    Units are those of the consuming QICK parameter. Use to_int to quantize all
    points with QICK's scalar conversion, then get_rounded for physical values.
    """

    def __init__(self, loop: str, values: Sequence[float]) -> None:
        self.loop = loop
        self.values = tuple(values)
        self._raw: TableRawParam | None = None
        self._raw_scale: float = 1.0

    def is_sweep(self) -> bool:
        """Return True, even for a constant or single-point table."""
        return True

    def __float__(self) -> float:
        raise TypeError("a TableParam cannot be converted to a scalar float")

    @staticmethod
    def _scalar(value: object) -> float:
        if not isinstance(value, (int, float)):
            raise TypeError("table arithmetic requires an int or float scalar")
        return value

    def __add__(self, other: object) -> "TableParam":
        scalar = self._scalar(other)
        return TableParam(self.loop, [v + scalar for v in self.values])

    def __radd__(self, other: object) -> "TableParam":
        return self + other

    def __sub__(self, other: object) -> "TableParam":
        scalar = self._scalar(other)
        return TableParam(self.loop, [v - scalar for v in self.values])

    def __rsub__(self, other: object) -> "TableParam":
        scalar = self._scalar(other)
        return TableParam(self.loop, [scalar - v for v in self.values])

    def __mul__(self, other: object) -> "TableParam":
        scalar = self._scalar(other)
        return TableParam(self.loop, [v * scalar for v in self.values])

    def __rmul__(self, other: object) -> "TableParam":
        return self * other

    def __truediv__(self, other: object) -> "TableParam":
        scalar = self._scalar(other)
        return TableParam(self.loop, [v / scalar for v in self.values])

    def __rtruediv__(self, other: object) -> "TableParam":
        scalar = self._scalar(other)
        return TableParam(self.loop, [scalar / v for v in self.values])

    def __neg__(self) -> "TableParam":
        return self * -1

    def minval(self) -> float:
        """Return the smallest physical value across all points."""
        return min(self.values)

    def maxval(self) -> float:
        """Return the largest physical value across all points."""
        return max(self.values)

    def __lt__(self, other: float) -> bool:
        return self.minval() < other

    def __le__(self, other: float) -> bool:
        return self.minval() <= other

    def __gt__(self, other: float) -> bool:
        return self.maxval() > other

    def __ge__(self, other: float) -> bool:
        return self.maxval() >= other

    def to_int(
        self,
        scale: float,
        quantize: int = 1,
        parname: str | None = None,
        *,
        trunc: bool = False,
    ) -> "TableRawParam":
        """Convert each point using QICK scalar rounding and return raw words.

        scale converts physical units to words; quantize is the integer grid.
        trunc selects rounding toward zero instead of nearest. parname names
        the waveform field or register parameter used by the QICK manager.
        """
        self._raw = TableRawParam(
            self.loop,
            [
                to_int(v, scale, quantize=quantize, parname=parname, trunc=trunc)
                for v in self.values
            ],
            "" if parname is None else parname,
            quantize,
        )
        self._raw_scale = scale
        return self._raw

    def to_array(self) -> NDArray[np.float64]:
        """Return a detached one-dimensional array in point order."""
        return np.asarray(self.values, dtype=np.float64)

    def get_rounded(self, loop_counts: dict[str, int] | None = None) -> "TableParam":
        """Return quantized physical values; raise RuntimeError before to_int.

        loop_counts supplies the declared loop sizes, as in QICK get_rounded.
        """
        if self._raw is None:
            raise RuntimeError("to_int has not been called on this TableParam")
        raw = self._raw
        if loop_counts is not None:
            raw.to_steps(loop_counts)
        return TableParam(
            self.loop,
            [(v - raw.offset) / raw.scale / self._raw_scale for v in raw.values],
        )


class TableRawParam(QickRawParam):
    """Integer lookup values recognized by QICK waveform compilation.

    loop is the owning top-level loop; values is the ordered tuple of words.
    par names the QICK waveform field (or register parameter). start is the
    first word. spans is empty because lookup updates are not linear steps.
    quantize is the original integer rounding grid, not a lookup increment.
    """

    def __init__(
        self,
        loop: str,
        values: Sequence[int],
        par: str = "",
        quantize: int = 1,
    ) -> None:
        self.loop = loop
        self.values = tuple(values)
        super().__init__(par=par, start=self.values[0], spans={}, quantize=quantize)

    def minval(self) -> int:
        """Return the smallest raw word across all points."""
        return min(self.values)

    def maxval(self) -> int:
        """Return the largest raw word across all points."""
        return max(self.values)
