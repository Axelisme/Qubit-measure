"""Pointwise QICK conversion for nonuniform top-level sweeps."""

from collections.abc import Sequence
from fractions import Fraction
from math import isfinite
from typing import Never, override

import numpy as np
from numpy.typing import NDArray
from qick.asm_v2 import QickParam, QickRawParam
from qick.helpers import to_int


class TableParam(QickParam):
    """Physical values belonging to one top-level loop.

    loop names the owning loop. values preserves point order, including repeats.
    Units are those of the consuming QICK parameter. Use to_int to quantize all
    points with QICK's scalar conversion, then get_rounded for physical values.
    """

    def __init__(self, loop: str, values: Sequence[float]) -> None:
        self.loop = loop
        self.values = tuple(values)
        if not self.values:
            raise ValueError(f"TableParam loop {loop!r} values must not be empty")
        if not all(isfinite(v) for v in self.values):
            raise ValueError(f"TableParam loop {loop!r} values must be finite")
        self._raw: TableRawParam | None = None
        self._raw_scale: float = 1.0

    def __getattr__(self, name: str) -> Never:
        if name in {"start", "spans", "raw_param", "derived_param"}:
            raise TypeError(f"TableParam loop {self.loop!r} has no affine {name}")
        raise AttributeError(name)

    def __repr__(self) -> str:
        return f"TableParam(loop={self.loop!r}, values={self.values!r})"

    def __str__(self) -> str:
        return repr(self)

    def __copy__(self) -> "TableParam":
        return TableParam(self.loop, self.values)

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

    @override
    def to_int(
        self,
        scale: float,
        quantize: int = 1,
        parname: str | None = None,
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

    @override
    def to_array(
        self, loop_counts: dict[str, int] | None = None, all_loops: bool = False
    ) -> NDArray[np.float64]:
        """Return a detached array in point order.

        With loop_counts and all_loops, retain every unrelated loop as a
        singleton dimension in declaration order, matching QICK's convention.
        Without all_loops, return the one-dimensional lookup values.
        """
        values = np.asarray(self.values, dtype=np.float64)
        if loop_counts is not None and all_loops:
            shape = [
                len(self.values) if name == self.loop else 1 for name in loop_counts
            ]
            return values.reshape(shape)
        return values

    def get_rounded(self, loop_counts: dict[str, int] | None = None) -> "TableParam":
        """Return quantized physical values; raise RuntimeError before to_int.

        loop_counts supplies the declared loop sizes, as in QICK get_rounded.
        """
        if self._raw is None:
            raise RuntimeError("to_int has not been called on this TableParam")
        raw = self._raw
        if loop_counts is not None and raw.steps is None:
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

    def to_steps(self, loops: dict[str, int]) -> None:
        """Validate the owning loop and point count; produce no linear steps.

        Unknown loops or a table length unlike the declared count raise
        ValueError. Repeated calls leave steps empty without changing values.
        """
        if self.loop not in loops:
            raise ValueError(f"table loop {self.loop!r} is not declared")
        if loops[self.loop] != len(self.values):
            raise ValueError(
                f"table loop {self.loop!r} has {loops[self.loop]} iterations "
                f"but the table has {len(self.values)} points"
            )
        self.steps = {}

    def __copy__(self) -> "TableRawParam":
        copied = TableRawParam(self.loop, self.values, self.par, self.quantize)
        copied.steps = None if self.steps is None else self.steps.copy()
        copied.scale = self.scale
        copied.offset = self.offset
        return copied

    def __imul__(self, factor: object) -> "TableRawParam":
        if not isinstance(factor, (int, Fraction)):
            raise TypeError("raw tables can only be multiplied by int or Fraction")
        if isinstance(factor, Fraction) and any(
            v % factor.denominator for v in self.values
        ):
            raise ValueError("Fraction must divide every raw table point evenly")
        self.values = tuple(int(v * factor) for v in self.values)
        self.start = self.values[0]
        self.scale *= factor
        self.offset *= factor
        return self

    def __iadd__(self, offset: object) -> "TableRawParam":
        if not isinstance(offset, int):
            raise TypeError("raw table offsets must be integers")
        self.values = tuple(v + offset for v in self.values)
        self.start = self.values[0]
        self.offset += offset
        return self

    def __truediv__(self, divisor: float) -> TableParam:
        return TableParam(self.loop, [v / divisor for v in self.values])

    def to_rounded(self) -> TableParam:
        """Undo manager scale/offset, retaining initial quantized raw units."""
        return TableParam(
            self.loop, [float((v - self.offset) / self.scale) for v in self.values]
        )

    def minval(self) -> int:
        """Return the smallest raw word across all points."""
        return min(self.values)

    def maxval(self) -> int:
        """Return the largest raw word across all points."""
        return max(self.values)
