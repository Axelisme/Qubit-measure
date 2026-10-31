"""Directed source ranges and one-axis bounds, independent of runtime recipes."""

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class VariableRange:
    """A source ID, directed cycle endpoints and per-source-axis deviations.

    source is a loop name, otherwise an independent identity. Loop start/end
    follow iteration order. deviation_minimum/maximum hold exact fractional
    distances from that line at each loop point; affine sources use zeros.
    Independent values use ordered low/high endpoints and singleton zeros.
    None marks an unavailable endpoint; a known independent lower endpoint
    still certifies nonnegativity, without inventing a finite upper bound.
    """

    source: str | object
    start: int | None
    end: int | None
    deviation_minimum: tuple[Fraction, ...] = (Fraction(0),)
    deviation_maximum: tuple[Fraction, ...] = (Fraction(0),)


def source_limits(
    terms: list[tuple[VariableRange, int | Fraction]],
) -> tuple[Fraction | None, Fraction | None]:
    """Sum one ID's coefficient-weighted cycle ranges along its own axis.

    Loop terms have matching axis lengths. Independent IDs have ordered
    endpoints. Missing endpoints propagate in the coefficient's direction.
    No Cartesian grid or runtime producer metadata participates in this rule.
    """
    first = terms[0][0]
    if not isinstance(first.source, str):
        lower: Fraction | None = Fraction(0)
        upper: Fraction | None = Fraction(0)
        for r, coefficient in terms:
            a, b = (r.start, r.end) if coefficient > 0 else (r.end, r.start)
            lower = None if lower is None or a is None else lower + coefficient * a
            upper = None if upper is None or b is None else upper + coefficient * b
        return lower, upper
    count = len(first.deviation_minimum)
    lows = [Fraction(0)] * count
    highs = [Fraction(0)] * count
    for r, coefficient in terms:
        if r.start is None or r.end is None:
            return None, None
        for i in range(count):
            line = Fraction(r.start) + Fraction((r.end - r.start) * i, max(1, count - 1))
            a = coefficient * (line + r.deviation_minimum[i])
            b = coefficient * (line + r.deviation_maximum[i])
            lows[i] += min(a, b)
            highs[i] += max(a, b)
    return min(lows), max(highs)
