"""Clock conversions for independent, caller-authored timing expectations.

These helpers only quantize physical durations on public QICK clock grids.
Tests compose scheduling, repeat and shell semantics at their own call sites.
"""

from fractions import Fraction
from functools import cache
from math import ceil

import numpy as np
from qick import QickConfig


def generator_duration(soccfg: QickConfig, length_us: float, ch: int = 0) -> Fraction:
    """Return the exact physical duration after native generator quantization."""
    cycles = soccfg.us2cycles(length_us, gen_ch=ch)
    return Fraction(int(cycles)) / Fraction(str(soccfg["gens"][ch]["f_fabric"]))


def generator_tproc_grid(soccfg: QickConfig, ch: int = 0) -> int:
    """Smallest generator-cycle step spanning a whole number of tProc cycles."""
    ratio = Fraction(str(soccfg["tprocs"][0]["f_time"])) / Fraction(
        str(soccfg["gens"][ch]["f_fabric"])
    )
    return ratio.denominator


def readout_duration(soccfg: QickConfig, length_us: float, ch: int = 0) -> Fraction:
    """Return the exact ADC window after native output-clock quantization."""
    cycles = soccfg.us2cycles(length_us, ro_ch=ch)
    return Fraction(int(cycles)) / Fraction(str(soccfg["readouts"][ch]["f_output"]))


def tproc_ceil(soccfg: QickConfig, duration_us: Fraction | float) -> int:
    """Round an exact physical duration upward onto the tProc clock grid."""
    return ceil(
        Fraction(str(duration_us)) * Fraction(str(soccfg["tprocs"][0]["f_time"]))
    )


@cache
def signed_power_cost(value: int) -> int:
    """Minimum number of signed powers of two summing to a nonnegative integer."""
    if value <= 1:
        return value
    if value % 2 == 0:
        return signed_power_cost(value // 2)
    return 1 + min(signed_power_cost(value // 2), signed_power_cost(value // 2 + 1))


@cache
def runtime_weight(tproc_mhz: float, fabric_mhz: float) -> int:
    """Find the least-CSD k=8 upper weight meeting the public 500-us budget.

    Enumerate the finite admissible interval and check every legal integer
    length. No scheduled output or compiler implementation supplies this oracle.
    """
    tproc = Fraction(str(tproc_mhz))
    ratio = tproc / Fraction(str(fabric_mhz))
    maximum = int(500 * fabric_mhz)
    counts = np.arange(maximum + 1, dtype=np.int64)
    admissible = []
    for candidate in range(ceil(256 * ratio), int(256 * (ratio + tproc / maximum)) + 2):
        error = (
            (counts * candidate + 255) // 256
        ) * ratio.denominator - counts * ratio.numerator
        budget = tproc * ratio.denominator
        if np.all(error >= 0) and np.all(
            error <= budget.numerator // budget.denominator
        ):
            admissible.append(candidate)
    if not admissible:
        raise ValueError("no k=8 upper weight satisfies the public timing budget")
    return min(admissible, key=lambda weight: (signed_power_cost(weight), weight))


def runtime_cycles(soccfg: QickConfig, length_cycles: int, ch: int = 0) -> Fraction:
    """Return runtime generator duration in fractional tProc cycles before post/ceil."""
    weight = runtime_weight(
        soccfg["tprocs"][0]["f_time"], soccfg["gens"][ch]["f_fabric"]
    )
    return Fraction(length_cycles * weight, 256)


def native_sweep(
    soccfg: QickConfig,
    start_us: float,
    stop_us: float,
    count: int,
    *,
    gen_ch: int | None = None,
) -> list[int]:
    """Public QICK sweep rule: round start/span, truncate step toward zero."""
    start = int(soccfg.us2cycles(start_us, gen_ch=gen_ch))
    span = int(soccfg.us2cycles(stop_us - start_us, gen_ch=gen_ch))
    step = int(span / (count - 1))
    return [start + i * step for i in range(count)]
