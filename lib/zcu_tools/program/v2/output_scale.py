"""Full-scale peak ratios for native QICK pulse representations."""

from fractions import Fraction
from typing import Literal

from qick import QickConfig


def pulse_output_scale(
    soccfg: QickConfig, gen_ch: int, style: Literal["const", "flat_top", "arb"]
) -> float:
    """Return the ideal full-envelope peak relative to DDS const at equal gain.

    Pass the board's QickConfig, the generator channel index, and the native
    pulse style. This describes normalized gain before QICK rounds gain words,
    for a single-tone generator and the default output selection of each style.
    A full envelope reaches the QICK envelope limit; lower peaks are not
    normalized up. Flat-top uses the larger of its ramp peak and DDS plateau.
    Const is the reference and returns 1.0. To preserve peak when switching
    style, multiply gain by the old scale and divide by the new.

    Raises ValueError for an unknown style, a missing generator channel, or
    missing maxv or maxv_scale metadata, naming the corresponding field.
    """
    if style not in ("const", "flat_top", "arb"):
        raise ValueError(f"unsupported pulse style: {style!r}")
    try:
        generators = soccfg["gens"]
    except KeyError as error:
        raise ValueError("soccfg is missing 'gens' generator metadata") from error
    if not 0 <= gen_ch < len(generators):
        raise ValueError(f"gen_ch {gen_ch} is not in soccfg['gens']")
    gen = generators[gen_ch]
    for field in ("maxv", "maxv_scale"):
        if field not in gen:
            raise ValueError(f"soccfg generator {gen_ch} is missing {field!r}")

    if style == "const":
        return 1.0
    headroom = gen["maxv_scale"]

    # ADR-0072: QICK pulse compilation owns this ratio, not a simulation model.
    # The product output takes DDS × envelope bits [31:16]. Arb and ramp gain
    # words use maxv, whereas const gain words include the headroom factor.
    product = float(soccfg.get_maxv(gen_ch)) / (65536 * headroom)
    if style == "arb":
        return product

    # QICK generates a separate DDS plateau; match its scale, not the integer
    # quantization of the ramp or plateau gain words.
    plateau = Fraction(1, 2) * Fraction(headroom).limit_denominator(20)
    return max(product, float(plateau) / headroom)
