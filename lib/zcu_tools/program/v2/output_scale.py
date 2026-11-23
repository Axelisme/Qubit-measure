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
    headroom = _generator_headroom(soccfg, gen_ch)
    if style == "const":
        return 1.0

    # ADR-0072: QICK pulse compilation owns this ratio, not a simulation model.
    # The product output takes DDS × envelope bits [31:16]. Arb and ramp gain
    # words use maxv, whereas const gain words include the headroom factor.
    product = float(soccfg.get_maxv(gen_ch)) / (65536 * headroom)
    if style == "arb":
        return product
    return max(product, _plateau_scale(headroom))


def flat_top_plateau_scale(soccfg: QickConfig, gen_ch: int) -> float:
    """Return a flat-top's ideal DDS plateau relative to const at equal gain.

    Use the board's QickConfig and a single-tone generator channel. The ratio
    excludes gain word quantization and describes only the plateau, even if
    the product ramp has a higher full-scale peak. An absent channel or missing
    generator scale metadata raises ValueError naming the field.
    """
    return _plateau_scale(_generator_headroom(soccfg, gen_ch))


def _generator_headroom(soccfg: QickConfig, gen_ch: int) -> float:
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

    return gen["maxv_scale"]


def _plateau_scale(headroom: float) -> float:
    # QICK generates a separate DDS plateau with this rational gain factor.
    plateau = Fraction(1, 2) * Fraction(headroom).limit_denominator(20)
    return float(plateau) / headroom
