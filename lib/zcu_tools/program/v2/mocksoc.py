"""QICK channel queries for caller-provided compile profiles."""

from __future__ import annotations

from math import isfinite

from qick import QickConfig


def pfb_channel(soccfg: QickConfig, ro_ch: int, frequency_mhz: float) -> int:
    """Return QICK's PFB channel for an absolute physical frequency in MHz.

    soccfg is a caller-provided QICK config, for example from build_soccfg.
    ro_ch is a logical ADC index configured as a PFB output in that profile.
    frequency_mhz is a finite float, including any generator mixer contribution,
    not a DDS offset. QICK rounds it to the ADC frequency grid before Nyquist
    folding and channel assignment. No generator-grid rounding or downconversion
    sign flip is added. The returned int indexes the profile's PFB input channels,
    not its ADC outputs. No profile mutation, hardware access or acquisition occurs.

    Invalid/non-PFB ADC indices and nonfinite frequencies raise ValueError.
    Incomplete profile metadata propagates KeyError from QICK.
    """
    if type(ro_ch) is not int or not 0 <= ro_ch < len(soccfg["readouts"]):
        raise ValueError("PFB ADC must be inside the configured channel count")
    rocfg = soccfg["readouts"][ro_ch]
    if "pfb_nout" not in rocfg:
        raise ValueError(f"ADC {ro_ch} is not a PFB output")
    if not isfinite(frequency_mhz):
        raise ValueError("PFB frequency must be finite MHz")
    registers: dict[str, float | int] = {}
    soccfg.calc_ro_freq(
        rocfg, {"freq": frequency_mhz, "gen_ch": None}, registers, True, 0.0
    )
    return int(registers["pfb_ch"])
