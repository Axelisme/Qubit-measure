import warnings

import numpy as np


def map2adcfreq(soccfg, fpts: np.ndarray, gen_ch: int, ro_ch: int) -> np.ndarray:
    """
    Map frequencies to ADC frequencies.

    This function converts the input frequencies to ADC frequencies using the
    system configuration and checks for any duplicated frequencies after conversion.

    Args:
        soccfg: SocCfg object containing the system configuration
        fpts: Array of frequencies in MHz
        gen_ch: Generator channel number
        ro_ch: Readout channel number or array of readout channels

    Returns:
        Array of mapped ADC frequencies in Hz
    """
    fpts = soccfg.adcfreq(fpts, gen_ch=gen_ch, ro_ch=ro_ch)
    if len(set(fpts)) != len(fpts):
        warnings.warn(
            "Some frequencies are duplicated, you sweep step may be too small"
        )
    return fpts
