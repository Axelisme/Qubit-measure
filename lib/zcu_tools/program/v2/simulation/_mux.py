"""Resolve static ADC demodulation parameters using QICK's compiled config."""

from ..base import MyProgramV2


def static_readout_parameters(
    prog: MyProgramV2, channel: int, *, mixer_mhz: float
) -> tuple[float, float]:
    """Return resolved static ADC demodulation MHz/degrees without hardware I/O.

    channel indexes a declared non-tProc-controlled ADC; its ro_config must have
    been compiled by QICK. mixer_mhz is the generator's already-rounded mixer.
    Missing declaration/configuration propagates as KeyError.
    """
    ro = prog.ro_chs[channel]
    registers = ro["ro_config"].copy()
    prog.soccfg.calc_ro_freq(
        prog.soccfg["readouts"][channel],
        ro,
        registers,
        prog.ABSOLUTE_FREQS,
        mixer_mhz,
        prog.FLIP_DOWNCONVERSION,
    )
    return float(registers["f_rounded"]), float(registers["phase_rounded"])
