"""Shared compiled mux parameter extraction for the two timeline interpreters."""

from dataclasses import dataclass

from ..base import MyProgramV2
from ..modular import ModularProgramV2
from ..modules.readout import MuxReadout


@dataclass(frozen=True)
class MuxToneParameters:
    """Rounded hardware parameters for one tone and its ADC.

    channel is the logical ADC index. frequency_mhz and demodulation_mhz are
    absolute carriers, not PFB aliases. phase_deg and gain are the compiled
    generator values. demodulation_phase_deg is the static ADC phase.
    """

    channel: int
    frequency_mhz: float
    phase_deg: float
    gain: float
    demodulation_mhz: float
    demodulation_phase_deg: float


def mux_parameters(
    prog: ModularProgramV2, module: MuxReadout
) -> tuple[MuxToneParameters, ...]:
    """Read compiled mux tones and round static demodulators with QICK itself.

    prog must contain the initialized module. Returned order is cfg tone order.
    Uses copies of readout registers, never configuring hardware or modifying
    program metadata. Missing compiled parameters propagate as KeyError.
    """
    gen = prog.gen_chs[module.cfg.gen_ch]
    mixer = float(gen["mixer_freq"]["rounded"])
    result = []
    for tone, compiled in zip(module.cfg.tones, gen["mux_tones"], strict=True):
        frequency, phase = static_readout_parameters(prog, tone.ro_ch, mixer_mhz=mixer)
        result.append(
            MuxToneParameters(
                tone.ro_ch,
                float(compiled["freq_rounded"]),
                float(compiled["phase_rounded"]),
                float(compiled["gain_rounded"]),
                frequency,
                phase,
            )
        )
    return tuple(result)


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
