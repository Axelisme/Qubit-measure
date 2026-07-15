from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Literal, Protocol, TypeGuard

from qick import QickConfig
from qick.asm_v2 import QickParam, QickSweep1D

from .sweep import SweepCfg

logger = logging.getLogger(__name__)

_UINT32_MODULUS = 2**32
FrequencyWordKind = Literal["gen", "ro"]


class QickParamLike(Protocol):
    start: float
    spans: Mapping[str, float]

    def is_sweep(self) -> bool: ...

    def minval(self) -> float: ...

    def maxval(self) -> float: ...

    def to_array(
        self, loop_counts: Mapping[str, int], *, all_loops: bool = False
    ) -> object: ...


def is_qick_param(value: object) -> TypeGuard[QickParamLike]:
    return isinstance(value, QickParam)


def sweep2param(name: str, sweep: SweepCfg) -> QickParam:
    """
    Convert formatted sweep dictionary to a QickSweep1D parameter.

    This function creates a QickSweep1D parameter from a formatted sweep dictionary,
    which is used in Qick v2 assembly programming.

    Args:
        name: Name of the sweep parameter
        sweep: Dictionary containing 'start' and 'stop' values for the sweep

    Returns:
        QickSweep1D: Qick v2 sweep parameter object
    """

    # convert formatted sweep to qick v2 sweep param
    return QickSweep1D(name, sweep.start, sweep.stop)


def param2str(param: float | QickParam) -> str:
    """Convert a parameter to a string."""
    if isinstance(param, QickParam):
        if not is_qick_param(param):
            raise TypeError(
                f"unsupported QickParam implementation: {type(param).__name__}"
            )
        if param.is_sweep():
            return f"sweep({param.minval():.3f}, {param.maxval():.3f})"
        else:
            return f"{param.start:.3f}"
    return f"{float(param):.3f}"


def readout_freq_words(
    soccfg: QickConfig,
    freqs_mhz: Iterable[float],
    *,
    gen_ch: int,
    ro_ch: int,
    mixer_freq: float | None,
    nqz: Literal[1, 2],
) -> tuple[list[int], list[int]]:
    """Encode absolute readout frequencies as final generator/readout words.

    The returned unsigned 32-bit patterns are ready for ``LoadWord`` and runtime
    wave-register playback. They include QICK's absolute-frequency mixer
    correction and readout downconversion sign, rather than the bare output of
    ``freq2reg`` or ``freq2reg_adc``.
    """

    rounded_mixer = _rounded_mixer_freq(
        soccfg,
        gen_ch=gen_ch,
        ro_ch=ro_ch,
        mixer_freq=mixer_freq,
        nqz=nqz,
    )
    gen_words: list[int] = []
    ro_words: list[int] = []
    absolute_freqs, flip_downconversion = _program_frequency_conventions()
    for freq_mhz in freqs_mhz:
        absolute_freq = float(freq_mhz)
        dds_freq = absolute_freq - rounded_mixer if absolute_freqs else absolute_freq
        gen_word = int(soccfg.freq2reg(dds_freq, gen_ch=gen_ch, ro_ch=ro_ch))
        ro_word = int(soccfg.freq2reg_adc(dds_freq, ro_ch=ro_ch, gen_ch=gen_ch)) + int(
            soccfg.freq2reg_adc(rounded_mixer, ro_ch=ro_ch)
        )
        if flip_downconversion:
            ro_word *= -1
        gen_words.append(gen_word % _UINT32_MODULUS)
        ro_words.append(ro_word % _UINT32_MODULUS)
    return gen_words, ro_words


def readout_freq_from_word(
    soccfg: QickConfig,
    word: int,
    *,
    kind: FrequencyWordKind,
    gen_ch: int,
    ro_ch: int,
    mixer_freq: float | None,
    nqz: Literal[1, 2],
    reference_freq_mhz: float | None = None,
) -> float:
    """Decode one final hardware frequency word to a semantic MHz value.

    Generator words identify the semantic frequency directly. Readout words only
    identify an ADC-DDS alias, so ``reference_freq_mhz`` is required to unwrap the
    nearest alias. This is valid when the actual frequency is within half an ADC
    DDS period of the reference.
    """

    rounded_mixer = _rounded_mixer_freq(
        soccfg,
        gen_ch=gen_ch,
        ro_ch=ro_ch,
        mixer_freq=mixer_freq,
        nqz=nqz,
    )
    raw_word = int(word) % _UINT32_MODULUS
    absolute_freqs, flip_downconversion = _program_frequency_conventions()
    if kind == "gen":
        dds_word = raw_word
        if absolute_freqs and soccfg["gens"][gen_ch]["has_mixer"]:
            dds_word = _as_signed_dds_word(
                dds_word, bits=int(soccfg["gens"][gen_ch]["b_dds"])
            )
        dds_freq = float(soccfg.reg2freq(dds_word, gen_ch=gen_ch))
    elif kind == "ro":
        if flip_downconversion:
            raw_word = -raw_word % _UINT32_MODULUS
        mixer_word = int(soccfg.freq2reg_adc(rounded_mixer, ro_ch=ro_ch))
        dds_word = (raw_word - mixer_word) % _UINT32_MODULUS
        dds_freq = float(soccfg.reg2freq_adc(dds_word, ro_ch=ro_ch))
    else:
        raise ValueError(f"unsupported frequency word kind: {kind!r}")

    semantic_freq = dds_freq + rounded_mixer if absolute_freqs else dds_freq
    if kind == "ro":
        if reference_freq_mhz is None:
            raise ValueError(
                "readout frequency words encode only an ADC-DDS alias; "
                "reference_freq_mhz is required"
            )
        period_mhz = float(soccfg["readouts"][ro_ch]["f_dds"])
        alias_index = round((reference_freq_mhz - semantic_freq) / period_mhz)
        semantic_freq += alias_index * period_mhz
    return semantic_freq


def readout_freq_from_words(
    soccfg: QickConfig,
    gen_word: int,
    ro_word: int,
    *,
    gen_ch: int,
    ro_ch: int,
    mixer_freq: float | None,
    nqz: Literal[1, 2],
) -> float:
    """Decode a generator word and validate its paired readout word bit-exactly."""

    freq_mhz = readout_freq_from_word(
        soccfg,
        gen_word,
        kind="gen",
        gen_ch=gen_ch,
        ro_ch=ro_ch,
        mixer_freq=mixer_freq,
        nqz=nqz,
    )
    expected_gen, expected_ro = readout_freq_words(
        soccfg,
        [freq_mhz],
        gen_ch=gen_ch,
        ro_ch=ro_ch,
        mixer_freq=mixer_freq,
        nqz=nqz,
    )
    actual_gen = int(gen_word) % _UINT32_MODULUS
    actual_ro = int(ro_word) % _UINT32_MODULUS
    if actual_gen != expected_gen[0] or actual_ro != expected_ro[0]:
        raise ValueError(
            "generator/readout frequency words are inconsistent: "
            f"got gen=0x{actual_gen:08x}, ro=0x{actual_ro:08x}; "
            f"expected gen=0x{expected_gen[0]:08x}, ro=0x{expected_ro[0]:08x}"
        )
    return freq_mhz


def _rounded_mixer_freq(
    soccfg: QickConfig,
    *,
    gen_ch: int,
    ro_ch: int,
    mixer_freq: float | None,
    nqz: Literal[1, 2],
) -> float:
    gencfg = soccfg["gens"][gen_ch]
    if not gencfg["has_mixer"]:
        return 0.0
    if mixer_freq is None:
        raise ValueError(
            f"generator {gen_ch} has a digital mixer but mixer_freq is None"
        )
    mixer_cfg = soccfg.calc_mixer_freq(gen_ch, mixer_freq, nqz, ro_ch)
    return float(mixer_cfg["rounded"])


def _program_frequency_conventions() -> tuple[bool, bool]:
    # Lazy import avoids utils -> base -> macro -> utils during package import.
    from .base import MyProgramV2

    return MyProgramV2.ABSOLUTE_FREQS, MyProgramV2.FLIP_DOWNCONVERSION


def _as_signed_dds_word(word: int, *, bits: int) -> int:
    modulus = 1 << bits
    wrapped = word % modulus
    if wrapped >= modulus // 2:
        return wrapped - modulus
    return wrapped
