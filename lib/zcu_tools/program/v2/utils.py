from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, TypeGuard

import numpy as np
from numpy.typing import NDArray
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

    gen, ro = _encode_readout_freqs(
        soccfg,
        np.asarray(list(freqs_mhz), dtype=np.float64),
        gen_ch=gen_ch,
        ro_ch=ro_ch,
        mixer_freq=mixer_freq,
        nqz=nqz,
    )
    return gen.tolist(), ro.tolist()


@dataclass(frozen=True)
class _FrequencyArray:
    """Adapt MHz arrays to QICK's to_int hook without duplicating DDS rules."""

    values: NDArray[np.float64]

    def to_int(self, scale, quantize=1, parname=None, *, trunc=False):
        """Implement qick.helpers.to_int's parameter conversion protocol."""
        del parname  # QICK uses the name only for symbolic sweep parameters.
        rounding = np.trunc if trunc else np.round
        return np.asarray(
            quantize * rounding(self.values * scale / quantize), dtype=np.int64
        )

    def __gt__(self, limit: float) -> bool:
        return bool(np.any(self.values > limit))

    def __lt__(self, limit: float) -> bool:
        return bool(np.any(self.values < limit))

    def __float__(self) -> float:
        # QICK formats one representative value in its out-of-range diagnostic.
        return float(self.values.flat[0])


def _encode_readout_freqs(
    soccfg: QickConfig,
    freqs_mhz: NDArray[np.float64],
    *,
    gen_ch: int,
    ro_ch: int,
    mixer_freq: float | None,
    nqz: Literal[1, 2],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    rounded_mixer = _rounded_mixer_freq(
        soccfg,
        gen_ch=gen_ch,
        ro_ch=ro_ch,
        mixer_freq=mixer_freq,
        nqz=nqz,
    )
    absolute_freqs, flip_downconversion = _program_frequency_conventions()
    dds = freqs_mhz - rounded_mixer if absolute_freqs else freqs_mhz
    gen = np.asarray(
        soccfg.freq2reg(_FrequencyArray(dds), gen_ch=gen_ch, ro_ch=ro_ch),
        dtype=np.int64,
    )
    ro = np.asarray(
        soccfg.freq2reg_adc(_FrequencyArray(dds), ro_ch=ro_ch, gen_ch=gen_ch),
        dtype=np.int64,
    )
    ro = ro + int(soccfg.freq2reg_adc(rounded_mixer, ro_ch=ro_ch))
    if flip_downconversion:
        ro = -ro
    return gen % _UINT32_MODULUS, ro % _UINT32_MODULUS


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

    reference = (
        None
        if reference_freq_mhz is None
        else np.asarray([reference_freq_mhz], dtype=np.float64)
    )
    return float(
        readout_freqs_from_words(
            soccfg,
            np.asarray([word], dtype=np.int64),
            kind=kind,
            gen_ch=gen_ch,
            ro_ch=ro_ch,
            mixer_freq=mixer_freq,
            nqz=nqz,
            reference_freq_mhz=reference,
        )[0]
    )


def readout_freqs_from_words(
    soccfg: QickConfig,
    words: NDArray[np.int64],
    *,
    kind: FrequencyWordKind,
    gen_ch: int,
    ro_ch: int,
    mixer_freq: float | None,
    nqz: Literal[1, 2],
    reference_freq_mhz: NDArray[np.float64] | None = None,
    paired_ro_words: NDArray[np.int64] | None = None,
) -> NDArray[np.float64]:
    """Decode final hardware words into MHz, preserving the input array shape.

    Generator words resolve absolute frequencies. Readout words require a
    same-shaped MHz reference to select the nearest ADC-DDS alias. With kind=gen,
    paired_ro_words validates same-shaped ADC words by bit-exact re-encoding;
    inconsistent pairs raise ValueError. Mixer and sign conventions are shared
    with the scalar and iterable APIs.
    """
    rounded_mixer = _rounded_mixer_freq(
        soccfg,
        gen_ch=gen_ch,
        ro_ch=ro_ch,
        mixer_freq=mixer_freq,
        nqz=nqz,
    )
    raw = words % _UINT32_MODULUS
    absolute_freqs, flip_downconversion = _program_frequency_conventions()
    if kind == "gen":
        if absolute_freqs and soccfg["gens"][gen_ch]["has_mixer"]:
            modulus = 1 << int(soccfg["gens"][gen_ch]["b_dds"])
            raw = (raw + modulus // 2) % modulus - modulus // 2
        dds = np.asarray(soccfg.reg2freq(raw, gen_ch=gen_ch), dtype=np.float64)
    elif kind == "ro":
        if flip_downconversion:
            raw = -raw % _UINT32_MODULUS
        mixer_word = int(soccfg.freq2reg_adc(rounded_mixer, ro_ch=ro_ch))
        dds = np.asarray(
            soccfg.reg2freq_adc(
                (raw - mixer_word) % _UINT32_MODULUS,
                ro_ch=ro_ch,
            ),
            dtype=np.float64,
        )
    else:
        raise ValueError(f"unsupported frequency word kind: {kind!r}")
    freq = dds + rounded_mixer if absolute_freqs else dds
    if kind == "ro":
        if reference_freq_mhz is None or reference_freq_mhz.shape != words.shape:
            raise ValueError(
                "readout frequency words encode only an ADC-DDS alias; "
                "same-shaped reference_freq_mhz is required"
            )
        period = float(soccfg["readouts"][ro_ch]["f_dds"])
        freq = freq + np.rint((reference_freq_mhz - freq) / period) * period
    if paired_ro_words is not None:
        if kind != "gen" or paired_ro_words.shape != words.shape:
            raise ValueError("paired_ro_words requires same-shaped generator words")
        expected_gen, expected_ro = _encode_readout_freqs(
            soccfg,
            freq,
            gen_ch=gen_ch,
            ro_ch=ro_ch,
            mixer_freq=mixer_freq,
            nqz=nqz,
        )
        if np.any(words % _UINT32_MODULUS != expected_gen) or np.any(
            paired_ro_words % _UINT32_MODULUS != expected_ro
        ):
            raise ValueError("generator/readout frequency words are inconsistent")
    return freq


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

    return float(
        readout_freqs_from_words(
            soccfg,
            np.asarray([gen_word], dtype=np.int64),
            kind="gen",
            gen_ch=gen_ch,
            ro_ch=ro_ch,
            mixer_freq=mixer_freq,
            nqz=nqz,
            paired_ro_words=np.asarray([ro_word], dtype=np.int64),
        )[0]
    )


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
