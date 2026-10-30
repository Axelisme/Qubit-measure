from __future__ import annotations

import math
from collections.abc import Callable
from numbers import Number
from typing import Any, Literal, TypeVar

import numpy as np
from numpy.typing import NDArray
from qick import QickConfig
from qick.asm_v2 import QickParam, QickSweep1D
from qick.helpers import to_int

from zcu_tools.program.v2 import SweepCfg

T_Value = TypeVar("T_Value", NDArray[np.float64], float)


def round_zcu_time(
    us: T_Value,
    soccfg: QickConfig,
    gen_ch: int | None = None,
    ro_ch: int | None = None,
    scaler: float = 1.0,
) -> T_Value:
    """Round scalar times or independent time points to the selected QICK clock.

    Times are microseconds; ``scaler`` converts the caller's value to that time.
    Regular hardware sweeps must use ``round_sweep_dict`` instead, because QICK
    rounds their start/span before truncating the signed step toward zero.
    """

    def _convert_time(t: float) -> float:
        return (
            soccfg.cycles2us(
                soccfg.us2cycles(scaler * t, gen_ch=gen_ch, ro_ch=ro_ch),
                gen_ch=gen_ch,
                ro_ch=ro_ch,
            )
            / scaler
        )

    if isinstance(us, (Number, float)):
        return _convert_time(float(us))
    return np.vectorize(_convert_time)(us)


def round_zcu_freq(
    freq: T_Value,
    soccfg: QickConfig,
    gen_ch: int,
    ro_ch: int | None = None,
    scaler: float = 1.0,
    *,
    mixer_freq: float | None = None,
) -> T_Value:
    """Round independent frequencies using QICK's signed DDS conversion.

    Values are MHz after multiplication by finite nonzero ``scaler``. With
    ``mixer_freq`` (requested MHz), values are absolute RF frequencies and the
    quantized mixer is subtracted before DDS conversion. Without it, values
    are DDS frequencies. ``ro_ch`` requests DAC/ADC grid matching. Regular
    hardware sweeps must use ``round_sweep_dict`` for signed step rounding.
    QICK rejects frequencies outside an interpolated generator's DDS range.
    """

    def _convert_freq(f: float) -> float:
        param = QickParam(f)
        _convert_frequency_param(param, soccfg, gen_ch, ro_ch, scaler, mixer_freq)
        return float(param.get_actual_values({}))

    if isinstance(freq, (Number, float)):
        return _convert_freq(float(freq))
    return np.vectorize(_convert_freq)(freq)


def _convert_frequency_param(
    param: QickParam,
    soccfg: QickConfig,
    gen_ch: int,
    ro_ch: int | None,
    scaler: float,
    mixer_freq: float | None,
) -> None:
    if not math.isfinite(scaler) or scaler == 0.0:
        raise ValueError("frequency scaler must be finite and nonzero")
    mixer = 0.0
    if mixer_freq is not None:
        if not soccfg["gens"][gen_ch]["has_mixer"]:
            raise ValueError("mixer_freq requires a generator with a mixer")
        # NQZ only affects setval; rounded is the absolute-frequency offset.
        mixer = soccfg.calc_mixer_freq(gen_ch, mixer_freq, nqz=1, ro_ch=ro_ch)[
            "rounded"
        ]
    soccfg.freq2reg(scaler * param - mixer, gen_ch=gen_ch, ro_ch=ro_ch)


def _round_frequency_sweep(
    sweep: SweepCfg,
    soccfg: QickConfig,
    gen_ch: int,
    ro_ch: int | None = None,
    scaler: float = 1.0,
    mixer_freq: float | None = None,
) -> SweepCfg:
    param = QickParam(sweep.start, {"sweep": sweep.stop - sweep.start})
    _convert_frequency_param(param, soccfg, gen_ch, ro_ch, scaler, mixer_freq)
    try:
        values = param.get_actual_values({"sweep": sweep.expts})
    except RuntimeError as exc:
        raise ValueError(
            f"Frequency sweep hardware quantization failed: {exc}"
        ) from exc
    start, stop = float(values[0]), float(values[-1])
    step = (stop - start) / (sweep.expts - 1) if sweep.expts > 1 else 0.0
    return SweepCfg(start=start, stop=stop, expts=sweep.expts, step=step)


def round_zcu_phase(
    phase: T_Value,
    soccfg: QickConfig,
    gen_ch: int,
    ro_ch: int | None = None,
    scaler: float = 1.0,
) -> T_Value:
    one_gain = soccfg.reg2deg(1, gen_ch=gen_ch, ro_ch=ro_ch) - soccfg.reg2deg(
        0, gen_ch=gen_ch, ro_ch=ro_ch
    )

    def _convert_phase(p: float) -> float:
        num_2pi = int(scaler * p / 360.0)
        return (
            360.0 * num_2pi  # restore the wrapped-around phase to the original value
            + soccfg.reg2deg(
                soccfg.deg2reg(scaler * p - 0.5 * one_gain, gen_ch=gen_ch, ro_ch=ro_ch),
                gen_ch=gen_ch,
                ro_ch=ro_ch,
            )
            / scaler
        )

    if isinstance(phase, (Number, float)):
        return _convert_phase(float(phase))
    return np.vectorize(_convert_phase)(phase)


def round_zcu_gain(
    gain: T_Value, soccfg: QickConfig, gen_ch: int, scaler: float = 1.0
) -> T_Value:
    """Return scalar or array gain coordinates that compile to their target words.

    ``gen_ch`` selects a generator in ``soccfg``. Gains are normalized fractions
    after multiplication by finite nonzero ``scaler`` (negative is allowed).
    Target words retain truncation toward zero using ``soccfg.get_maxv``.
    Each returned float is nearest to word / (scaler * maxv) and reproduces
    that word through QICK's constant-pulse conversion. Array shape is retained.
    Zero words return 0.0. Raise ValueError for invalid scaler or generator scale,
    or RuntimeError if eight outward ULP adjustments cannot reproduce a word.
    """
    maxv, qick_scale = _gain_scales(soccfg, gen_ch, scaler)

    def _convert_gain(g: float) -> float:
        word = int(np.trunc(scaler * g * maxv))
        return _stable_gain(word, maxv, scaler, qick_scale)

    if isinstance(gain, (Number, float)):
        return _convert_gain(float(gain))
    return np.vectorize(_convert_gain)(gain)


def _gain_scales(soccfg: QickConfig, gen_ch: int, scaler: float) -> tuple[int, float]:
    if not math.isfinite(scaler) or scaler == 0.0:
        raise ValueError("gain scaler must be finite and nonzero")
    maxv = soccfg.get_maxv(gen_ch)
    gen = soccfg["gens"][gen_ch]
    qick_scale = gen["maxv"] * gen["maxv_scale"]
    if maxv <= 0 or not math.isfinite(qick_scale) or qick_scale <= 0:
        raise ValueError("generator gain scale must be finite and positive")
    return maxv, qick_scale


def _stable_gain(word: int, maxv: float, scaler: float, qick_scale: float) -> float:
    if word == 0:
        return 0.0
    candidate = word / (scaler * maxv)
    direction = math.copysign(math.inf, candidate)
    # Search in float order so the first match is the nearest to the ideal.
    for adjustment in range(9):
        if to_int(scaler * candidate, qick_scale, parname="gain", trunc=True) == word:
            return float(candidate)
        if adjustment == 8:
            break
        candidate = np.nextafter(candidate, direction)
    raise RuntimeError(f"Gain word {word} is not stable within eight ULP adjustments")


def _gain_sweep_words(sweep: SweepCfg, maxv: int, scaler: float) -> tuple[int, int]:
    start = int(np.trunc(scaler * sweep.start * maxv))
    if sweep.expts == 1:
        return start, 0
    requested_step = (sweep.stop - sweep.start) / (sweep.expts - 1)
    step = int(np.trunc(scaler * requested_step * maxv))
    if step == 0:
        raise ValueError(_format_zero_step_error(sweep, "gain", requested_step, 0.0))
    return start, step


def _host_gain_grid(
    sweep: SweepCfg, soccfg: QickConfig, gen_ch: int, scaler: float = 1.0
) -> NDArray[np.float64]:
    maxv, qick_scale = _gain_scales(soccfg, gen_ch, scaler)
    start, step = _gain_sweep_words(sweep, maxv, scaler)
    return np.asarray(
        [
            _stable_gain(start + i * step, maxv, scaler, qick_scale)
            for i in range(sweep.expts)
        ],
        dtype=np.float64,
    )


def _round_gain_sweep(
    sweep: SweepCfg, soccfg: QickConfig, gen_ch: int, scaler: float = 1.0
) -> SweepCfg:
    maxv, qick_scale = _gain_scales(soccfg, gen_ch, scaler)
    start_word, step_word = _gain_sweep_words(sweep, maxv, scaler)
    start = _stable_gain(start_word, maxv, scaler, qick_scale)
    if sweep.expts == 1:
        return SweepCfg(start=start, stop=start, expts=1, step=0.0)
    span_word = (sweep.expts - 1) * step_word
    # Stabilize the span before endpoint addition: near cancellation, stop's
    # ULP can be too small to change the span recovered by stop - start.
    span = _stable_gain(span_word, qick_scale, scaler, qick_scale)
    stop = start + span
    direction = math.copysign(math.inf, span_word / scaler)
    # QickSweep1D subtracts endpoints before scaling its span. Stabilizing the
    # step alone cannot protect that subtraction or the subsequent truncation.
    for adjustment in range(9):
        param = scaler * QickSweep1D("gain", start, stop)
        raw = param.to_int(qick_scale, quantize=1, parname="gain", trunc=True)
        if raw.start == start_word and raw.spans["gain"] == span_word:
            return SweepCfg(
                start=start,
                stop=stop,
                expts=sweep.expts,
                step=(stop - start) / (sweep.expts - 1),
            )
        if adjustment == 8:
            break
        stop = float(np.nextafter(stop, direction))
    raise RuntimeError(
        f"Gain sweep span word {span_word} is not stable within eight ULP adjustments"
    )


def apply_round(
    val: T_Value,
    round_type: Literal["none", "time", "freq", "phase", "gain"],
    round_info: dict[str, Any],
) -> T_Value:
    ROUND_FN_MAP: dict[str, Callable[..., T_Value]] = {
        "none": lambda v: v,
        "time": round_zcu_time,
        "freq": round_zcu_freq,
        "phase": round_zcu_phase,
        "gain": round_zcu_gain,
    }

    if round_fn := ROUND_FN_MAP.get(round_type):
        return round_fn(val, **round_info)
    raise ValueError(f"Invalid round type: {round_type}")


def _format_zero_step_error(
    sweep: SweepCfg,
    round_type: str,
    requested_step: float,
    rounded_step: float,
) -> str:
    reason = (
        "sweep step is effectively zero"
        if round_type == "none"
        else "sweep step becomes zero after hardware quantization"
    )
    return (
        f"{reason}: "
        f"round_type={round_type!r}, start={sweep.start}, stop={sweep.stop}, "
        f"expts={sweep.expts}, requested_step={requested_step}, "
        f"rounded_step={rounded_step}. Increase the sweep span or reduce expts so "
        "adjacent points remain distinct on the hardware grid."
    )


def _round_time_sweep(
    sweep: SweepCfg,
    soccfg: QickConfig,
    gen_ch: int | None = None,
    ro_ch: int | None = None,
    scaler: float = 1.0,
) -> SweepCfg:
    if not math.isfinite(scaler) or scaler == 0.0:
        raise ValueError("time sweep scaler must be finite and nonzero")
    param = QickParam(sweep.start, {"sweep": sweep.stop - sweep.start})
    # QICK records the conversion on the parameter for get_actual_values().
    soccfg.us2cycles(scaler * param, gen_ch=gen_ch, ro_ch=ro_ch)
    try:
        values = param.get_actual_values({"sweep": sweep.expts})
    except RuntimeError as exc:
        raise ValueError(f"Time sweep hardware quantization failed: {exc}") from exc
    start, stop = float(values[0]), float(values[-1])
    step = (stop - start) / (sweep.expts - 1) if sweep.expts > 1 else 0.0
    return SweepCfg(start=start, stop=stop, expts=sweep.expts, step=step)


def round_sweep_dict(
    sweep: SweepCfg,
    round_type: Literal["none", "time", "freq", "phase", "gain"] = "none",
    round_info: dict[str, Any] | None = None,
) -> SweepCfg:
    """Return a regular sweep on the requested hardware grid without acquiring.

    ``round_info`` supplies the selected rounder's clock/channel and scaler.
    Time/frequency sweeps use QICK's start/span and signed step quantization.
    Gain sweeps truncate start and derived step to words, then stabilize the
    endpoints so QICK compiles that start and the exact integer-word span.
    Its internal span uses the generator's constant-pulse conversion factor,
    which may differ from the floored ``get_maxv`` used for target words.
    Gain scaler may be negative but must be finite and nonzero. Raise
    RuntimeError if eight endpoint ULP adjustments cannot reproduce the span.
    Raise ValueError when adjacent requested points collapse on the grid.
    """
    if round_info is None:
        round_info = {}
    if round_type == "gain":
        return _round_gain_sweep(sweep, **round_info)
    if round_type == "time":
        return _round_time_sweep(sweep, **round_info)
    if round_type == "freq":
        return _round_frequency_sweep(sweep, **round_info)

    expts = sweep.expts
    span = sweep.stop - sweep.start
    round_start = apply_round(sweep.start, round_type, round_info)
    if expts == 1:
        return SweepCfg(start=round_start, stop=round_start, expts=expts, step=0.0)

    requested_step = span / (expts - 1)
    round_step = apply_round(requested_step, round_type, round_info)
    if math.isclose(float(round_step), 0.0, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(
            _format_zero_step_error(
                sweep, round_type, requested_step, float(round_step)
            )
        )
    round_span = (expts - 1) * round_step
    round_stop = round_start + round_span

    return SweepCfg(start=round_start, stop=round_stop, expts=expts, step=round_step)


def sweep2array[T_SweepItem](
    sweep: SweepCfg | list[T_SweepItem] | np.ndarray,
    round_type: Literal["none", "time", "freq", "phase", "gain"] = "none",
    round_info: dict[str, Any] | None = None,
    *,
    allow_array: bool = False,
) -> NDArray[np.float64]:
    """Materialize sweep coordinates as float64 values without acquiring hardware.

    ``sweep`` may be a regular ``SweepCfg`` or, with ``allow_array=True``, a
    list or ndarray. A regular sweep uses ``round_sweep_dict`` to rebuild its
    grid from ``start``, ``stop`` and ``expts``, not its stored ``step``. Time
    and frequency grids use QICK start/span and signed step quantization.
    Gain grids truncate start and derived step separately to integer words,
    then return each word's nearest QICK-stable scalar coordinate. They do
    not interpolate floats or re-truncate the resulting points. Phase rounds
    the start and the step derived from the span. Even
    ``round_type="none"`` rebuilds the step, without hardware quantization.
    A single-point regular sweep returns only its rounded start.

    Custom lists and arrays first convert to float64, then use ``apply_round``
    to round each independent point. Array shape is preserved, including
    multidimensional input; custom points need not form a regular grid.

    ``round_type`` selects no rounding (``"none"``), time in microseconds
    (``"time"``), frequency in MHz (``"freq"``), phase in degrees (``"phase"``),
    or gain relative to the generator's maximum (``"gain"``).
    ``round_info`` is a keyword-argument dict for the selected rounder; None
    means an empty dict. All hardware modes require ``soccfg``. Frequency,
    phase and gain also require ``gen_ch``. Time accepts optional ``gen_ch``
    and ``ro_ch`` to select a clock; frequency and phase accept optional
    ``ro_ch`` for DAC/ADC grid matching. Hardware modes accept ``scaler``
    (default 1.0) to scale values before quantization and rescale the result.
    Frequency also accepts ``mixer_freq`` in MHz: with it, values are absolute
    RF frequencies and the quantized mixer offset is subtracted; without it,
    values are DDS frequencies. The ``"none"`` mode takes no rounder kwargs.

    Raise ValueError for disallowed custom input, an invalid sweep format or
    round type, or a regular sweep whose derived step is effectively zero
    or whose adjacent points collide after hardware quantization. Conversion
    errors and selected-rounder errors propagate, including missing or invalid
    rounder kwargs. Gain conversion raises RuntimeError when eight ULP
    adjustments cannot reproduce a target word. Custom arrays do not receive
    regular-grid collision checks.
    """
    if round_info is None:
        round_info = {}

    if isinstance(sweep, SweepCfg):
        if round_type == "gain":
            return _host_gain_grid(sweep, **round_info)
        round_sweep = round_sweep_dict(sweep, round_type, round_info)

        return round_sweep.start + np.linspace(
            0, round_sweep.stop - round_sweep.start, round_sweep.expts
        )
    # tolerated: Reject runtime inputs outside the annotated union instead of converting them as custom arrays.
    if isinstance(sweep, (list, np.ndarray)):  # pyright: ignore[reportUnnecessaryIsInstance]
        if not allow_array:
            raise ValueError(f"Custom sweep is not allowed: {sweep}")
        sweep_array = np.asarray(sweep, dtype=np.float64)
        return apply_round(sweep_array, round_type, round_info)
    raise ValueError(f"Invalid sweep format: {sweep}")
