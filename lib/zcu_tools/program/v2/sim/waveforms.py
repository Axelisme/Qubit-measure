"""Peak-normalized waveform envelope sampling shared by lowering and readout.

Two consumers need the *same* envelope shapes but sample them on different time
grids:

  - :mod:`lowering` samples a drive pulse on piecewise-constant sub-segment
    midpoints (to preserve the pulse area / Rabi angle), and
  - :mod:`readout` (the decimated/lookback time-domain model, model A) samples
    the readout pulse on the ADC time grid.

This module owns the *shape functions* (gauss / raised-cosine) as pure
time-array evaluators, plus :func:`envelope_at`, the top-level continuous
envelope (peak 1, zero outside the pulse window) used by the decimated readout
model.  Lowering reuses the same shape functions on its own segment-midpoint
grid, so the numbers it produced before this extraction are unchanged (the
sampling *grid* stayed in lowering; only the *shape evaluation* moved here).

Unit basis: every ``length`` / ``sigma`` / time argument is in microseconds
(µs), matching the QICK-native pulse length unit used throughout lowering.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.waveform import (
    ArbWaveformCfg,
    ConstWaveformCfg,
    CosineWaveformCfg,
    DragWaveformCfg,
    FlatTopWaveformCfg,
    GaussWaveformCfg,
)


def gauss_shape(
    t: NDArray[np.float64], length: float, sigma: float
) -> NDArray[np.float64]:
    """Peak-normalized Gaussian ``exp(-(t-mu)^2 / (2 sigma^2))`` over a pulse.

    Mirrors QICK's ``gauss`` shape with the peak at the pulse center
    ``mu = length / 2``.  Evaluated pointwise at ``t`` (µs); the caller owns the
    grid (segment midpoints for lowering, ADC samples for readout), which is why
    this is a pure shape evaluator and applies no windowing.
    """

    mu = length / 2.0
    return np.exp(-((t - mu) ** 2) / (2.0 * sigma**2))


def cosine_shape(t: NDArray[np.float64], length: float) -> NDArray[np.float64]:
    """Peak-normalized raised-cosine ramp ``(1 - cos(2*pi*t/length)) / 2``.

    Evaluated pointwise at ``t`` (µs) over ``[0, length]``; like
    :func:`gauss_shape` it applies no windowing (the caller owns the grid).
    """

    x = 2.0 * math.pi * t / length
    return (1.0 - np.cos(x)) / 2.0


def _windowed(
    amp: NDArray[np.float64], t: NDArray[np.float64], length: float
) -> NDArray[np.float64]:
    """Zero ``amp`` outside the pulse window ``[0, length)``.

    The continuous envelope is only defined while the pulse plays; before it
    starts and after it ends the generator emits nothing, so the envelope is 0.
    """

    return np.where((t >= 0.0) & (t < length), amp, 0.0)


def arb_waveform_abs_at(
    wav: ArbWaveformCfg, t: NDArray[np.float64], length: float
) -> NDArray[np.float64]:
    """Sample an ArbWaveform on its stored reference time axis.

    The asset's stored time axis is the playback window; it is not stretched or
    compressed. The Bloch/readout simulator uses one scalar envelope, so the I/Q
    asset is represented by ``abs(I + jQ)``.
    """

    from zcu_tools.meta_tool.arb_waveform import ArbWaveformDatabase

    idata_raw, qdata_raw, time_raw = ArbWaveformDatabase.get(wav.data)
    idata = np.asarray(idata_raw, dtype=np.float64)
    qdata = None if qdata_raw is None else np.asarray(qdata_raw, dtype=np.float64)
    time = np.asarray(time_raw, dtype=np.float64)

    ienv = np.interp(t, time, idata, left=0.0, right=0.0)
    if qdata is None:
        amp = np.abs(ienv)
    else:
        qenv = np.interp(t, time, qdata, left=0.0, right=0.0)
        amp = np.hypot(ienv, qenv)

    return _windowed(amp, t, length)


def envelope_at(
    cfg: PulseCfg, t: NDArray[np.float64], length: float
) -> NDArray[np.float64]:
    """Peak-normalized envelope of ``cfg``'s waveform at times ``t`` (µs).

    ``t`` is measured from the pulse start; samples outside ``[0, length)`` are
    0.  ``length`` is the resolved pulse length at the current sweep point (the
    cfg field may be a swept ``QickParam``, so the caller resolves it and passes
    the concrete value here — this keeps sweep resolution in one place).

    Shape per waveform style (matching the lowering / hardware shapes):
      - const     -> 1 inside the window,
      - gauss/drag -> the in-phase Gaussian bell (drag's derivative term has no
        two-level analogue, so it is dropped, consistent with lowering),
      - cosine    -> raised-cosine ramp,
      - arb       -> abs(I+jQ) sampled on the stored reference time axis,
      - flat_top  -> rise ramp + flat top (1) + fall ramp, the ramp shape taken
        from the nested ``raise_waveform``.

    Fast-fail (per CLAUDE.md): an unknown waveform style or a flat_top shorter
    than its ramp raises rather than being silently approximated.
    """

    t = np.asarray(t, dtype=np.float64)
    wav = cfg.waveform

    if isinstance(wav, ConstWaveformCfg):
        return _windowed(np.ones_like(t), t, length)

    if isinstance(wav, (GaussWaveformCfg, DragWaveformCfg)):
        sigma = (wav.sigma / wav.length) * length
        return _windowed(gauss_shape(t, length, sigma), t, length)

    if isinstance(wav, CosineWaveformCfg):
        return _windowed(cosine_shape(t, length), t, length)

    if isinstance(wav, ArbWaveformCfg):
        return arb_waveform_abs_at(wav, t, length)

    if isinstance(wav, FlatTopWaveformCfg):
        return _flat_top_envelope_at(wav, t, length)

    raise ValueError(
        f"unsupported waveform style {type(wav).__name__} for envelope sampling"
    )


def _flat_top_envelope_at(
    wav: FlatTopWaveformCfg, t: NDArray[np.float64], length: float
) -> NDArray[np.float64]:
    """Continuous flat_top envelope: rising ramp, flat top (1), falling ramp.

    The ramp (length ``ramp_len`` from the nested ``raise_waveform``) is split
    into a rising half over ``[0, ramp_len/2)`` and a falling half over
    ``[length - ramp_len/2, length)``, with a flat top of value 1 in between —
    the same rise/flat/fall decomposition lowering uses, evaluated continuously
    on the ADC grid here.  ``raise_waveform`` is a gauss, cosine, or arb ramp; a
    const ramp degrades to a flat window (matching lowering's const-ramp
    fallback).
    """

    ramp_len = float(wav.raise_waveform.length)
    flat_len = length - ramp_len
    if flat_len < 0.0:
        raise ValueError(
            f"flat_top length {length} µs is shorter than its ramp "
            f"{ramp_len} µs; cannot sample envelope"
        )

    half = ramp_len / 2.0
    ramp_cfg = wav.raise_waveform

    # The ramp is the rising then falling half of a full-width-ramp_len shape.
    # Map a falling-half time back onto the rising half by mirroring about the
    # pulse center, so a single shape evaluation drives both ramps.
    rise_region = (t >= 0.0) & (t < half)
    fall_region = (t >= length - half) & (t < length)
    mirrored = np.where(fall_region, length - t, t)

    if isinstance(ramp_cfg, GaussWaveformCfg):
        sigma = (ramp_cfg.sigma / ramp_cfg.length) * ramp_len
        ramp_amp = gauss_shape(mirrored, ramp_len, sigma)
    elif isinstance(ramp_cfg, CosineWaveformCfg):
        ramp_amp = cosine_shape(mirrored, ramp_len)
    elif isinstance(ramp_cfg, ArbWaveformCfg):
        ramp_amp = arb_waveform_abs_at(ramp_cfg, mirrored, ramp_len)
    else:
        # Const ramp: lowering approximates it as a single full-amplitude segment,
        # i.e. no ramp shaping — mirror that by a flat window here.
        return _windowed(np.ones_like(t), t, length)

    amp = np.where(rise_region | fall_region, ramp_amp, 1.0)
    return _windowed(amp, t, length)
