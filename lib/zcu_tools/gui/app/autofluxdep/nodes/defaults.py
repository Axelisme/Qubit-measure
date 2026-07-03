"""Context-aware default seeds for autofluxdep node cfg schemas.

The helpers mirror measure-gui's md/ml seeding pattern without importing the
measure app directly: fresh autofluxdep placements can adopt calibrated modules
and live md expressions, while persisted schemas remain the stored value tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any

from zcu_tools.gui.app.autofluxdep.cfg import EvalValue, SweepValue
from zcu_tools.program.v2.modules import DirectReadoutCfg, PulseCfg, PulseReadoutCfg

if TYPE_CHECKING:
    from zcu_tools.gui.session.types import ExpContext
    from zcu_tools.meta_tool import ModuleLibrary


@dataclass(frozen=True)
class PulseSeed:
    waveform: str | None
    ch: Any
    nqz: int
    freq: Any
    gain: float
    length: float
    from_module: bool


@dataclass(frozen=True)
class ReadoutSeed:
    freq: float
    gain: float
    from_module: bool


def md_has(ctx: ExpContext | Any | None, key: str) -> bool:
    md = getattr(ctx, "md", None)
    if md is None:
        return False
    sentinel = object()
    try:
        value = md.get(key, sentinel)
    except Exception:
        return False
    return value is not sentinel and value is not None


def md_scalar(ctx: ExpContext | Any | None, key: str, fallback: Any) -> Any:
    """Return a live md expression when the key exists, else a plain fallback."""
    if md_has(ctx, key):
        return EvalValue(key)
    return fallback


def md_scalar_first(
    ctx: ExpContext | Any | None, keys: tuple[str, ...], fallback: Any
) -> Any:
    for key in keys:
        if md_has(ctx, key):
            return EvalValue(key)
    return fallback


def md_float(ctx: ExpContext | Any | None, key: str, fallback: float) -> float:
    value = _md_number(ctx, key)
    return value if value is not None else fallback


def md_scaled(
    ctx: ExpContext | Any | None,
    key: str,
    factor: float,
    fallback: float,
    *,
    minimum: float | None = None,
) -> float | EvalValue:
    """Default for ``factor * md.<key>``.

    Numeric md values are materialized immediately so floor constraints can be
    honoured. Non-numeric-but-present md keys keep a live expression; the shared
    evaluator will fast-fail on lower if the md value is not numeric.
    """
    value = _md_number(ctx, key)
    if value is not None:
        scaled = factor * value
        return max(minimum, scaled) if minimum is not None else scaled
    if md_has(ctx, key):
        return EvalValue(f"{factor} * {key}")
    scaled = factor * fallback
    return max(minimum, scaled) if minimum is not None else scaled


def md_scaled_sweep_stop(
    ctx: ExpContext | Any | None,
    key: str,
    factor: float,
    fallback: float,
    *,
    minimum: float | None = None,
) -> float | EvalValue:
    return md_scaled(ctx, key, factor, fallback, minimum=minimum)


def preferred_waveform(
    ctx: ExpContext | Any | None, names: tuple[str, ...], fallback: str | None
) -> str | None:
    ml = _context_ml(ctx)
    if ml is not None:
        waveforms = getattr(ml, "waveforms", {})
        for name in names:
            if name in waveforms:
                return name
    return fallback


def pulse_seed(
    ctx: ExpContext | Any | None,
    *,
    module_aliases: tuple[str, ...],
    waveform_aliases: tuple[str, ...],
    fallback_waveform: str | None,
    fallback_ch: Any,
    fallback_nqz: int,
    fallback_freq: Any,
    fallback_gain: float,
    fallback_length: float,
) -> PulseSeed:
    waveform = preferred_waveform(ctx, waveform_aliases, fallback_waveform)
    module = _first_module(ctx, module_aliases, PulseCfg)
    if module is None:
        return PulseSeed(
            waveform=waveform,
            ch=fallback_ch,
            nqz=_valid_nqz(fallback_nqz, 2),
            freq=fallback_freq,
            gain=fallback_gain,
            length=fallback_length,
            from_module=False,
        )
    length = _number(getattr(module.waveform, "length", None))
    freq = _number(module.freq)
    gain = _number(module.gain)
    return PulseSeed(
        waveform=waveform,
        ch=int(module.ch),
        nqz=_valid_nqz(module.nqz, fallback_nqz),
        freq=freq if freq is not None else fallback_freq,
        gain=gain if gain is not None else fallback_gain,
        length=length if length is not None else fallback_length,
        from_module=True,
    )


def readout_seed(
    ctx: ExpContext | Any | None,
    aliases: tuple[str, ...],
    *,
    fallback_freq: float,
    fallback_gain: float,
) -> ReadoutSeed:
    module = _first_readout(ctx, aliases)
    if isinstance(module, PulseReadoutCfg):
        freq = _number(module.pulse_cfg.freq)
        if freq is None:
            freq = _number(module.ro_cfg.ro_freq)
        gain = _number(module.pulse_cfg.gain)
        return ReadoutSeed(
            freq=freq if freq is not None else fallback_freq,
            gain=gain if gain is not None else fallback_gain,
            from_module=True,
        )
    if isinstance(module, DirectReadoutCfg):
        freq = _number(module.ro_freq)
        return ReadoutSeed(
            freq=freq if freq is not None else fallback_freq,
            gain=fallback_gain,
            from_module=True,
        )
    return ReadoutSeed(freq=fallback_freq, gain=fallback_gain, from_module=False)


def ro_freq_range(
    ctx: ExpContext | Any | None,
    *,
    center: float,
    fallback_half_width: float,
    expts: int,
) -> SweepValue:
    if md_has(ctx, "r_f") and md_has(ctx, "rf_w"):
        return SweepValue(
            start=EvalValue("r_f - 0.2 * rf_w"),
            stop=EvalValue("r_f + 0.2 * rf_w"),
            expts=expts,
        )
    width = md_float(ctx, "rf_w", fallback_half_width / 0.2)
    half_width = 0.2 * width if width > 0.0 else fallback_half_width
    return SweepValue(start=center - half_width, stop=center + half_width, expts=expts)


def ro_gain_range(
    ctx: ExpContext | Any | None,
    *,
    center: float,
    half_width: float,
    expts: int,
) -> SweepValue:
    value = _md_number(ctx, "best_ro_gain")
    if value is not None:
        return _clipped_gain_range(value, half_width=half_width, expts=expts)
    if md_has(ctx, "best_ro_gain"):
        return SweepValue(
            start=EvalValue(f"best_ro_gain - {half_width}"),
            stop=EvalValue(f"best_ro_gain + {half_width}"),
            expts=expts,
        )
    return _clipped_gain_range(center, half_width=half_width, expts=expts)


def _clipped_gain_range(center: float, *, half_width: float, expts: int) -> SweepValue:
    return SweepValue(
        start=round(max(0.0, center - half_width), 12),
        stop=round(min(1.0, center + half_width), 12),
        expts=expts,
    )


def _context_ml(ctx: ExpContext | Any | None) -> ModuleLibrary | None:
    return getattr(ctx, "ml", None)


def _first_module[T](
    ctx: ExpContext | Any | None, aliases: tuple[str, ...], module_type: type[T]
) -> T | None:
    ml = _context_ml(ctx)
    if ml is None:
        return None
    modules = getattr(ml, "modules", {})
    for name in aliases:
        module = modules.get(name)
        if isinstance(module, module_type):
            return module
    return None


def _first_readout(
    ctx: ExpContext | Any | None, aliases: tuple[str, ...]
) -> PulseReadoutCfg | DirectReadoutCfg | None:
    ml = _context_ml(ctx)
    if ml is None:
        return None
    modules = getattr(ml, "modules", {})
    for name in aliases:
        module = modules.get(name)
        if isinstance(module, (PulseReadoutCfg, DirectReadoutCfg)):
            return module
    return None


def _md_number(ctx: ExpContext | Any | None, key: str) -> float | None:
    md = getattr(ctx, "md", None)
    if md is None:
        return None
    try:
        value = md.get(key)
    except Exception:
        return None
    return _number(value)


def _number(value: object, fallback: float | None = None) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return fallback
    return float(value)


def _valid_nqz(value: object, fallback: int) -> int:
    if value in (1, 2):
        return int(value)
    return fallback if fallback in (1, 2) else 2
