"""Module tree -> Bloch timeline lowering for the SimEngine.

This is the bridge between the *semantic* v2 program (a list of ``Module``
instances plus the sweep axes) and the *physics* core (:mod:`bloch`).  For one
sweep point it walks the module list in execution order and emits:

  - a list of :class:`bloch.Segment` describing the qubit's piecewise-constant
    evolution (reset pulses, drive pulses, idle / free-evolution delays) that
    happen *before* readout, and
  - a :class:`ReadoutPlan` describing the resolved readout settings at this
    point: probe frequency, ADC integration window, drive gain, and optional
    generator pulse-envelope metadata.

Responsibility boundary (kept deliberately narrow):
  - Lowering resolves sweep axes (via ``QickParam.to_array``) and the dmem
    register indirection used by the non-uniform T1 path, and maps each module
    to segments under a single, consistent time/frequency basis.
  - Lowering does NOT compute ``f_qubit`` (the engine derives it from
    ``SimParams`` + flux via fluxonium physics and passes it in), does NOT touch
    ``acc_buf`` / noise, and does NOT compute S21 / resonator response.  Those
    belong to the engine / readout layers.

Unit basis (the single most important correctness invariant)
------------------------------------------------------------
Everything is expressed in microseconds (µs) for time and rad/µs for angular
frequency, so that ``omega`` / ``delta`` / ``t`` / ``t1`` / ``t2`` handed to
:class:`bloch.Segment` are all consistent:

  - cfg pulse ``length`` / ``pre_delay`` / ``post_delay`` / delay  -> µs (QICK
    native unit), used verbatim as the segment duration ``t``.
  - cfg ``freq`` / ``ro_freq`` -> MHz (QICK native unit).  Detuning is
    ``delta = 2*pi*(f_qubit_MHz - f_drive_MHz)`` in rad/µs, following the
    :mod:`bloch` convention ``delta = omega_qubit - omega_drive`` (qubit minus
    drive).  ``f_qubit`` arrives in GHz, so it is scaled by ``1e3`` to MHz.
  - cfg ``phase`` -> degrees (QICK native unit), converted to radians.
  - ``SimParams.T1`` / ``T2`` are already in µs and are threaded onto every
    segment unchanged.

Single rotating frame (Ramsey / echo correctness)
-------------------------------------------------
The whole timeline lives in *one* rotating frame whose carrier is the qubit
control pulses' frequency ``f_ref`` (the top-level :class:`Pulse` modules — reset
prep pulses and the readout do not define the qubit frame).  In that frame every
segment, drive *and* idle, precesses about z at the same frame detuning
``delta = 2*pi*(f_qubit_MHz - f_ref_MHz)``:

  - drive segments already carry ``2*pi*(f_qubit - pulse.freq)``; since a qubit
    pulse's ``pulse.freq`` *is* ``f_ref``, that equals the frame detuning.
  - idle / free-evolution segments (delays, pre/post idle) must carry the *same*
    frame detuning, NOT 0.  A zero idle detuning would freeze the Bloch vector
    between pulses and kill Ramsey fringes.

When the qubit is driven on resonance (``f_ref == f_qubit``) the frame detuning
is 0 and idle segments are static.  A Ramsey experiment that detunes the control pulses
(``f_ref = f_qubit + detuning``) gets idle precession at the detuning, producing
fringes at the detuning frequency.

Drive amplitude (the gain -> Rabi-rate chain)
---------------------------------------------
``Omega(t) = (pi / pi_gain_len) * gain * envelope(t)`` where ``envelope`` peaks
at 1.  For a const pulse ``envelope == 1`` over its whole length, so the rotation
angle is ``theta = Omega * length = pi * gain * length / pi_gain_len`` — i.e.
``gain * length == pi_gain_len`` gives an exact pi rotation.  Shaped pulses
(gauss / drag / flat_top) are discretized into piecewise-constant sub-segments
that sample the envelope, preserving the pulse area (and therefore the chevron /
Rabi fidelity).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from qick.asm_v2 import QickParam

from zcu_tools.program.v2.modules.base import Module
from zcu_tools.program.v2.modules.control import Branch
from zcu_tools.program.v2.modules.delay import Delay, DelayAuto, SoftDelay
from zcu_tools.program.v2.modules.dmem import LoadValue
from zcu_tools.program.v2.modules.pulse import Pulse, PulseCfg
from zcu_tools.program.v2.modules.readout import (
    AbsReadout,
    DirectReadout,
    PulseReadout,
)
from zcu_tools.program.v2.modules.reset import (
    BathReset,
    NoneReset,
    PulseReset,
    TwoPulseReset,
)
from zcu_tools.program.v2.modules.waveform import (
    ArbWaveformCfg,
    ConstWaveformCfg,
    CosineWaveformCfg,
    DragWaveformCfg,
    FlatTopWaveformCfg,
    GaussWaveformCfg,
)
from zcu_tools.program.v2.sweep import SweepCfg
from zcu_tools.program.v2.utils import is_qick_param

from .bloch import Segment
from .params import SimParams
from .waveforms import arb_waveform_abs_at, cosine_shape, gauss_shape

# Number of piecewise-constant sub-segments used to sample a smoothly shaped
# pulse envelope (gauss / drag / cosine ramp).  ~32 keeps the area / chevron
# error well below the per-segment matrix-exponential error while staying cheap.
_SHAPED_PULSE_SEGMENTS = 32

# GHz -> MHz scale for f_qubit (the engine works in GHz; the timeline in MHz).
_GHZ_TO_MHZ = 1e3


@dataclass(frozen=True)
class ReadoutPlan:
    """Readout description produced by lowering for one experiment.

    Fields
    ------
    f_ro_ghz : float
        Readout probe frequency in GHz at *this* sweep point (the cfg
        ``ro_freq`` in MHz, resolved per point and scaled down).  A swept
        ``ro_freq`` is resolved here to its value at the current sweep index, so
        the engine reads it back point-by-point without needing to know whether
        it is swept or fixed — every experiment goes through one unified path.
    ro_length_us : float
        ADC integration window in µs from the semantic readout cfg.  The engine
        still uses the compiled sample count as the raw-buffer normalization
        source of truth, but the semantic plan carries this value explicitly.
    trig_offset_us : float
        Resolved ADC trigger offset in µs.  Accumulated ``PulseReadout`` uses it
        to align the integration samples against the generator pulse envelope.
    readout_gain : float
        Resolved generator gain for a ``PulseReadout``.  ``DirectReadout`` has no
        drive-gain cfg, so it uses unity.
    pulse_cfg : PulseCfg | None
        The generator pulse cfg whose envelope defines the effective signal area,
        or ``None`` for ``DirectReadout``.
    pulse_length_us : float | None
        Resolved generator pulse/envelope length in µs, or ``None`` for
        ``DirectReadout``.
    pulse_pre_delay_us : float
        Resolved generator pulse pre-delay in µs.  ``PulseReadout`` triggers its
        ADC window and generator pulse from the same module time, but the
        generator pulse itself is played at ``t + pre_delay``.
    """

    f_ro_ghz: float
    ro_length_us: float
    trig_offset_us: float = 0.0
    readout_gain: float = 1.0
    pulse_cfg: PulseCfg | None = None
    pulse_length_us: float | None = None
    pulse_pre_delay_us: float = 0.0


@dataclass(frozen=True)
class LoweredPoint:
    """The full per-sweep-point lowering result consumed by the engine."""

    segments: list[Segment]
    readout: ReadoutPlan


class UnsupportedModuleError(ValueError):
    """Raised when a module cannot be faithfully lowered to a Bloch timeline.

    Fast-fail (per CLAUDE.md): register-driven constructs whose values cannot be
    recovered statically, and any unknown module type raise rather than being
    silently approximated.  Deterministic ``Branch`` (selected by a registered
    sweep-loop counter) *is* lowered; only measurement-conditional branches,
    nested branches, and readout inside a branch fast-fail.
    """


def _loop_counts(sweep: Sequence[tuple[str, SweepCfg | int]]) -> dict[str, int]:
    """Build the ``{axis_name: count}`` map (outermost first) for ``to_array``."""

    return {
        name: (spec.expts if isinstance(spec, SweepCfg) else int(spec))
        for name, spec in sweep
    }


def _resolve_scalar(
    value: float | QickParam,
    loop_counts: dict[str, int],
    point: dict[str, int],
) -> float:
    """Resolve a possibly-swept field to its concrete value at one sweep point.

    A plain float is returned as-is.  A ``QickParam`` is evaluated with
    ``to_array(..., all_loops=True)`` (so every loop appears as an axis, size 1
    for loops the param does not span) and indexed at ``point``; axes the param
    does not span use index 0.  This matches how QICK assigns sweep values and
    works for scaled params (e.g. ``0.5 * length_param`` in t2echo).
    """

    if not isinstance(value, QickParam):
        return float(value)

    if not is_qick_param(value):
        raise TypeError(f"unsupported QickParam implementation: {type(value).__name__}")
    array = value.to_array(loop_counts, all_loops=True)
    index = tuple(point[name] if name in value.spans else 0 for name in loop_counts)
    return float(np.asarray(array)[index])


def _segment_midpoints(length: float, n: int) -> np.ndarray:
    """The ``n`` piecewise-constant sub-segment midpoints over ``[0, length]``.

    Midpoint sampling keeps a discretized shape's area close to the continuous
    integral (preserving the Rabi / chevron angle).  This is the sampling *grid*
    lowering owns; the shape *values* come from :mod:`waveforms`.
    """

    edges = np.linspace(0.0, length, n + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def _gauss_amplitudes(length: float, sigma: float, n: int) -> np.ndarray:
    """Peak-normalized Gaussian sampled on the ``n`` segment midpoints.

    Reuses the shared :func:`waveforms.gauss_shape` so the readout time-domain
    model and lowering evaluate the identical bell; only the sampling grid (here,
    midpoints) is lowering-specific.
    """

    return gauss_shape(_segment_midpoints(length, n), length, sigma)


def _cosine_amplitudes(length: float, n: int) -> np.ndarray:
    """Peak-normalized raised-cosine ramp sampled on the ``n`` segment midpoints."""

    return cosine_shape(_segment_midpoints(length, n), length)


def _arb_amplitudes(wav: ArbWaveformCfg, length: float, n: int) -> np.ndarray:
    """Arbitrary I/Q waveform magnitude sampled on the ``n`` segment midpoints."""

    return arb_waveform_abs_at(wav, _segment_midpoints(length, n), length)


def _drive_amp_segments(
    cfg: PulseCfg, length: float, n: int
) -> tuple[list[float], list[float]]:
    """Return ``(durations, amplitudes)`` for the pulse envelope (peak = 1).

    Const and flat_top flat tops collapse to single full-amplitude segments;
    gauss / drag / cosine / arb shapes (and flat_top ramps) are discretized into
    ``n`` midpoint samples each.  Lengths are in µs.
    """

    wav = cfg.waveform

    if isinstance(wav, ConstWaveformCfg):
        return [length], [1.0]

    if isinstance(wav, GaussWaveformCfg):
        ratio = wav.sigma / wav.length
        amps = _gauss_amplitudes(length, ratio * length, n)
        durs = [length / n] * n
        return durs, list(amps)

    if isinstance(wav, DragWaveformCfg):
        # Only the in-phase Gaussian envelope drives the (two-level) Rabi rate;
        # the DRAG derivative term is a leakage correction with no TLS analogue.
        ratio = wav.sigma / wav.length
        amps = _gauss_amplitudes(length, ratio * length, n)
        durs = [length / n] * n
        return durs, list(amps)

    if isinstance(wav, CosineWaveformCfg):
        amps = _cosine_amplitudes(length, n)
        durs = [length / n] * n
        return durs, list(amps)

    if isinstance(wav, ArbWaveformCfg):
        amps = _arb_amplitudes(wav, length, n)
        durs = [length / n] * n
        return durs, [float(amp) for amp in amps]

    if isinstance(wav, FlatTopWaveformCfg):
        # flat_top = rise ramp + flat top + fall ramp.  The ramp shape is the
        # nested raise_waveform; QICK plays it rising then falling around a flat
        # region of length (total - ramp).  Discretize each ramp half and keep
        # the flat top as one full-amplitude segment.
        ramp_len = float(wav.raise_waveform.length)
        flat_len = length - ramp_len
        if flat_len < 0.0:
            raise UnsupportedModuleError(
                f"flat_top length {length} µs is shorter than its ramp "
                f"{ramp_len} µs; cannot lower"
            )
        ramp_cfg = wav.raise_waveform
        half = ramp_len / 2.0
        if isinstance(ramp_cfg, GaussWaveformCfg):
            # A flat_top gauss ramp uses the rising then falling half of a
            # Gaussian whose full width equals ramp_len.
            ratio = ramp_cfg.sigma / ramp_cfg.length
            sigma = ratio * ramp_len
            rise = _gauss_amplitudes(ramp_len, sigma, n)[: n // 2 or 1]
            durs = [half / len(rise)] * len(rise)
            durs += [flat_len]
            durs += list(reversed(durs[: len(rise)]))
            amps = list(rise) + [1.0] + list(reversed(rise))
            return durs, amps
        if isinstance(ramp_cfg, CosineWaveformCfg):
            rise = _cosine_amplitudes(ramp_len, n)[: n // 2 or 1]
            durs = [half / len(rise)] * len(rise)
            durs += [flat_len]
            durs += list(reversed(durs[: len(rise)]))
            amps = list(rise) + [1.0] + list(reversed(rise))
            return durs, amps
        if isinstance(ramp_cfg, ArbWaveformCfg):
            rise = _arb_amplitudes(ramp_cfg, ramp_len, n)[: n // 2 or 1]
            durs = [half / len(rise)] * len(rise)
            durs += [flat_len]
            durs += list(reversed(durs[: len(rise)]))
            amps = (
                [float(amp) for amp in rise]
                + [1.0]
                + [float(amp) for amp in reversed(rise)]
            )
            return durs, amps
        # Const ramp: approximate the ramp as full amplitude (its area is small
        # relative to the flat top); fall back to a single flat segment of the
        # full length.
        return [length], [1.0]

    raise UnsupportedModuleError(
        f"unsupported waveform style {type(wav).__name__} for Bloch lowering"
    )


def _pulse_segments(
    cfg: PulseCfg,
    sim: SimParams,
    f_qubit_mhz: float,
    frame_detuning: float,
    detune_offset: float,
    loop_counts: dict[str, int],
    point: dict[str, int],
) -> list[Segment]:
    """Lower one Pulse cfg to drive segments (with pre/post idle) at one point.

    ``frame_detuning`` is the single-frame idle precession rate (rad/µs) shared by
    this pulse's pre/post idle and every other free segment in the timeline; see
    the module docstring.  The drive segments themselves use this pulse's own
    detuning, which equals ``frame_detuning`` whenever the pulse sits at the frame
    carrier (the normal case for qubit gates).

    ``detune_offset`` (rad/µs) is a static, global frame shift added to every
    segment's delta (drive and the pre/post idle); see ``lower_point``.  The
    engine uses it to sample the Lorentzian quasi-static detune ensemble
    without lowering having to know about the
    ensemble — lowering only shifts the frame.
    """

    pre_delay = _resolve_scalar(cfg.pre_delay, loop_counts, point)
    post_delay = _resolve_scalar(cfg.post_delay, loop_counts, point)
    length = _resolve_scalar(cfg.waveform.length, loop_counts, point)
    gain = _resolve_scalar(cfg.gain, loop_counts, point)
    freq_mhz = _resolve_scalar(cfg.freq, loop_counts, point)
    phase_deg = _resolve_scalar(cfg.phase, loop_counts, point)

    # bloch convention: delta = omega_qubit - omega_drive (qubit minus drive).
    # detune_offset is the global static frame shift applied to every segment.
    delta = 2.0 * math.pi * (f_qubit_mhz - freq_mhz) + detune_offset
    phase = math.radians(phase_deg)
    omega_scale = (math.pi / sim.pi_gain_len) * gain

    durs, amps = _drive_amp_segments(cfg, length, _SHAPED_PULSE_SEGMENTS)

    idle_detuning = frame_detuning + detune_offset

    segments: list[Segment] = []
    if pre_delay > 0.0:
        segments.append(_idle_segment(sim, pre_delay, idle_detuning))
    for dur, amp in zip(durs, amps):
        segments.append(
            Segment(
                omega=omega_scale * amp,
                delta=delta,
                phase=phase,
                t=dur,
                t1=sim.T1,
                t2=sim.T2,
                thermal_pop=sim.thermal_pop,
            )
        )
    if post_delay > 0.0:
        segments.append(_idle_segment(sim, post_delay, idle_detuning))
    return segments


def _idle_segment(sim: SimParams, t: float, frame_detuning: float) -> Segment:
    """A free-evolution (Omega = 0) segment of duration ``t`` µs.

    ``frame_detuning`` (rad/µs) is the single-frame precession rate
    ``2*pi*(f_qubit - f_ref)``: the Bloch vector keeps precessing about z during
    the idle, which is what makes Ramsey fringes appear.  It is 0 when the qubit
    is driven on resonance (then the idle is static, as in T1).
    """

    return Segment(
        omega=0.0,
        delta=frame_detuning,
        phase=0.0,
        t=t,
        t1=sim.T1,
        t2=sim.T2,
        thermal_pop=sim.thermal_pop,
    )


def _reset_segments(
    module: Module,
    sim: SimParams,
    f_qubit_mhz: float,
    frame_detuning: float,
    detune_offset: float,
    loop_counts: dict[str, int],
    point: dict[str, int],
) -> list[Segment]:
    """Lower a reset module to its unconditional pulse sequence.

    Passive reset (NoneReset) emits nothing — the relax_delay between shots is a
    free-evolution effect the engine handles via the inter-shot reset to thermal
    equilibrium, so an explicit idle here would double-count it.  The active
    schemes (PulseReset / TwoPulseReset / BathReset) are unconditional pulse
    sequences and lower to their constituent drive pulses in run order.
    ``frame_detuning`` is threaded through only for these pulses' pre/post idle
    segments (the drive segments use each reset pulse's own freq); ``detune_offset``
    is the global static frame shift forwarded verbatim.
    """

    if isinstance(module, NoneReset):
        return []

    if isinstance(module, PulseReset):
        return _pulse_segments(
            module.cfg.pulse_cfg,
            sim,
            f_qubit_mhz,
            frame_detuning,
            detune_offset,
            loop_counts,
            point,
        )

    if isinstance(module, TwoPulseReset):
        # Hardware plays both pulses starting at the same t (overlapping); a
        # single-qubit Bloch model cannot represent two simultaneous drives, so
        # sequence them.  Reset pulses are unconditional preparation, not the
        # measured dynamics, so the ordering does not affect the gate under test.
        return _pulse_segments(
            module.cfg.pulse1_cfg,
            sim,
            f_qubit_mhz,
            frame_detuning,
            detune_offset,
            loop_counts,
            point,
        ) + _pulse_segments(
            module.cfg.pulse2_cfg,
            sim,
            f_qubit_mhz,
            frame_detuning,
            detune_offset,
            loop_counts,
            point,
        )

    if isinstance(module, BathReset):
        return (
            _pulse_segments(
                module.cfg.cavity_tone_cfg,
                sim,
                f_qubit_mhz,
                frame_detuning,
                detune_offset,
                loop_counts,
                point,
            )
            + _pulse_segments(
                module.cfg.qubit_tone_cfg,
                sim,
                f_qubit_mhz,
                frame_detuning,
                detune_offset,
                loop_counts,
                point,
            )
            + _pulse_segments(
                module.cfg.pi2_cfg,
                sim,
                f_qubit_mhz,
                frame_detuning,
                detune_offset,
                loop_counts,
                point,
            )
        )

    raise UnsupportedModuleError(
        f"unsupported reset module {type(module).__name__} for Bloch lowering"
    )


def _delay_segment(
    module: Delay | DelayAuto | SoftDelay,
    sim: SimParams,
    idle_detuning: float,
    loop_counts: dict[str, int],
    point: dict[str, int],
    dmem_values: dict[str, list[int]],
    cycles2us: object,
) -> Segment:
    """Lower a delay module to one free-evolution segment.

    A ``DelayAuto`` whose ``t`` is a register name is the non-uniform T1 path:
    the per-point delay lives in a dmem table loaded by a ``LoadValue`` earlier
    in the module list, indexed by the ``*_idx`` sweep axis.  The cycle count is
    recovered from ``dmem_values`` and converted to µs via ``cycles2us``.
    ``idle_detuning`` is the single-frame idle precession rate (already including
    the global ``detune_offset``) applied to the emitted free segment.
    """

    if isinstance(module, DelayAuto) and isinstance(module.t, str):
        return _dmem_delay_segment(
            module, sim, idle_detuning, point, dmem_values, cycles2us
        )

    # DelayAuto stores its (non-register) duration in `.t`; Delay and SoftDelay
    # both store it in `.delay`.  DelayAuto.t may be a register name (str), but
    # that case is handled above, so here it is always float | QickParam.
    raw_t: float | QickParam
    if isinstance(module, DelayAuto):
        assert not isinstance(module.t, str)  # str path handled above
        raw_t = module.t
    else:
        raw_t = module.delay
    t = _resolve_scalar(raw_t, loop_counts, point)
    return _idle_segment(sim, t, idle_detuning)


def _dmem_delay_segment(
    module: DelayAuto,
    sim: SimParams,
    idle_detuning: float,
    point: dict[str, int],
    dmem_values: dict[str, list[int]],
    cycles2us: object,
) -> Segment:
    """Recover a register-driven DelayAuto's per-point delay from the dmem table.

    The non-uniform T1 program is::

        LoadValue("...", values=cycles, idx_reg="length_idx", val_reg="t1_delay_cycle")
        DelayAuto("t1_delay", t="t1_delay_cycle")
        sweep=[("length_idx", N)]

    so ``val_reg`` names the dmem table (recorded under that key by
    :func:`_collect_dmem_values`) and the matching ``*_idx`` sweep axis selects
    the entry.  ``cycles2us`` converts the cycle count to a µs duration.
    """

    val_reg = module.t
    assert isinstance(val_reg, str)
    if val_reg not in dmem_values:
        raise UnsupportedModuleError(
            f"DelayAuto reads register {val_reg!r} but no LoadValue populates it; "
            f"cannot recover the delay value"
        )

    values = dmem_values[val_reg]

    # The sweep axis that indexes this table is the LoadValue's idx_reg.  We
    # recorded it alongside the values; find the matching axis index in `point`.
    idx_axis = _DMEM_IDX_AXES.get(val_reg)
    if idx_axis is None or idx_axis not in point:
        raise UnsupportedModuleError(
            f"cannot determine which sweep axis indexes dmem register {val_reg!r}"
        )

    idx = point[idx_axis]
    if not 0 <= idx < len(values):
        raise UnsupportedModuleError(
            f"dmem index {idx} out of range for register {val_reg!r} "
            f"(table size {len(values)})"
        )

    cycles = values[idx]
    t_us = float(cycles2us(int(cycles)))  # type: ignore[operator]
    return _idle_segment(sim, t_us, idle_detuning)


# Module-list-local mapping from a LoadValue val_reg to the sweep axis (idx_reg)
# that indexes it.  Populated per-lowering by _collect_dmem_values; module-global
# because lowering is single-threaded and re-populates it on every call.
_DMEM_IDX_AXES: dict[str, str] = {}


def _collect_dmem_values(
    modules: Sequence[Module],
) -> dict[str, list[int]]:
    """Scan the module list for LoadValue tables, keyed by their ``val_reg``.

    Also records the LoadValue's ``idx_reg`` (the sweep axis name) so a later
    register-driven DelayAuto can map its register back to the correct sweep
    axis.  ``auto_compress`` tables would change the stored layout; the T1 path
    uses ``auto_compress=False`` so values are stored verbatim, which is the only
    layout this lowering supports — anything compressed fast-fails.
    """

    _DMEM_IDX_AXES.clear()
    tables: dict[str, list[int]] = {}
    for module in modules:
        if isinstance(module, LoadValue):
            if module._is_compressed:
                raise UnsupportedModuleError(
                    f"LoadValue {module.name!r} is compressed; the Bloch lowering "
                    f"only supports verbatim (auto_compress=False) dmem tables"
                )
            tables[module.val_reg] = list(module.values)
            _DMEM_IDX_AXES[module.val_reg] = module.idx_reg
    return tables


def _readout_plan(
    module: AbsReadout,
    loop_counts: dict[str, int],
    point: dict[str, int],
) -> ReadoutPlan:
    """Build the ReadoutPlan from a readout module at one sweep point."""

    if isinstance(module, DirectReadout):
        ro_freq = module.cfg.ro_freq
        ro_length = module.cfg.ro_length
        trig_offset = module.cfg.trig_offset
        f_ro_mhz = _resolve_scalar(ro_freq, loop_counts, point)
        ro_length_us = _resolve_scalar(ro_length, loop_counts, point)
        trig_offset_us = _resolve_scalar(trig_offset, loop_counts, point)
        return ReadoutPlan(
            f_ro_ghz=f_ro_mhz / _GHZ_TO_MHZ,
            ro_length_us=ro_length_us,
            trig_offset_us=trig_offset_us,
        )
    elif isinstance(module, PulseReadout):
        ro_freq = module.cfg.ro_cfg.ro_freq
        ro_length = module.cfg.ro_cfg.ro_length
        trig_offset = module.cfg.ro_cfg.trig_offset
        readout_gain = _resolve_scalar(module.cfg.pulse_cfg.gain, loop_counts, point)
        pulse_length_us = _resolve_scalar(
            module.cfg.pulse_cfg.waveform.length, loop_counts, point
        )
        pulse_pre_delay_us = _resolve_scalar(
            module.cfg.pulse_cfg.pre_delay, loop_counts, point
        )
        f_ro_mhz = _resolve_scalar(ro_freq, loop_counts, point)
        ro_length_us = _resolve_scalar(ro_length, loop_counts, point)
        trig_offset_us = _resolve_scalar(trig_offset, loop_counts, point)
        return ReadoutPlan(
            f_ro_ghz=f_ro_mhz / _GHZ_TO_MHZ,
            ro_length_us=ro_length_us,
            trig_offset_us=trig_offset_us,
            readout_gain=readout_gain,
            pulse_cfg=module.cfg.pulse_cfg,
            pulse_length_us=pulse_length_us,
            pulse_pre_delay_us=pulse_pre_delay_us,
        )
    else:
        raise UnsupportedModuleError(
            f"unsupported readout module {type(module).__name__} for lowering"
        )


def _select_branch(branch: Branch, point: dict[str, int]) -> list[Module]:
    """Pick the active sub-sequence of a deterministic ``Branch`` at one point.

    ``Branch`` (control.py) does not create its own loop: it selects branch *i*
    from the value of ``compare_reg``, which is the counter of an *external* sweep
    loop registered via ``add_loop``.  Lowering already holds that counter as the
    ``point[compare_reg]`` index, so the branch taken at this sweep point is fully
    determined statically — no measurement feedback is involved.

    Fast-fail (per CLAUDE.md): if ``compare_reg`` is not a sweep axis in ``point``
    the selector is not a registered loop counter (e.g. a measurement-conditional
    branch), which this lowering cannot resolve; raise instead of guessing.  An
    out-of-range index also raises.
    """

    if branch.compare_reg not in point:
        raise UnsupportedModuleError(
            f"Branch {branch.name!r} selects on register {branch.compare_reg!r}, "
            "which is not a sweep axis at this point; only deterministic branches "
            "driven by a registered sweep-loop counter can be lowered"
        )
    idx = point[branch.compare_reg]
    if not 0 <= idx < len(branch.branches):
        raise UnsupportedModuleError(
            f"Branch {branch.name!r} index {idx} is out of range "
            f"(it has {len(branch.branches)} branches)"
        )
    return branch.branches[idx]


def _iter_evolution_modules(
    modules: Sequence[Module], point: dict[str, int]
) -> list[Module]:
    """Flatten the timeline at one point, descending into the selected branch.

    A ``Branch`` is replaced by the sub-sequence chosen by ``point`` so that a
    qubit ``Pulse`` living inside the taken branch is visible to both the frame
    computation and the segment emission.  Nested branches are not flattened here;
    they fast-fail when encountered (see ``_select_branch`` callers).
    """

    flat: list[Module] = []
    for module in modules:
        if isinstance(module, Branch):
            flat.extend(_select_branch(module, point))
        else:
            flat.append(module)
    return flat


def _frame_detuning(
    modules: Sequence[Module],
    f_qubit_mhz: float,
    loop_counts: dict[str, int],
    point: dict[str, int],
) -> float:
    """Compute the single rotating-frame idle precession rate (rad/µs) at a point.

    The frame carrier ``f_ref`` is the frequency of the top-level qubit control
    pulses (``Pulse`` modules in the timeline); reset prep pulses live inside
    ``Reset`` modules and the readout is an ``AbsReadout``, so neither defines the
    qubit frame.  Returns ``2*pi*(f_qubit - f_ref)``.

    A qubit ``Pulse`` may sit inside the branch a ``Branch`` selects at this
    point (e.g. the g/e prep ``Branch("ge", [], Pulse(pi))``); the timeline is
    flattened via ``_iter_evolution_modules`` first so such a pulse still defines
    the frame, otherwise ``f_ref`` would be taken from the wrong (or no) pulse.

    Fast-fail (per CLAUDE.md): if the qubit pulses disagree in frequency at this
    point there is no single well-defined rotating frame, so raise rather than
    silently picking one.  When there is no qubit pulse at all (e.g. a pure
    onetone readout sweep, which has no Bloch evolution) the frame detuning is
    unused, so 0 is returned.
    """

    freqs = {
        round(_resolve_scalar(module.cfg.freq, loop_counts, point), 9)
        for module in _iter_evolution_modules(modules, point)
        if isinstance(module, Pulse) and module.cfg is not None
    }
    if not freqs:
        return 0.0
    if len(freqs) > 1:
        raise UnsupportedModuleError(
            "qubit control pulses have differing frequencies "
            f"{sorted(freqs)} MHz at this sweep point; a single rotating frame "
            "(required to lower idle precession) is undefined"
        )
    (f_ref_mhz,) = freqs
    return 2.0 * math.pi * (f_qubit_mhz - f_ref_mhz)


def lower_point(
    modules: Sequence[Module],
    sweep: Sequence[tuple[str, SweepCfg | int]] | None,
    sim: SimParams,
    f_qubit_ghz: float,
    point: dict[str, int],
    cycles2us: object,
    detune_offset: float = 0.0,
) -> LoweredPoint:
    """Lower the module tree to a Bloch timeline + readout plan at one sweep point.

    Parameters
    ----------
    modules
        The program's module list, in execution order (the same
        ``Sequence[Module]`` held by ``ModularProgramV2``).
    sweep
        The sweep axes as ``[(name, SweepCfg | count), ...]`` (outermost first),
        or None for an unswept program.
    sim
        Physical parameters (T1/T2/thermal_pop/pi_gain_len are read here).
    f_qubit_ghz
        The qubit 0->1 transition frequency in GHz at the current flux, computed
        by the engine.  Lowering never derives this itself.
    point
        The sweep multi-index ``{axis_name: index}``; empty for an unswept
        program.  A deterministic ``Branch`` selects its sub-sequence from this
        index (via the branch's ``compare_reg`` axis).
    cycles2us
        ``soccfg.cycles2us`` callable, used only to convert dmem cycle counts
        (non-uniform T1 path) into µs delays.
    detune_offset
        A static, global rotating-frame shift in rad/µs (same unit as
        ``Segment.delta``), added to every segment's detuning — both drives and
        idle/free segments.  Responsibility boundary: the engine owns the
        Lorentzian quasi-static detune ensemble / quadrature and feeds each
        ensemble node's static δ in here; lowering only applies the frame shift.
        The default ``0.0`` is the no-detune timeline.

    Returns
    -------
    LoweredPoint
        ``segments`` (pre-readout Bloch evolution) and ``readout`` (the readout
        plan).  Exactly one readout module is required.
    """

    sweep = sweep or []
    loop_counts = _loop_counts(sweep)
    f_qubit_mhz = f_qubit_ghz * _GHZ_TO_MHZ

    dmem_values = _collect_dmem_values(modules)
    frame_detuning = _frame_detuning(modules, f_qubit_mhz, loop_counts, point)
    idle_detuning = frame_detuning + detune_offset

    segments: list[Segment] = []
    readout: ReadoutPlan | None = None

    for module in modules:
        if isinstance(module, AbsReadout):
            if readout is not None:
                raise UnsupportedModuleError(
                    "more than one readout module in the timeline is not supported"
                )
            readout = _readout_plan(module, loop_counts, point)
            continue
        if isinstance(module, Branch):
            # Deterministic branch: descend into the selected sub-sequence only.
            # A Readout inside a branch makes the per-point readout window
            # branch-dependent (semantically complex), so it is forbidden here.
            for sub in _select_branch(module, point):
                if isinstance(sub, AbsReadout):
                    raise UnsupportedModuleError(
                        f"Branch {module.name!r} contains a readout module; "
                        "readout inside a branch is not supported"
                    )
                _lower_module(
                    sub,
                    sim,
                    f_qubit_mhz,
                    frame_detuning,
                    detune_offset,
                    idle_detuning,
                    loop_counts,
                    point,
                    dmem_values,
                    cycles2us,
                    segments,
                )
            continue
        _lower_module(
            module,
            sim,
            f_qubit_mhz,
            frame_detuning,
            detune_offset,
            idle_detuning,
            loop_counts,
            point,
            dmem_values,
            cycles2us,
            segments,
        )

    if readout is None:
        raise UnsupportedModuleError("no readout module found in the timeline")

    return LoweredPoint(segments=segments, readout=readout)


def inter_shot_relax_segment(
    modules: Sequence[Module],
    sweep: Sequence[tuple[str, SweepCfg | int]] | None,
    sim: SimParams,
    f_qubit_ghz: float,
    point: dict[str, int],
    duration_us: float,
    detune_offset: float = 0.0,
) -> Segment | None:
    """Return the passive free-evolution segment between two body executions.

    QICK's ``final_delay`` / `ProgramV2Cfg.relax_delay` happens after the readout
    and before the next body execution.  Lowering owns the rotating-frame policy,
    so the engine asks this helper for the matching idle detuning rather than
    reconstructing frame state from emitted segments.
    """

    duration = float(duration_us)
    if not math.isfinite(duration) or duration < 0.0:
        raise ValueError(f"relax_delay must be finite and >= 0.0, got {duration_us!r}")
    if duration == 0.0:
        return None

    sweep = sweep or []
    loop_counts = _loop_counts(sweep)
    f_qubit_mhz = f_qubit_ghz * _GHZ_TO_MHZ
    frame_detuning = _frame_detuning(modules, f_qubit_mhz, loop_counts, point)
    return _idle_segment(sim, duration, frame_detuning + detune_offset)


def _lower_module(
    module: Module,
    sim: SimParams,
    f_qubit_mhz: float,
    frame_detuning: float,
    detune_offset: float,
    idle_detuning: float,
    loop_counts: dict[str, int],
    point: dict[str, int],
    dmem_values: dict[str, list[int]],
    cycles2us: object,
    segments: list[Segment],
) -> None:
    """Lower one evolution module (pulse / delay / reset / dmem) into ``segments``.

    Shared by the top-level timeline loop and the selected-branch recursion.  It
    does NOT handle ``AbsReadout`` (the readout plan is owned by ``lower_point``)
    nor ``Branch`` (selection / nesting policy is owned by the caller); both fall
    through to the fast-fail at the end if they reach here.  A nested ``Branch``
    therefore fast-fails: it is unused by any real experiment, and resolving it
    would need branch-selection policy this single-level helper deliberately
    omits.
    """

    if isinstance(module, LoadValue):
        return  # dmem setup, already collected; emits no evolution
    if isinstance(module, (NoneReset, PulseReset, TwoPulseReset, BathReset)):
        segments.extend(
            _reset_segments(
                module,
                sim,
                f_qubit_mhz,
                frame_detuning,
                detune_offset,
                loop_counts,
                point,
            )
        )
        return
    if isinstance(module, Pulse):
        if module.cfg is None:
            return  # disabled optional pulse (e.g. init_pulse=None)
        segments.extend(
            _pulse_segments(
                module.cfg,
                sim,
                f_qubit_mhz,
                frame_detuning,
                detune_offset,
                loop_counts,
                point,
            )
        )
        return
    if isinstance(module, (Delay, DelayAuto, SoftDelay)):
        segments.append(
            _delay_segment(
                module,
                sim,
                idle_detuning,
                loop_counts,
                point,
                dmem_values,
                cycles2us,
            )
        )
        return

    raise UnsupportedModuleError(
        f"module {type(module).__name__}({module.name!r}) cannot be lowered "
        f"to a Bloch timeline"
    )
