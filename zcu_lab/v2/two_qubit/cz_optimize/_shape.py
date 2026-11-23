"""CZ start display, symmetric arb fit, and fixed-duration rendering."""

from dataclasses import dataclass
from enum import IntEnum
from math import lcm
from typing import Literal, override

import numpy as np
from numpy.typing import NDArray
from qick import QickConfig
from qick.asm_v2 import QickProgramV2
from scipy.interpolate import CubicSpline
from zcu_tools.core.waveform_format import WaveformSamples
from zcu_tools.program.v2 import (
    Pulse,
    PulseCfg,
    TimelineBuilder,
    flat_top_plateau_scale,
    pulse_output_scale,
)
from zcu_tools.program.v2.modules.waveform import InMemoryArbWaveformCfg, WaveformCfg

from .._drive import QubitDriveCfg

KNOTS = 3
KNOT_COLUMNS = 2 * KNOTS
GAIN_INDEX = KNOT_COLUMNS
Z1 = GAIN_INDEX + 1
Z2 = Z1 + 1
PARAM_COUNT = Z2 + 1


class StartStyle(IntEnum):
    """Persistent identities of the original waveform (not the CMA playback style)."""

    CONST = 0
    COSINE = 1
    GAUSS = 2
    DRAG = 3
    ARB = 4
    FLAT_TOP = 5


type _NativeStyle = Literal["const", "flat_top", "arb"]


class _EnvelopeEvaluator(QickProgramV2):
    """Collect the waveform owner's evaluated samples without compiling a pulse."""

    iq: NDArray[np.complex128]

    @override
    def add_envelope(self, ch, name, idata=None, qdata=None) -> None:
        self.iq = np.asarray(idata, dtype=np.complex128)
        if qdata is not None:
            self.iq += 1j * np.asarray(qdata, dtype=np.float64)


@dataclass(frozen=True)
class _StartingEnvelope:
    style: StartStyle
    iq: NDArray[
        np.complex128
    ]  # Display shape; one is full-scale in the starting style.
    duration_us: float


@dataclass(frozen=True)
class StartShape:
    """Display and fit snapshot for the start; evaluation 0 plays the unmodified cfg.

    iq is the played Qubit1 field sampled uniformly in DDS const units.
    style is the original shape identity. duration_us is its quantized support.
    cycle_us and samples_per_cycle are Qubit1's grid. carrier_mhz is the
    shared programmed carrier. knots hold the free I then Q values.
    arb_gains are each line's raw gain at physical g=1. field_at_one is
    Qubit1's equivalent DDS gain at g=1 for the fit and candidates.
    """

    iq: NDArray[np.complex128]
    style: StartStyle
    duration_us: float
    cycle_us: float
    samples_per_cycle: int
    carrier_mhz: float
    knots: NDArray[np.float64]
    arb_gains: NDArray[np.float64]
    field_at_one: float


def _native_style(style: StartStyle) -> _NativeStyle:
    if style == StartStyle.CONST:
        return "const"
    if style == StartStyle.FLAT_TOP:
        return "flat_top"
    return "arb"


def _starting_envelope(
    waveform: WaveformCfg, board: QickConfig, ch: int
) -> _StartingEnvelope:
    """Evaluate the configured shape for display and fit, not for playback."""
    if isinstance(waveform, InMemoryArbWaveformCfg):
        raise ValueError("Two-line CZ drive requires a serializable starting waveform")
    spc = int(board["gens"][ch]["samps_per_clk"])
    style = StartStyle[waveform.style.upper()]
    wave = waveform.build("cz_start")
    native = wave.to_wav_kwargs()
    native_style = native["style"]
    dds_cycles = 0
    if native_style != "arb":
        # For flat_top, QICK's length keyword is only the DDS plateau.
        length = native.get("length")
        if not isinstance(length, (float, int)):
            raise ValueError("Two-line CZ drive waveform length must be scalar")
        dds_cycles = int(board.us2cycles(length, gen_ch=ch))
    if native_style == "const":
        return _StartingEnvelope(
            style,
            np.ones(dds_cycles * spc, dtype=np.complex128),
            float(board.cycles2us(dds_cycles, gen_ch=ch)),
        )

    prog = _EnvelopeEvaluator(board)
    wave.create(prog, ch)
    iq = prog.iq / board.get_maxv(ch)
    if native_style == "flat_top":
        iq = _flat_top_samples(iq, dds_cycles * spc, board, ch)
    return _StartingEnvelope(
        style, iq, float(board.cycles2us(len(iq) // spc, gen_ch=ch))
    )


def _flat_top_samples(
    ramp: NDArray[np.complex128], plateau_samples: int, board: QickConfig, ch: int
) -> NDArray[np.complex128]:
    if plateau_samples < 0:
        raise ValueError(
            "Two-line CZ drive flat_top plateau duration must be nonnegative"
        )
    # Express each segment in units of the full-scale flat_top peak.
    scale = pulse_output_scale(board, ch, "flat_top")
    ramp = ramp * (pulse_output_scale(board, ch, "arb") / scale)
    plateau = flat_top_plateau_scale(board, ch) / scale
    center = len(ramp) // 2
    return np.concatenate(
        (ramp[:center], np.full(plateau_samples, plateau), ramp[center:])
    )


def _same_start(first: _StartingEnvelope, second: _StartingEnvelope) -> bool:
    return (
        first.style == second.style
        and first.duration_us == second.duration_us
        and np.array_equal(first.iq, second.iq)
    )


def _compiled_cycles(pulse: PulseCfg, board: QickConfig) -> int:
    builder = TimelineBuilder(board)
    module = Pulse("cz_start", pulse)
    builder.emit(module)
    program = builder.build()
    duration = program.get_pulse_param(module.pulse_name(program), "total_length")
    if not isinstance(duration, (float, int)):
        raise ValueError("Two-line CZ drive waveform length must be scalar")
    return int(board.us2cycles(duration, gen_ch=pulse.ch))


def _spline(knots: NDArray[np.float64], *, in_phase: bool) -> CubicSpline:
    # Start I and Q at zero; only Q must also vanish at the center.
    zero = np.zeros_like(knots[:1])
    half = np.concatenate((zero, knots) if in_phase else (zero, knots, zero))
    reflection = half[-2::-1] if in_phase else -half[-2::-1]
    full = np.concatenate((half, reflection))
    return CubicSpline(np.linspace(0.0, 1.0, len(full)), full, bc_type="natural")


def _fit_knots(iq: NDArray[np.complex128]) -> NDArray[np.float64]:
    # Reflect the centers of the DAC sample intervals. Odd I and even Q
    # disappear from the mean but remain in the measured start.
    domain = (np.arange(len(iq)) + 0.5) / len(iq)
    even_i = (iq.real + iq.real[::-1]) / 2
    odd_q = (iq.imag - iq.imag[::-1]) / 2
    basis = np.eye(KNOTS)
    return np.concatenate(
        [
            np.linalg.lstsq(
                _spline(basis, in_phase=in_phase)(domain), values, rcond=None
            )[0]
            for in_phase, values in ((True, even_i), (False, odd_q))
        ]
    )


def prepare_start(
    drive: QubitDriveCfg, board: QickConfig, *, g_max: float
) -> StartShape:
    """Evaluate and fit aligned scalar pulses before acquiring any counts.

    Compare shapes on Qubit1's display grid, then require equal played durations
    on the two native generator grids before fitting on Qubit1's grid.
    """
    first, second = drive.qubit1, drive.qubit2
    start = _starting_envelope(first.waveform, board, first.ch)
    second_on_first_grid = (
        start
        if first.waveform == second.waveform
        else _starting_envelope(second.waveform, board, first.ch)
    )
    if not _same_start(start, second_on_first_grid):
        raise ValueError(
            "drive.qubit1.waveform and drive.qubit2.waveform must have the same CZ shape and duration"
        )
    ch = first.ch
    first_cycles = _compiled_cycles(first, board)
    second_cycles = _compiled_cycles(second, board)
    duration_us = float(board.cycles2us(first_cycles, gen_ch=ch))
    second_duration = float(board.cycles2us(second_cycles, gen_ch=second.ch))
    if duration_us != second_duration:
        raise ValueError(
            "Starting CZ pulses must have equal generator-quantized durations: "
            f"Qubit1 {duration_us:.12g} µs, Qubit2 {second_duration:.12g} µs"
        )
    if len(start.iq) < 2:
        raise ValueError("Two-line CZ drive must contain at least two samples")

    cycle_us = float(board.cycles2us(1, gen_ch=ch))
    samples_per_cycle = int(board["gens"][ch]["samps_per_clk"])
    # QICK can append a zero-gain FIR clearing segment to a flat-top pulse.
    # Include the compiled support in T, but do not model the analog response.
    tail_samples = first_cycles * samples_per_cycle - len(start.iq)
    iq = np.pad(start.iq, (0, tail_samples)) if tail_samples else start.iq
    knots = _fit_knots(iq)
    _, fit_i, fit_q = _sample_knots(
        knots,
        duration_us=duration_us,
        cycle_us=cycle_us,
        samples_per_cycle=samples_per_cycle,
    )
    p0 = float(np.max(np.hypot(fit_i, fit_q)))
    if not np.isfinite(p0) or p0 == 0.0:
        raise ValueError("CZ starting fit must have a finite nonzero complex peak")

    # Only re-express the shape in the CMA box. Keep the original fit's
    # physical field by absorbing its unscaled peak into both gains once.
    knots /= max(1.0, float(np.max(np.abs(knots))))
    original = _native_style(start.style)
    arb_gains = []
    for name, pulse in (("qubit1", first), ("qubit2", second)):
        gain = (
            float(pulse.gain)
            * p0
            * pulse_output_scale(board, pulse.ch, original)
            / pulse_output_scale(board, pulse.ch, "arb")
        )
        max_raw_gain = abs(gain) * g_max
        if max_raw_gain > 1.0:
            raise ValueError(
                f"g_max with drive.{name}.gain produces |raw gain|={max_raw_gain} > 1 for arb"
            )
        arb_gains.append(gain)
    return StartShape(
        iq=iq * float(first.gain) * pulse_output_scale(board, ch, original),
        style=start.style,
        duration_us=duration_us,
        cycle_us=cycle_us,
        samples_per_cycle=samples_per_cycle,
        carrier_mhz=float(first.freq),
        knots=knots,
        arb_gains=np.array(arb_gains, dtype=np.float64),
        field_at_one=arb_gains[0] * pulse_output_scale(board, ch, "arb"),
    )


def _sample_knots(
    knots: NDArray[np.float64],
    *,
    duration_us: float,
    cycle_us: float,
    samples_per_cycle: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    if not np.all(np.isfinite(knots)):
        raise ValueError("CZ envelope knots must be finite")
    samples = round(duration_us / cycle_us) * samples_per_cycle
    block = lcm(16, samples_per_cycle)
    padded = ((samples + block - 1) // block) * block
    step_us = cycle_us / samples_per_cycle
    domain = np.arange(samples, dtype=np.float64) / samples
    time = np.arange(padded + 1, dtype=np.float64) * step_us
    i = np.zeros_like(time)
    q = np.zeros_like(time)
    i[:samples] = _spline(knots[:KNOTS], in_phase=True)(domain)
    q[:samples] = _spline(knots[KNOTS:], in_phase=False)(domain)
    if not (np.all(np.isfinite(i)) and np.all(np.isfinite(q))):
        raise ValueError("CZ envelope samples must be finite")
    # The last time point is the support boundary, not a DAC sample.
    return time, i, q


def render_candidate(
    params: NDArray[np.float64],
    *,
    duration_us: float,
    cycle_us: float,
    samples_per_cycle: int,
) -> WaveformSamples:
    """Sample a fixed-duration arb and normalize I and Q by one positive scale."""
    knots = params[:KNOT_COLUMNS]
    if not np.all(np.isfinite(knots)):
        raise ValueError("CZ envelope knots must be finite")
    scale = float(np.max(np.abs(knots)))
    # The spline is linear in the knots. Scaling before evaluation avoids
    # losing subnormal shapes to floating-point underflow inside the spline.
    time, i, q = _sample_knots(
        knots / scale if scale else knots,
        duration_us=duration_us,
        cycle_us=cycle_us,
        samples_per_cycle=samples_per_cycle,
    )
    peak = float(np.max(np.hypot(i, q)))
    if peak:
        i /= peak
        q /= peak
    measured = float(np.max(np.hypot(i, q)))
    if measured > 1.0:
        # The arb validator is strict. Correct roundoff with one scale for
        # the entire envelope, not per-sample clipping.
        factor = (1.0 - 4 * np.finfo(np.float64).eps) / measured
        i *= factor
        q *= factor
    return WaveformSamples(time, i, q if np.any(q) else None)
