"""CZ start display, symmetric ramp fit, and single-arb candidate rendering."""

from dataclasses import dataclass
from enum import IntEnum
from typing import Literal, override

import numpy as np
from numpy.typing import NDArray
from qick import QickConfig
from qick.asm_v2 import QickProgramV2
from scipy.interpolate import CubicSpline
from scipy.optimize import lsq_linear
from zcu_tools.core.waveform_format import WaveformSamples
from zcu_tools.program.v2 import pulse_output_scale
from zcu_tools.program.v2.modules.waveform import (
    ConstWaveformCfg,
    FlatTopWaveformCfg,
    InMemoryArbWaveformCfg,
    WaveformCfg,
)

from .._drive import QubitDriveCfg

KNOTS = 3
RAMP_SCALE = 2 * KNOTS
PLATEAU = RAMP_SCALE + 1
GAIN_SCALE = PLATEAU + 1
Z1 = GAIN_SCALE + 1
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
    ramp: NDArray[np.complex128]
    midpoint: complex
    plateau_us: float
    ramp_us: float
    native_duration_us: float


@dataclass(frozen=True)
class StartShape:
    """Display and fit snapshot for the start; evaluation 0 plays the unmodified cfg.

    iq is the Qubit1 field sampled uniformly in units of DDS const at gain 1.
    style is the original shape identity. duration_us is its quantized support.
    ramp_us is the reference ramp duration used for the fit; plateau_us is the
    starting fit plateau. cycle_us and samples_per_cycle are Qubit1's grid.
    carrier_mhz is the shared programmed carrier. knots hold I then Q values,
    normalized to one at the real midpoint. arb_gains are each line's raw gain
    at physical g=1 for a normalized arb. field_at_one is Qubit1's equivalent
    DDS gain at g=1 for the fit and candidates.
    """

    iq: NDArray[np.complex128]
    style: StartStyle
    duration_us: float
    ramp_us: float
    plateau_us: float
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


def _length(wave: WaveformCfg) -> float:
    length = wave.build("cz_start").length
    if not isinstance(length, (float, int)):
        raise ValueError("Two-line CZ drive waveform length must be scalar")
    return float(length)


def _starting_envelope(
    waveform: WaveformCfg, board: QickConfig, ch: int
) -> _StartingEnvelope:
    """Evaluate the configured shape for display and fit, not for playback."""
    if isinstance(waveform, InMemoryArbWaveformCfg) or (
        isinstance(waveform, FlatTopWaveformCfg)
        and isinstance(waveform.raise_waveform, InMemoryArbWaveformCfg)
    ):
        raise ValueError("Two-line CZ drive requires a serializable starting waveform")
    spc = int(board["gens"][ch]["samps_per_clk"])
    cycle_us = float(board.cycles2us(1, gen_ch=ch))
    style = StartStyle[waveform.style.upper()]
    length = _length(waveform)
    if isinstance(waveform, ConstWaveformCfg):
        cycles = int(board.us2cycles(length, gen_ch=ch))
        ramp = np.sin(np.pi * (np.arange(6 * spc) + 0.5) / (6 * spc)) ** 2
        return _StartingEnvelope(
            style,
            np.ones(cycles * spc, dtype=np.complex128),
            ramp.astype(np.complex128),
            1.0 + 0.0j,
            max(0.0, (cycles - 3) * cycle_us),
            6 * cycle_us,
            cycles * cycle_us,
        )

    prog = _EnvelopeEvaluator(board)
    waveform.build("cz_start").create(prog, ch)
    ramp = prog.iq / board.get_maxv(ch)
    # QICK built-ins use a midpoint between the two central samples. Arb
    # interpolates at t=0, T/N, ..., so its midpoint is at N/2.
    center = len(ramp) // 2
    if center == 0:
        raise ValueError("Two-line CZ drive ramp must contain at least two samples")
    ramp_cfg = (
        waveform.raise_waveform
        if isinstance(waveform, FlatTopWaveformCfg)
        else waveform
    )
    midpoint = complex(
        ramp[center]
        if ramp_cfg.style == "arb"
        else (ramp[center - 1] + ramp[center]) / 2
    )
    ramp_us = len(ramp) / spc * cycle_us
    if not isinstance(waveform, FlatTopWaveformCfg):
        return _StartingEnvelope(style, ramp, ramp, midpoint, 0.0, ramp_us, ramp_us)

    plateau_cycles = int(
        board.us2cycles(length - _length(waveform.raise_waveform), gen_ch=ch)
    )
    if plateau_cycles < 0:
        raise ValueError(
            "Two-line CZ drive flat_top plateau duration must be nonnegative"
        )
    plateau_us = plateau_cycles * cycle_us
    # The ramp is product output like arb. The flat segment sets the native
    # flat-top scale; do not normalize the DDS segment by the ramp midpoint.
    ramp_ratio = pulse_output_scale(board, ch, "arb") / pulse_output_scale(
        board, ch, "flat_top"
    )
    iq = np.concatenate(
        (
            ramp[:center] * ramp_ratio,
            np.ones(plateau_cycles * spc, dtype=np.complex128),
            ramp[center:] * ramp_ratio,
        )
    )
    return _StartingEnvelope(
        style, iq, ramp, midpoint, plateau_us, ramp_us, ramp_us + plateau_us
    )


def _same_start(first: _StartingEnvelope, second: _StartingEnvelope) -> bool:
    return (
        first.style == second.style
        and first.plateau_us == second.plateau_us
        and first.ramp_us == second.ramp_us
        and np.array_equal(first.iq, second.iq)
    )


def _spline(knots: NDArray[np.float64], *, in_phase: bool) -> CubicSpline:
    # Reflection clamps I's midpoint slope to zero. Antireflection leaves
    # Q's midpoint slope free while keeping its midpoint at zero.
    reflection = knots[-2::-1] if in_phase else -knots[-2::-1]
    symmetric_knots = np.concatenate((knots, reflection), axis=0)
    return CubicSpline(
        np.linspace(0.0, 2.0, len(symmetric_knots)), symmetric_knots, bc_type="natural"
    )


def _fit_knots(start: _StartingEnvelope) -> NDArray[np.float64]:
    # Fit the ascending half only. The renderer constructs the other half by
    # reflecting I and negating Q; the measured start never uses this fit.
    half = len(start.ramp) // 2
    offset = 0.5
    domain = (np.arange(half) + offset) / half
    normalized = start.ramp[:half] / start.midpoint.real
    result = []
    for in_phase, samples in ((True, normalized.real), (False, normalized.imag)):
        basis = np.eye(KNOTS + 2, dtype=np.float64)[:, 1:]
        interpolated = _spline(basis, in_phase=in_phase)(domain)
        target = samples - interpolated[:, -1] if in_phase else samples
        fit = lsq_linear(interpolated[:, :-1], target, bounds=(-1.0, 1.0))
        result.append(fit.x)
    return np.concatenate(result)


def prepare_start(
    drive: QubitDriveCfg, board: QickConfig, *, g_max: float
) -> StartShape:
    """Evaluate and fit aligned scalar pulses before acquiring any counts.

    Compare both declared shapes on a common display grid, then fit the first
    half of the ramp on Qubit1's generator grid. Reject mismatched shapes,
    midpoints off the positive real axis and candidate raw gain above one.
    Only arb is a reachable CMA representation in this experiment revision.
    """
    first, second = drive.qubit1, drive.qubit2
    start = _starting_envelope(first.waveform, board, first.ch)
    if first.waveform != second.waveform and not _same_start(
        start, _starting_envelope(second.waveform, board, first.ch)
    ):
        raise ValueError(
            "drive.qubit1.waveform and drive.qubit2.waveform must have the same CZ shape and duration"
        )
    if start.midpoint.real <= 0.0 or abs(start.midpoint.imag) > 1.0 / board.get_maxv(
        first.ch
    ):
        raise ValueError(
            f"Two-line CZ drive ramp midpoint must have positive I and zero Q; got I={start.midpoint.real:.6g}, Q={start.midpoint.imag:.6g}"
        )

    original = _native_style(start.style)
    arb_gains = []
    for name, pulse in (("qubit1", first), ("qubit2", second)):
        # An existing flat-top plateau is a separate DDS segment, so its
        # physical reference does not shrink with an underfilled ramp.
        if start.style in (StartStyle.CONST, StartStyle.FLAT_TOP):
            scale_at_one = pulse_output_scale(board, pulse.ch, original)
        else:
            scale_at_one = start.midpoint.real * pulse_output_scale(
                board, pulse.ch, original
            )
        gain = (
            float(pulse.gain)
            * scale_at_one
            / pulse_output_scale(board, pulse.ch, "arb")
        )
        if abs(gain) * g_max > 1.0:
            raise ValueError(
                f"g_max with drive.{name}.gain produces |raw gain|={abs(gain) * g_max:.6g} > 1 for arb"
            )
        arb_gains.append(gain)
    ch = first.ch
    return StartShape(
        iq=start.iq * float(first.gain) * pulse_output_scale(board, ch, original),
        style=start.style,
        duration_us=start.native_duration_us,
        ramp_us=start.ramp_us,
        plateau_us=start.plateau_us,
        cycle_us=float(board.cycles2us(1, gen_ch=ch)),
        samples_per_cycle=int(board["gens"][ch]["samps_per_clk"]),
        carrier_mhz=float(first.freq),
        knots=_fit_knots(start),
        arb_gains=np.array(arb_gains, dtype=np.float64),
        field_at_one=arb_gains[0] * pulse_output_scale(board, ch, "arb"),
    )


def render_candidate(
    params: NDArray[np.float64],
    *,
    ramp_us: float,
    cycle_us: float,
    samples_per_cycle: int,
) -> WaveformSamples:
    """Sample the physical candidate coordinates into a normalized arb envelope.

    The shape depends on knots, ramp scale, and plateau duration (s), not g
    or local Z. Quantize the plateau to a fabric cycle and zero-pad the end to
    a whole cycle. The last time point is the support boundary, not a DAC sample.
    The caller rejects peaks above one before constructing an arb cfg.
    """
    ramp = float(params[RAMP_SCALE]) * ramp_us
    plateau = round(float(params[PLATEAU]) * 1e6 / cycle_us) * cycle_us
    cycles = max(3, int(np.ceil((ramp + plateau) / cycle_us - 1e-12)))
    step_us = cycle_us / samples_per_cycle
    time = np.arange(cycles * samples_per_cycle + 1, dtype=np.float64) * step_us
    half = ramp / 2
    rising = time < half
    flat = (time >= half) & (time < half + plateau)
    falling = (time >= half + plateau) & (time < ramp + plateau)
    i_spline = _spline(np.r_[0.0, params[:KNOTS], 1.0], in_phase=True)
    q_spline = _spline(np.r_[0.0, params[KNOTS:RAMP_SCALE], 0.0], in_phase=False)
    up = time[rising] / half
    down = (ramp + plateau - time[falling]) / half
    i = np.zeros_like(time)
    q = np.zeros_like(time)
    i[rising], i[flat], i[falling] = i_spline(up), 1.0, i_spline(down)
    q[rising], q[falling] = q_spline(up), -q_spline(down)
    return WaveformSamples(time, i, q if np.any(q) else None)
