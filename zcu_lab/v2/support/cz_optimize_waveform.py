"""CZ start display, FAST-DRAG coefficients and fixed-duration arb rendering."""

from dataclasses import dataclass
from enum import IntEnum
from math import lcm
from typing import Annotated, Literal, override

import numpy as np
from numpy.typing import NDArray
from pydantic import FiniteFloat
from qick import QickConfig
from qick.asm_v2 import QickProgramV2
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.core.waveform_format import WaveformSamples
from zcu_tools.program.v2 import (
    Pulse,
    PulseCfg,
    TimelineBuilder,
    flat_top_plateau_scale,
    pulse_output_scale,
)
from zcu_tools.program.v2.modules.waveform import InMemoryArbWaveformCfg, WaveformCfg

from zcu_lab.v2.support.paired_drive import QubitDriveCfg

# Three I harmonics retain the FAST-DRAG spectral design. Q uses their
# derivative, not separate weights. Fixing one I weight eliminates scale.
I_TERMS = 3
I_RATIO_COUNT = I_TERMS - 1
BETA_INDEX = I_TERMS
GAIN_INDEX = BETA_INDEX + 1
Z1 = GAIN_INDEX + 1
Z2 = Z1 + 1
PARAM_COUNT = Z2 + 1
SEARCH_PARAM_COUNT = PARAM_COUNT - 1


class FastDragStartCfg(ConfigBase):
    """Original FAST-DRAG design weights and signed derivative coefficient.

    c1, c2 and c3 multiply 1-cos(nθ), for n=1, 2 and 3. Their common scale
    is arbitrary. beta_ns sets the derivative quadrature in nanoseconds.
    The two calibrated pulses still play unmodified at evaluation zero.
    """

    c1: Annotated[FiniteFloat, FieldMeta(label="First I harmonic c1")]
    c2: Annotated[FiniteFloat, FieldMeta(label="Second I harmonic c2")]
    c3: Annotated[FiniteFloat, FieldMeta(label="Third I harmonic c3")]
    beta_ns: Annotated[FiniteFloat, FieldMeta(label="FAST-DRAG β", unit="ns")]


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
    """Original pulse display and separate CMA start for the same duration.

    iq is the played Qubit1 field sampled uniformly in DDS const units.
    style is the original shape identity. duration_us is its quantized support.
    cycle_us and samples_per_cycle are Qubit1's grid. carrier_mhz is the
    shared programmed carrier. coefficients are c1…c3 with max |c| = 1;
    beta_s is the FAST-DRAG coefficient in seconds. arb_gains are each line's
    raw gain at physical g=1; field_at_one is Qubit1's equivalent DDS gain.
    """

    iq: NDArray[np.complex128]
    style: StartStyle
    duration_us: float
    cycle_us: float
    samples_per_cycle: int
    carrier_mhz: float
    coefficients: NDArray[np.float64]
    beta_s: float
    arb_gains: NDArray[np.float64]
    field_at_one: float

    @property
    def i_ratios(self) -> NDArray[np.float64]:
        """Return the two free I weights relative to the anchor's magnitude."""
        return np.delete(self.coefficients, self._anchor)

    def expand_i(self, ratios: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return three I weights in harmonic order, inserting the signed anchor."""
        anchor = self._anchor
        return np.insert(ratios, anchor, self.coefficients[anchor])

    @property
    def _anchor(self) -> int:
        return int(np.argmax(np.abs(self.coefficients)))


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


_HARMONICS = np.arange(1, I_TERMS + 1, dtype=np.float64)


def _harmonic_angles(samples: int) -> NDArray[np.float64]:
    return (2 * np.pi / samples) * np.arange(samples)[:, None] * _HARMONICS


def _i_basis(angles: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1.0 - np.cos(angles)


def _quadrature(
    angles: NDArray[np.float64], coefficients: NDArray[np.float64]
) -> NDArray[np.float64]:
    # Q = (β/2π) dI/dt; multiply this unit-β/T profile by β/T.
    return np.sin(angles) @ (_HARMONICS * coefficients)


def _starting_coefficients(
    iq: NDArray[np.complex128],
    style: StartStyle,
    duration_s: float,
    fast_drag: FastDragStartCfg | None,
) -> tuple[NDArray[np.float64], float, float]:
    if fast_drag is not None:
        # The design weights describe shape, not the asset's playback scale.
        return (
            np.array([fast_drag.c1, fast_drag.c2, fast_drag.c3]),
            fast_drag.beta_ns * 1e-9,
            float(np.max(np.abs(iq))),
        )
    if style == StartStyle.COSINE:
        return np.array([1.0, 0.0, 0.0]), 0.0, float(np.max(np.abs(iq)))

    # Other starts need a projection. Fit I on the playback grid, then choose
    # the single β that best matches the original Q for that I.
    angles = _harmonic_angles(len(iq))
    basis = _i_basis(angles)
    coefficients = np.asarray(
        np.linalg.lstsq(basis, iq.real, rcond=None)[0], dtype=np.float64
    )
    derivative = _quadrature(angles, coefficients)
    norm = float(derivative @ derivative)
    beta_ratio = float(derivative @ iq.imag) / norm if norm else 0.0
    peak = float(np.max(np.hypot(basis @ coefficients, beta_ratio * derivative)))
    return coefficients, beta_ratio * duration_s, peak


def prepare_start(
    drive: QubitDriveCfg,
    board: QickConfig,
    *,
    g_max: float,
    fast_drag: FastDragStartCfg | None = None,
) -> StartShape:
    """Evaluate aligned pulses and seed the search without changing evaluation 0.

    Compare shapes on Qubit1's grid and require equal compiled durations. If
    given, fast_drag must describe the selected calibrated asset's design;
    use its weights and β directly. Cosine seeds only c1. Project other
    shapes onto the FAST-DRAG form. Reject an empty or nonfinite start and
    g_max values that overdrive either line in arb playback.
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
    coefficients, beta_s, p0 = _starting_coefficients(
        iq, start.style, duration_us * 1e-6, fast_drag
    )
    anchor = float(np.max(np.abs(coefficients)))
    if not (np.isfinite(p0) and p0 > 0.0 and np.isfinite(anchor) and anchor > 0.0):
        raise ValueError(
            "CZ Fourier start needs a finite nonzero I shape and complex peak"
        )
    # Keep the largest start weight fixed at ±1 for this search. Changing
    # either other weight changes shape rather than rescaling the envelope.
    coefficients /= anchor
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
        coefficients=coefficients,
        beta_s=beta_s,
        arb_gains=np.array(arb_gains, dtype=np.float64),
        field_at_one=arb_gains[0] * pulse_output_scale(board, ch, "arb"),
    )


def _sample_fourier(
    weights: NDArray[np.float64],
    *,
    duration_us: float,
    cycle_us: float,
    samples_per_cycle: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    samples = round(duration_us / cycle_us) * samples_per_cycle
    block = lcm(16, samples_per_cycle)
    padded = ((samples + block - 1) // block) * block
    step_us = cycle_us / samples_per_cycle
    time = np.arange(padded + 1, dtype=np.float64) * step_us
    i = np.zeros_like(time)
    q = np.zeros_like(time)
    angles = _harmonic_angles(samples)
    coefficients = weights[:I_TERMS]
    scale = float(np.max(np.abs(coefficients)))
    # The final normalized samples do not depend on the common I scale.
    if scale:
        coefficients = coefficients / scale
    beta_ratio = weights[BETA_INDEX] / (duration_us * 1e-6)
    i[:samples] = _i_basis(angles) @ coefficients
    q[:samples] = beta_ratio * _quadrature(angles, coefficients)
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
    """Render [c1, c2, c3, β (s), ...] on the DAC grid, with DMA zero padding.

    β sets Q = (β/2π)·dI/dt. Gain and Local Z do not change the unit-peak
    envelope. Nonfinite shape parameters or samples raise ValueError.
    The final time point is the support boundary, not a DAC sample.
    """
    weights = params[:GAIN_INDEX]
    if not np.all(np.isfinite(weights)):
        raise ValueError("CZ Fourier shape parameters must be finite")
    time, i, q = _sample_fourier(
        weights,
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
