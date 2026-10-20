"""Common carrier frame with explicit SW2 beat channels and secular CF4.

This numerical owner does not choose dispatch order or retain a cache. Products
are immutable model-only data. Admission is row-local and must be repeated for
current gains, envelopes and duration, even when coefficients are reused.
"""

from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations_with_replacement

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

from ..kernel import SegmentTable
from ..secular import ModelSource, SecularModel
from ._common_frame import (
    Channels,
    amplitudes,
    coefficients,
    first_order_channels,
    propagate,
    remainder_channels,
    remainder_estimates,
)
from ._secular import secular_operators

# These are calibrated indicators, not rigorous trace-distance certificates.
COMMON_FRAME_POLICY = ("common-frame-sw2-v1", 0.03, 0.01, 8.0, 0.08, 2.0)
_COHERENT_LIMIT, _BATH_LIMIT = 0.03, 0.01
_MAX_STEP_NS, _PHASE_CYCLES, _MIN_STEP_NS = 8.0, 0.08, 2.0


def coefficient_keys(
    table: SegmentTable, rows: NDArray[np.int64], source: ModelSource
) -> NDArray[np.void]:
    """Hash complete coefficient physics; gains, phases and envelopes stay row-local.

    Source identity describes the complete builder. Model identity and level
    describe the local basis, bath and drives. Labels and the retained SW2 set
    are deterministic functions of these inputs under COMMON_FRAME_POLICY.
    """
    dc = source.model((0,) * table.level.shape[1]).identity
    keys = np.empty(len(rows), dtype="V32")
    for index, row in enumerate(rows):
        count = int(table.tones[row])
        model = source.model(tuple(int(v) for v in table.level[row]))
        digest = sha256(repr((dc, model.identity, COMMON_FRAME_POLICY)).encode())
        for values in (
            table.level[row],
            table.tone_port[row, :count],
            table.tone_frequency_ghz[row, :count],
        ):
            digest.update(values.tobytes())
        keys[index] = np.void(digest.digest())
    return keys


@dataclass(frozen=True)
class CommonFrame:
    """Immutable model-only coefficients, including cost-rejected builds.

    model: full secular model, including basis, rates and drive quadratures.
    ports: ordered model drive indices (one or two, duplicates permitted).
    frequencies_ghz: matching signed carriers in cycles/ns, read-only copy.
    step_ns: CF4 step ceiling from active first/SW2 frequencies; a ceiling below
        2 ns rejects on work, not on an accuracy/carrier-spacing veto.
    cost_bypass: True if a lower bound already exceeded the work policy or the
        active SW2 channels did. No remainder tensors are built in either case.
    channels: opaque native first/SW2 Hamiltonian and endpoint generators, or
        None after an early cost exit. Frequencies use angular ns**-1 there.
    remainder: (monomials,M,M) absolute omitted-channel GHz coefficients.
    monomials: (terms,2*tones) exponents of signed tone gains; degrees: their
        total orders (2, 3 or 4). Grouping sums absolute ordered coefficients.
    slow_norms, fast_norms: (2*tones,) spectral norms in GHz/gain for BCH tail.
    Callers use nbytes for array accounting, excluding the borrowed model and
    Python metadata. The
    caller owns residency; no global or instance cache is created by this owner.
    """

    model: SecularModel
    ports: tuple[int, ...]
    frequencies_ghz: NDArray[np.float64]
    step_ns: float
    cost_bypass: bool
    channels: Channels | None
    remainder: NDArray[np.float64]
    monomials: NDArray[np.int64]
    degrees: NDArray[np.int64]
    slow_norms: NDArray[np.float64]
    fast_norms: NDArray[np.float64]

    @property
    def nbytes(self) -> int:
        """Owned array bytes, excluding the borrowed model and scalar metadata."""
        arrays: list[np.ndarray] = [
            self.frequencies_ghz,
            self.remainder,
            self.monomials,
            self.degrees,
            self.slow_norms,
            self.fast_norms,
        ]
        if self.channels is not None:
            arrays.extend(
                value for value in self.channels if isinstance(value, np.ndarray)
            )
        return sum(array.nbytes for array in arrays)


@dataclass(frozen=True)
class CommonFrameAdmission:
    """Current row's detuning-aware indicators, not mathematical certificates.

    accepted: coherent sum <= .03, bath <= .01, no denominator/work rejection.
    reason: 'accepted', 'cost', 'denominator', 'coherent', or 'bath'. Dispatch
        must use an existing algorithm for every rejected row.
    step_ns, steps: CF4 ceiling in ns and ceil(duration/step), at least one.
    eta: maximum absolute matrix row sum of peak-weighted SW generators.
    second, third, fourth: omitted order-2/3/4 channel integral indicators.
    tail: fifth-and-higher BCH norm tail, without oscillatory cancellation.
    derivative: envelope-derivative omission indicator.
    bath: first-order oscillatory bath dressing plus secular T*gamma*eta**2.
    Cost/denominator exits have zero indicators, which do not imply accuracy.
    Slow-propagator detuning changes and mixed bath/envelope higher orders are
    not rigorously bounded. These are calibrated admission data only.
    """

    accepted: bool
    reason: str
    step_ns: float
    steps: int
    eta: float = 0.0
    second: float = 0.0
    third: float = 0.0
    fourth: float = 0.0
    tail: float = 0.0
    derivative: float = 0.0
    bath: float = 0.0

    @property
    def coherent(self) -> float:
        """Sum of omitted coherent channels, BCH tail and derivative indicator."""
        return self.second + self.third + self.fourth + self.tail + self.derivative


def _step(frequency: float) -> float:
    return min(_MAX_STEP_NS, _PHASE_CYCLES / max(frequency, 1e-12))


def _monomial_inventory(count: int):
    monomials = []
    degrees = []
    powers = 5 ** np.arange(count, dtype=np.int64)
    groups = np.full(5**count, -1, dtype=np.int64)
    for degree in (2, 3, 4):
        for indices in combinations_with_replacement(range(count), degree):
            exponents = np.bincount(indices, minlength=count)
            groups[int(exponents @ powers)] = len(monomials)
            monomials.append(exponents)
            degrees.append(degree)
    return (
        np.array(monomials, dtype=np.int64),
        np.array(degrees, dtype=np.int64),
        groups,
        powers,
    )


def _first_order_step(energies, full, frequencies) -> float:
    carrier = float(frequencies.mean())
    labels = np.rint(energies / carrier)
    mask = labels[:, None] - labels[None, :] + 1 == 0
    active = np.any(np.abs(full[:, mask]) > 1e-16, axis=1)
    frequency = float(np.max(np.abs(frequencies[active] - carrier), initial=0))
    return _step(frequency)


def common_frame_cost_viable(
    model: SecularModel, ports: tuple[int, ...], frequencies_ghz: NDArray[np.float64]
) -> bool:
    """Whether first-order playback can meet the unchanged build work policy.

    Inputs are the ordered ports and signed carriers used by build_common_frame.
    False proves its first-order cost exit without building SW2/remainders or
    evaluating numerical admission. True says nothing about later admission.
    This is an active-channel work bound, not a carrier-spacing accuracy veto.
    """
    if abs(float(np.mean(frequencies_ghz))) < 1e-10:
        return False
    full = np.array(
        [
            (model.drives[p].in_phase_ghz - 1j * model.drives[p].quadrature_ghz) / 2
            for p in ports
        ]
    )
    return (
        _first_order_step(
            model.energies_ghz - model.energies_ghz[0], full, frequencies_ghz
        )
        >= _MIN_STEP_NS
    )


def common_frame_strengths(
    model: SecularModel, ports: tuple[int, ...], frequencies_ghz: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return (tones, states) absolute SW row sums for cold-cost risk pricing.

    Inputs match build_common_frame. Only first-order algebra is built, not SW2
    or remainder tensors. Multiplying each tone by its gain/envelope peak and
    summing gives the eta used by the build-risk cost model. This value is not
    an admission decision or an accuracy veto. Zero mean carrier returns inf
    so dispatch retains the original solver rather than building an invalid frame.
    """
    if abs(float(np.mean(frequencies_ghz))) < 1e-10:
        return np.full((len(ports), len(model.energies_ghz)), np.inf)
    full = np.array(
        [
            (model.drives[p].in_phase_ghz - 1j * model.drives[p].quadrature_ghz) / 2
            for p in ports
        ]
    )
    generators = first_order_channels(
        model.energies_ghz - model.energies_ghz[0], full, frequencies_ghz
    )[-1]
    return np.abs(generators).sum(axis=2).reshape(len(ports), 2, -1).sum(axis=1)


def build_common_frame(
    model: SecularModel, ports: tuple[int, ...], frequencies_ghz: NDArray[np.float64]
) -> CommonFrame:
    """Build a reusable one/two-tone model-only common-frame product.

    ports are ordered indices into model.drives; carriers are finite signed GHz
    with nonzero mean. Raises ValueError for invalid domain inputs. Gains,
    phases, duration and sampled envelopes do not enter these coefficients.
    Labels are round((E-E0)/mean(f)). Retain first-order matching labels and
    ordered second-order SW beat/two-photon channels. Endpoint SW dressing
    restores the omitted fast channels to first order.

    A first-order CF4 cost lower bound precedes SW construction. Active SW2
    channels refine it before remainder construction. Equal signed monomials
    share packed absolute remainder matrices, instead of dense q**4 tensors.
    Arrays are read-only; independent calls and worker threads share no state.
    """
    frequencies = np.array(frequencies_ghz, dtype=np.float64, copy=True)
    if (
        not 1 <= len(ports) <= 2
        or frequencies.shape != (len(ports),)
        or not np.isfinite(frequencies).all()
        or abs(float(frequencies.mean())) < 1e-10
        or any(p < 0 or p >= len(model.drives) for p in ports)
    ):
        raise ValueError(
            "common frame requires one/two valid ports and finite carriers with nonzero mean"
        )
    ports = tuple(int(port) for port in ports)
    energies = model.energies_ghz - model.energies_ghz[0]
    full = np.array(
        [
            (model.drives[p].in_phase_ghz - 1j * model.drives[p].quadrature_ghz) / 2
            for p in ports
        ]
    )
    labels = np.rint(energies / float(frequencies.mean()))
    step = _first_order_step(energies, full, frequencies)
    channels = None
    monomials = np.empty((0, 2 * len(ports)), dtype=np.int64)
    degrees = np.empty(0, dtype=np.int64)
    remainder = np.empty((0, len(energies), len(energies)))
    slow_norms, fast_norms = np.empty(0), np.empty(0)
    if step >= _MIN_STEP_NS:
        channels = coefficients(energies, full, frequencies)
        active_channels = np.any(np.abs(channels.operators) > 1e-16, axis=(1, 2))
        slow_frequency = float(
            np.max(np.abs(channels.omega[active_channels]), initial=0) / (2 * np.pi)
        )
        step = _step(slow_frequency)
        if step >= _MIN_STEP_NS and channels.denominator_min > 1e-10:
            monomials, degrees, groups, powers = _monomial_inventory(2 * len(ports))
            remainder = remainder_channels(channels, labels, groups, powers)
            count = 2 * len(ports)
            slow_norms = np.array(
                [np.linalg.norm(channels.operators[a], 2) for a in range(count)],
                dtype=np.float64,
            )
            fast_norms = np.array(
                [
                    np.linalg.norm(channels.signed[a] - channels.operators[a], 2)
                    for a in range(count)
                ],
                dtype=np.float64,
            )
    arrays: list[np.ndarray] = [
        frequencies,
        remainder,
        monomials,
        degrees,
        slow_norms,
        fast_norms,
    ]
    if channels is not None:
        arrays.extend(value for value in channels if isinstance(value, np.ndarray))
    for array in arrays:
        array.setflags(write=False)
    return CommonFrame(
        model,
        ports,
        frequencies,
        step,
        step < _MIN_STEP_NS,
        channels,
        remainder,
        monomials,
        degrees,
        slow_norms,
        fast_norms,
    )


def _validate_row(frame: CommonFrame, table: SegmentTable, row: int) -> None:
    if not 0 <= row < len(table.timeline):
        raise ValueError("common-frame row is out of range")
    count = int(table.tones[row])
    if (
        count != len(frame.ports)
        or tuple(table.tone_port[row, :count]) != frame.ports
        or not np.array_equal(
            table.tone_frequency_ghz[row, :count], frame.frequencies_ghz
        )
        or tuple(table.level[row]) != frame.model.level
    ):
        raise ValueError("common-frame row does not match model/ports/carriers")
    if table.flux_waveform is not None and (
        table.flux_waveform_available is None
        or table.flux_waveform_available[table.timeline[row]]
    ):
        raise ValueError("common frame requires fixed flux")


def _row_envelopes(frame: CommonFrame, table: SegmentTable, row: int):
    count = len(frame.ports)
    envelopes = [table.envelopes[i] for i in table.tone_envelope[row, :count]]
    counts = np.array([len(e.samples) for e in envelopes], dtype=np.int64)
    width = max(1, int(counts.max()))
    times = np.zeros((count, width))
    values = np.ones((count, width), dtype=np.complex128)
    peaks, slopes = np.ones(count), np.zeros(count)
    for a, envelope in enumerate(envelopes):
        if counts[a]:
            times[a, : counts[a]] = envelope.sample_ns
            values[a, : counts[a]] = envelope.samples
            peaks[a] = np.max(np.abs(envelope.samples))
            if counts[a] > 1:
                slopes[a] = np.max(
                    np.abs(np.diff(envelope.samples)) / np.diff(envelope.sample_ns)
                )
    gains = table.tone_amplitude[row, :count]
    args = (
        gains,
        table.tone_theta_rad[row, :count],
        table.tone_envelope_offset_ns[row, :count],
        times,
        values,
        counts,
    )
    return args, np.abs(gains) * peaks, np.abs(gains) * slopes


def _admission(
    frame: CommonFrame, duration: float, peaks, slopes
) -> CommonFrameAdmission:
    steps = max(1, int(np.ceil(duration / frame.step_ns)))
    if frame.cost_bypass:
        return CommonFrameAdmission(False, "cost", frame.step_ns, steps)
    channels = frame.channels
    if channels is None or channels.denominator_min <= 1e-10:
        return CommonFrameAdmission(False, "denominator", frame.step_ns, steps)
    signed_frequencies = np.repeat(frame.frequencies_ghz, 2) * np.tile(
        [1, -1], len(frame.ports)
    )
    model = frame.model
    gamma = float(
        np.max(model.rates_per_ns.sum(axis=0)) + np.max(model.dephasing_per_ns)
    )
    eta, second, third, fourth, derivative, bath = remainder_estimates(
        (
            model.energies_ghz - model.energies_ghz[0],
            signed_frequencies,
            frame.remainder,
            frame.monomials,
            frame.degrees,
            channels.generators,
        ),
        peaks,
        slopes,
        duration,
        gamma,
    )
    # Sum positive factorial terms by recurrence, avoiding separate powers or
    # factorials overflowing on strong rejected rows. Large eta rejects early.
    x = 2 * eta
    if not np.isfinite(x) or x > 100:
        tail = np.inf
    else:
        # Begin at order five, then bound all subsequent terms geometrically
        # after the ratio is below one. No fixed truncation loses the tail.
        order = 5
        slow_term, fast_term = x**4 / 24, x**4 / 30
        slow_tail, fast_tail = 0.0, 0.0
        while True:
            slow_tail += slow_term
            fast_tail += fast_term
            ratio = x / (order - 1)
            if ratio < 0.5 and max(slow_term, fast_term) < 1e-16:
                slow_tail += slow_term * ratio / (1 - ratio)
                fast_tail += fast_term * ratio / (1 - ratio)
                break
            slow_term *= x / order
            fast_term *= x * order / ((order - 1) * (order + 1))
            order += 1
        tail = float(
            2
            * np.pi
            * duration
            * np.sum(
                np.repeat(peaks, 2)
                * (frame.slow_norms * slow_tail + frame.fast_norms * fast_tail)
            )
        )
    coherent = second + third + fourth + tail + derivative
    reason = (
        "coherent"
        if not np.isfinite(coherent) or coherent > _COHERENT_LIMIT
        else "bath"
        if not np.isfinite(bath) or bath > _BATH_LIMIT
        else "accepted"
    )
    return CommonFrameAdmission(
        reason == "accepted",
        reason,
        frame.step_ns,
        steps,
        eta,
        second,
        third,
        fourth,
        tail,
        derivative,
        bath,
    )


def admit_common_frame(
    frame: CommonFrame, table: SegmentTable, row: int
) -> CommonFrameAdmission:
    """Assess one matching fixed-flux row without reading/evolving a density.

    frame must have been built for this row's model level, ports and carriers.
    Mismatches or continuous flux raise ValueError. Envelope peak/slope includes
    complex samples and held endpoints. Gains, duration and offsets remain row
    data; conservative full-envelope bounds also cover sliced pulses. Rejection
    never mutates the product or runs a fallback. Indicators are not rigorous
    certificates; see CommonFrameAdmission for their limitations.
    """
    _validate_row(frame, table, row)
    _, peaks, slopes = _row_envelopes(frame, table, row)
    return _admission(frame, float(table.duration_ns[row]), peaks, slopes)


def advance_common_frame(
    frame: CommonFrame, table: SegmentTable, row: int, state: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    """Advance a matching admitted row, returning a new secular-basis density.

    state is finite complex128 (M,M) in frame.model's basis at segment start.
    Inputs remain untouched. Reassesses current admission; rejected rows or
    invalid state shape/finiteness raise ValueError, never silently fall back.
    CF4 propagates explicit first/SW2 slow channels between exact secular bath
    half-steps. exp(S) endpoint dressing and the common-frame DC restoration
    preserve coherences. No clipping, renormalization or thermal weighting.
    """
    _validate_row(frame, table, row)
    size = len(frame.model.energies_ghz)
    if (
        state.shape != (size, size)
        or state.dtype != np.complex128
        or not np.isfinite(state).all()
    ):
        raise ValueError(
            "common-frame state must be a finite complex128 model-sized matrix"
        )
    _, peaks, slopes = _row_envelopes(frame, table, row)
    duration = float(table.duration_ns[row])
    admission = _admission(frame, duration, peaks, slopes)
    channels = frame.channels
    if not admission.accepted or channels is None:
        raise ValueError(f"common-frame row rejected: {admission.reason}")
    return advance_admitted_common_frame(frame, table, row, state, admission)


def advance_admitted_common_frame(
    frame: CommonFrame,
    table: SegmentTable,
    row: int,
    state: NDArray[np.complex128],
    admission: CommonFrameAdmission,
) -> NDArray[np.complex128]:
    """Apply a dispatch admission for this exact immutable table row and frame.

    The caller must supply the admission previously computed for these unchanged
    inputs. This entry avoids reevaluating indicators during production apply.
    Standalone callers should use advance_common_frame, which validates inputs
    and computes admission itself. A rejected admission raises ValueError.
    """
    channels = frame.channels
    if not admission.accepted or channels is None:
        raise ValueError(f"common-frame row rejected: {admission.reason}")
    args, _, _ = _row_envelopes(frame, table, row)
    duration = float(table.duration_ns[row])
    dt = duration / admission.steps
    generator, decay = secular_operators(frame.model)
    populations = expm(generator * dt / 2)
    coherences = np.exp(-decay * dt / 2)
    np.fill_diagonal(coherences, 1.0)

    def dress(time: float):
        z = amplitudes(time, args)
        generator = np.zeros_like(state)
        for a in range(len(z)):
            sign = 1 - 2 * (a % 2)
            frequency = (
                channels.frame[:, None]
                - channels.frame[None, :]
                + sign * frame.frequencies_ghz[a // 2]
            )
            generator += (
                channels.generators[a] * z[a] * np.exp(2j * np.pi * frequency * time)
            )
        return expm(generator)

    start, end = dress(0.0), dress(duration)
    result = propagate(
        channels,
        args,
        (duration, admission.steps),
        start @ state @ start.conj().T,
        (populations, coherences),
    )
    result = end.conj().T @ result @ end
    phase = np.exp(-2j * np.pi * channels.frame * duration)
    return result * phase[:, None] * phase.conj()[None, :]
