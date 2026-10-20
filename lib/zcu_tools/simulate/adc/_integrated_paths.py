"""Exact discrete path sums; noiseless baselines are reusable, never shot LUTs."""

from typing import NamedTuple

import numpy as np
from numba import njit
from numpy.typing import NDArray

from ._jump_paths import JumpPaths
from ._trajectories import PathFields
from .model import ReadoutModel, RealArray
from .request import ReadoutPulse, SampleGrid


class _Wave(NamedTuple):
    """Mutable workspace owned by one materialization, never a prepare cache.

    states: int64 (shot,) current model-state indices. left: float64 (shot,)
    current segment starts in emitted ns. memory: complex128 (shot,2) cavity
    amplitudes relative to that state's vacuum baseline at left. Kernels update
    these arrays in place; the tuple only groups the coherent path state.
    """

    states: NDArray[np.int64]
    left: RealArray
    memory: NDArray[np.complex128]


class _GridSums(NamedTuple):
    """Noiseless one-grid inputs, owned by the per-content path workspace.

    times: float64 (sample,) increasing emitted ns. oscillator: complex128
    (sample,) ADC demodulation including the reference-frame delay phase.
    prefix: complex128 (sample+1,state) cumulative weighted vacuum output.
    samples: complex128 (sample,state) weighted vacuum output at each actual
    sample, retained separately so overlapping trace windows agree bitwise.
    powers/sums: complex128 (state,bit,2,2) binary homogeneous step powers and
    discrete sums. uniform: True only for an exactly uniform stored grid.
    These arrays contain no sampled paths, noisy IQ or jump-time lookup tables.
    """

    times: RealArray
    oscillator: NDArray[np.complex128]
    prefix: NDArray[np.complex128]
    samples: NDArray[np.complex128]
    powers: NDArray[np.complex128]
    sums: NDArray[np.complex128]
    uniform: bool


@njit(nogil=True, cache=False)
def _exp2(matrix, duration):
    """Two-dimensional exponential, including coincident/defective modes."""
    mean = (matrix[0, 0] + matrix[1, 1]) / 2
    reduced = matrix - mean * np.eye(2, dtype=np.complex128)
    root = np.sqrt(reduced[0, 0] ** 2 + reduced[0, 1] * reduced[1, 0])
    z = root * duration
    if abs(z) < 1e-5:
        scale = np.exp(mean * duration)
        cosine = scale * (1 + z * z / 2 + z**4 / 24)
        sine = scale * duration * (1 + z * z / 6 + z**4 / 120)
    else:
        plus = np.exp((mean + root) * duration)
        minus = np.exp((mean - root) * duration)
        cosine = (plus + minus) / 2
        sine = (plus - minus) / (2 * root)
    return cosine * np.eye(2, dtype=np.complex128) + sine * reduced


@njit(nogil=True, cache=False)
def _integral2(matrix, duration):
    """Scaling/squaring phi1 without a spectral division or inverse guard."""
    size = np.max(np.abs(matrix)) * duration * 2
    scale = max(0, int(np.ceil(np.log2(size / 0.5)))) if size > 0.5 else 0
    step = duration / 2.0**scale
    z = matrix * step
    term = np.eye(2, dtype=np.complex128)
    exponential = term.copy()
    integral = term.copy()
    for order in range(1, 25):
        term = term @ z / order
        exponential += term
        integral += term / (order + 1)
    integral *= step
    for _ in range(scale):
        integral += exponential @ integral
        exponential = exponential @ exponential
    return integral


@njit(nogil=True, cache=False)
def _flat_baselines(a, v, pulse, coupling, safe, times):
    start, stop, incident, omega = pulse
    fields = np.zeros((len(times), len(a), 2), dtype=np.complex128)
    for index, time in enumerate(times):
        if time <= start:
            continue
        driven = min(time, stop) - start
        for state in range(len(a)):
            if safe[state]:
                field = incident * (
                    np.exp(-1j * omega * driven) * coupling[state]
                    - _exp2(-a[state], driven) @ coupling[state]
                )
            else:
                shifted = -a[state] + 1j * omega * np.eye(2, dtype=np.complex128)
                field = (_integral2(shifted, driven) @ v) * incident
                field *= np.exp(-1j * omega * driven)
            if time > stop:
                field = _exp2(-a[state], time - stop) @ field
            fields[index, state] = field
    return fields


@njit(nogil=True, cache=False)
def _segment_sums(a, scattering, grid, wave, shots, right):
    times, oscillator, prefix, _, powers, sums, uniform = grid
    states, left, memory = wave
    output = np.zeros(len(shots), dtype=np.complex128)
    for index, shot in enumerate(shots):
        state = states[shot]
        first = np.searchsorted(times, left[shot])
        last = np.searchsorted(times, right[index])
        output[index] = prefix[last, state] - prefix[first, state]
        if first == last or not np.any(memory[shot]):
            continue
        if uniform:
            vector = _exp2(-a[state], times[first] - left[shot]) @ memory[shot]
            count = last - first
            total = np.zeros(2, dtype=np.complex128)
            bit = 0
            while count:
                if count & 1:
                    total += sums[state, bit] @ vector
                    vector = powers[state, bit] @ vector
                bit += 1
                count >>= 1
            output[index] -= oscillator[first] * (scattering @ total)
        else:
            for sample in range(first, last):
                field = _exp2(-a[state], times[sample] - left[shot]) @ memory[shot]
                output[index] -= oscillator[sample] * (scattering @ field)
    return output


@njit(nogil=True, cache=False)
def _segment_traces(a, scattering, grid, wave, step, output):
    shots, right, _ = step
    times, oscillator, _, samples, _, _, _ = grid
    states, left, memory = wave
    for index, shot in enumerate(shots):
        state = states[shot]
        first = np.searchsorted(times, left[shot])
        last = np.searchsorted(times, right[index])
        # Lookback samples have zero physical field; they remain zero until
        # receiver offset and noise are added after all segments.
        for sample in range(first, last):
            output[shot, sample] = samples[sample, state]
            if np.any(memory[shot]):
                field = _exp2(-a[state], times[sample] - left[shot]) @ memory[shot]
                output[shot, sample] -= oscillator[sample] * (scattering @ field)


@njit(nogil=True, cache=False)
def _jump_memory(a, shots, targets, jumps, baselines, wave):
    states, left, memory = wave
    for index, shot in enumerate(shots):
        state = states[shot]
        memory[shot] = (
            _exp2(-a[state], jumps[index] - left[shot]) @ memory[shot]
            + baselines[index, state]
            - baselines[index, targets[index]]
        )


class IntegratedPaths:
    """Noiseless coherent multi-carrier sums/traces with exact CTMC memory.

    Internal workspace for one content; model/pulses/grids are validated ADC
    records, rtol/atol are the existing deterministic baseline solver tolerances.
    Flat vacuum fields use compiled phi1 propagation, including singular driven
    blocks. Other envelopes use one DOP853 dense vacuum solve for all states,
    never an ODE per shot or an artificial jump-time interpolation grid. Prefix
    sums contain only the actual ADC sample coordinates. Uniform grids use exact
    binary geometric sums, irregular grids sum exact homogeneous exponentials.

    model/pulses/grids retain the constructor inputs for this workspace's lifetime.
    v/scattering are complex128 (2,) drive/output couplings; a is (state,2,2)
    complex128 damping/detuning. times holds each grid's emitted-ns vector;
    start/end are the earliest pulse start and latest emitted time, with end
    no earlier than start. path_fields holds a shaped vacuum solver or None
    per carrier. couplings holds each guarded flat driven solution (state,2)
    and its (state,) acceptance mask. plans holds one noiseless _GridSums per
    read. No shot state survives evaluate.
    Solver failures raise RuntimeError. Invalid envelopes and unresolved
    floating-point transition times raise ValueError without partial results.
    """

    def __init__(
        self,
        model: ReadoutModel,
        pulses: tuple[ReadoutPulse, ...],
        grids: tuple[SampleGrid, ...],
        rtol: float,
        atol: float,
    ) -> None:
        self.model, self.pulses, self.grids = model, pulses, grids
        self.start = min(pulse.start_ns for pulse in pulses)
        self.v = np.asarray(
            np.sqrt(model.kappa_external_per_ns / 2), dtype=np.complex128
        )
        self.scattering = self.v * np.exp(-1j * model.mismatch_phase_rad)
        self.a = np.array(
            [
                np.diag(model.kappa_internal_per_ns / 2 + 2j * np.pi * detuning)
                + np.outer(self.v, self.v)
                for detuning in model.frequencies_ghz - model.reference_frequency_ghz
            ]
        )
        delay = model.receiver.time_of_flight_ns
        self.times = tuple(grid.times_ns - delay for grid in grids)
        all_times = np.unique(np.concatenate(self.times))
        self.end = max(self.start, float(all_times[-1]))
        self.path_fields = tuple(
            PathFields(model, pulse, all_times, rtol, atol)
            if pulse.envelope is not None
            else None
            for pulse in pulses
        )
        self.couplings = tuple(self._flat_coupling(pulse) for pulse in pulses)
        self.plans = []
        delay_phase = np.exp(2j * np.pi * model.reference_frequency_ghz * delay)
        for grid, times in zip(grids, self.times, strict=True):
            baseline = self._baseline(times)
            incident = np.array(
                [
                    sum(
                        pulse.incident(float(t), model.reference_frequency_ghz)
                        for pulse in pulses
                    )
                    for t in times
                ]
            )
            outgoing = incident[:, None] - baseline @ self.scattering
            omega = (
                2
                * np.pi
                * (grid.demodulation_frequency_ghz - model.reference_frequency_ghz)
            )
            oscillator = (
                np.exp(1j * omega * grid.times_ns - 1j * grid.phase_rad) * delay_phase
            )
            prefix = np.concatenate(
                (
                    np.zeros((1, len(self.a)), complex),
                    np.cumsum(outgoing * oscillator[:, None], axis=0),
                )
            )
            # No tolerance-based resampling: only exactly uniform stored grids
            # use the geometric route. All others keep their original times.
            uniform = len(times) == 1 or np.all(np.diff(times) == times[1] - times[0])
            bits = max(1, len(times).bit_length())
            powers = np.empty((len(self.a), bits, 2, 2), complex)
            sums = np.empty_like(powers)
            dt = float(times[1] - times[0]) if len(times) > 1 else 0.0
            for state, a in enumerate(self.a):
                powers[state, 0] = _exp2(-a + 1j * omega * np.eye(2), dt)
                sums[state, 0] = np.eye(2)
                for bit in range(1, bits):
                    previous = powers[state, bit - 1]
                    sums[state, bit] = (
                        sums[state, bit - 1] + previous @ sums[state, bit - 1]
                    )
                    powers[state, bit] = previous @ previous
            self.plans.append(
                _GridSums(
                    times,
                    oscillator,
                    prefix,
                    outgoing * oscillator[:, None],
                    powers,
                    sums,
                    bool(uniform),
                )
            )

    def _flat_coupling(self, pulse):
        """Use a guarded driven solve; singular/cancellation cases keep phi1."""
        omega = 2 * np.pi * (pulse.frequency_ghz - self.model.reference_frequency_ghz)
        coupling = np.zeros((len(self.a), 2), dtype=np.complex128)
        safe = np.zeros(len(self.a), dtype=np.bool_)
        for state, a in enumerate(self.a):
            shifted = a - 1j * omega * np.eye(2)
            if np.linalg.cond(shifted) >= 1e4:
                continue
            value = np.linalg.solve(shifted, self.v)
            if (
                np.isfinite(value).all()
                and np.finfo(float).eps * np.linalg.norm(value, ord=np.inf) <= 1e-13
                and np.linalg.norm(shifted @ value - self.v, ord=np.inf) <= 1e-13
            ):
                coupling[state], safe[state] = value, True
        return coupling, safe

    def _baseline(self, times):
        fields = np.zeros((len(times), len(self.a), 2), dtype=np.complex128)
        for pulse, path_fields, (coupling, safe) in zip(
            self.pulses, self.path_fields, self.couplings, strict=True
        ):
            if path_fields is not None:
                fields += path_fields.baseline(times)
            else:
                omega = (
                    2
                    * np.pi
                    * (pulse.frequency_ghz - self.model.reference_frequency_ghz)
                )
                fields += _flat_baselines(
                    self.a,
                    self.v,
                    (
                        pulse.start_ns,
                        pulse.end_ns,
                        pulse.amplitude * np.exp(1j * pulse.phase_rad),
                        omega,
                    ),
                    coupling,
                    safe,
                    times,
                )
        return fields

    def evaluate(
        self, paths: JumpPaths, *, traces: bool = False
    ) -> tuple[RealArray, ...]:
        """Observe a shared path without RNG draws, returning one array per read.

        paths covers this workspace's pulse origin and latest emitted sample.
        traces=False returns (shot,IQ) discrete sums per grid; True returns
        (shot,sample,IQ) traces. Calibration/offset occur once after coherent
        carrier superposition. The caller owns receiver noise and rounding.
        """
        states = paths.initial.copy()
        left = np.full(len(states), paths.start, dtype=np.float64)
        memory = np.zeros((len(states), 2), complex)
        wave = _Wave(states, left, memory)
        output = [
            np.zeros(
                (len(states), len(grid.times_ns)) if traces else (len(states),), complex
            )
            for grid in self.grids
        ]
        for step in paths.steps:
            active, right, targets = step
            for read, grid in enumerate(self.plans):
                if traces:
                    _segment_traces(
                        self.a, self.scattering, grid, wave, step, output[read]
                    )
                else:
                    output[read][active] += _segment_sums(
                        self.a, self.scattering, grid, wave, active, right
                    )
            moving = targets >= 0
            shots, jumps, destinations = active[moving], right[moving], targets[moving]
            if len(shots):
                baselines = self._baseline(jumps)
                _jump_memory(self.a, shots, destinations, jumps, baselines, wave)
                states[shots], left[shots] = destinations, jumps
        receiver = self.model.receiver
        result = []
        for grid, outgoing in zip(self.grids, output, strict=True):
            if not np.isfinite(outgoing).all():
                raise RuntimeError("readout paths produced nonfinite fields")
            calibrated = (
                np.conj(outgoing) * receiver.gain * np.exp(1j * receiver.phase_rad)
            )
            iq = np.stack((calibrated.real, calibrated.imag), axis=-1)
            iq += (1 if traces else len(grid.times_ns)) * receiver.offset
            result.append(iq)
        return tuple(result)
