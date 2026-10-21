"""State-conditioned coherent fields before scattering and calibration."""

from dataclasses import dataclass, replace
from functools import lru_cache
from struct import pack
from sys import getsizeof

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

from ._constant import ConstantLine
from ._trajectories import PathFields, line_parameters
from .model import ReadoutModel, RealArray, readonly_array
from .request import PiecewiseLinearEnvelope, ReadoutPulse


def cavity_fields(
    model: ReadoutModel,
    pulses: tuple[ReadoutPulse, ...],
    times_ns: RealArray,
    *,
    initial_field: NDArray[np.complex128] | None = None,
    initial_time_ns: float = 0.0,
) -> NDArray[np.complex128]:
    """Return coherent intracavity alpha, shaped (time,state,cavity), in sqrt(photons).

    model sets joint-state order, resonant frequencies, reciprocal shared-line
    damping and the rotating reference. pulses are incident fields in that
    reference's pulse-local convention. Pulse contributions add coherently,
    including ringdown and overlapping/different-start supports. times_ns is a
    finite nonempty strictly increasing emitted-ns vector.
    initial_field is a finite complex (cavity,) field shared by all states, in
    sqrt(photons) in model's rotating reference at initial_time_ns. None means
    zero. initial_time_ns is finite nonnegative absolute emitted ns, default 0,
    in the same origin as pulses and times_ns. Only pulse history at/after that
    time propagates; pulse-local envelopes and phases are never restarted.
    Omitting both equals explicit vacuum at t=0, including mid-pulse queries.
    Empty pulses retain homogeneous memory decay. Invalid fields/times raise
    ValueError. A query before initial_time_ns names both times in ValueError.

    States remain fixed branches: directional population rates do not mix them.
    Receiver delay, gain, offset, noise and mismatch scattering phases do not
    change intracavity fields. Output owns its complex128 storage. Invalid
    coordinates or envelope values raise ValueError; solver failure raises
    RuntimeError. All-flat pulse tuples use singular-safe ConstantLine
    propagators. Shaped tuples use the field ODE with rtol=1e-9 and atol=1e-11.
    """
    times = readonly_array(times_ns, "cavity times")
    if times.ndim != 1 or not len(times) or np.any(np.diff(times) <= 0):
        raise ValueError("cavity times must be nonempty and strictly increasing")
    initial = _initial_field(initial_field, model.cavity_count)
    _validate_origin(initial_time_ns, float(times[0]))
    if pulses and all(pulse.envelope is None for pulse in pulses):
        return _flat_fields(model, pulses, times, initial, initial_time_ns)
    fields = _memory_fields(model, times, initial, initial_time_ns)
    for pulse in pulses:
        # Integrate near this carrier, then restore the caller's pulse-local
        # rotating coordinates. Absolute microwave frequencies need no ODE steps.
        carrier_model = replace(model, reference_frequency_ghz=pulse.frequency_ghz)
        path = PathFields(
            carrier_model,
            pulse,
            times,
            1e-9,
            1e-11,
            start_ns=initial_time_ns,
        )
        phase = np.exp(
            -2j
            * np.pi
            * (pulse.frequency_ghz - model.reference_frequency_ghz)
            * (times - pulse.start_ns)
        )
        fields += path.baseline(times) * phase[:, None, None]
    return fields


@dataclass(frozen=True)
class CavityFieldAverage:
    """Owned second moments in the ReadoutModel's fixed joint-state order.

    photons: (state,cavity) dimensionless time average of |alpha_s,r|².
    field_products: (state,cavity,state,cavity) complex128 average alpha_s,r conj(alpha_u,q),
        in photons. Flattening each (state,resonator) pair gives a Hermitian
        positive-semidefinite matrix, up to floating-point roundoff. photons is
        its real diagonal. Both include coherent carrier interference, not
        products of separately averaged fields. No population mixing occurs.
    """

    photons: RealArray
    field_products: NDArray[np.complex128]


class CavityAverageCache:
    """Byte-bounded owner of exact noiseless flat second moments.

    max_bytes: nonnegative integer resident limit, default 64 MiB, fixed for this
        owner. Zero disables admission. Oversized results still return normally;
        existing entries remain, with no eviction. Calls must be serialized.
    hits/misses: cumulative flat lookups found/not found, including failed solves
        after lookup. Invalid inputs, empty vacuum and shaped/callable inputs
        bypass lookups.
    resident_bytes: retained keys, whole owned array allocations and dictionary
        storage in bytes; zero after construction or clear(). Active calculation
        workspaces and caller-owned return arrays are outside this limit.

    Keys retain complete converted model/pulse content, exact coordinates and
    complex initial fields, not identities or rounded values. Only successful
    flat solves are admitted. Hits and misses return independent writable arrays.
    Constructor raises ValueError for a noninteger or negative capacity. Solver
    errors propagate without admission. This owner retains no model or callback.
    """

    def __init__(self, max_bytes: int = 64 * 1024**2) -> None:
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("cavity cache max_bytes must be a nonnegative integer")
        self._max_bytes = max_bytes
        self.hits: int = 0
        self.misses: int = 0
        self.resident_bytes: int = 0
        self._products: dict[bytes, NDArray[np.complex128]] = {}
        self._payload_bytes = 0

    @property
    def max_bytes(self) -> int:
        """Fixed nonnegative resident capacity in bytes; zero disables admission."""
        return self._max_bytes

    def clear(self) -> None:
        """Release all resident content and storage, preserving hits/misses."""
        self._products = {}
        self._payload_bytes = 0
        self.resident_bytes = 0

    def average(
        self,
        model: ReadoutModel,
        pulses: tuple[ReadoutPulse, ...],
        start_ns: float,
        end_ns: float,
        *,
        initial_field: NDArray[np.complex128] | None = None,
        initial_time_ns: float = 0.0,
    ) -> CavityFieldAverage:
        """Return owned second moments with exact flat-content reuse.

        model fixes joint-state order and ns/GHz units. pulses are ordered
        coherent ReadoutPulse inputs, with absolute supports and local phases.
        Finite absolute emitted endpoints require end_ns > start_ns >=
        initial_time_ns. initial_field is a finite complex (2,) shared field in
        sqrt(photons) in model's reference at initial_time_ns (finite nonnegative
        ns). None means vacuum; the default origin is zero. History, cross-state
        and cross-carrier products and ringdown follow cavity_field_average.
        Shaped/callable inputs bypass reuse and retain its quadrature tolerances.
        Invalid inputs raise ValueError; solver/convergence failures propagate
        without admission. Capacity exhaustion still returns the computed result.
        Result arrays are writable independent copies on hits and misses.
        """
        if not np.isfinite([start_ns, end_ns]).all() or end_ns <= start_ns:
            raise ValueError("cavity average interval must be finite and positive")
        initial = _initial_field(initial_field)
        _validate_origin(initial_time_ns, start_ns)
        if any(p.envelope is not None for p in pulses) or not (
            pulses or np.any(initial)
        ):
            return cavity_field_average(
                model,
                pulses,
                start_ns,
                end_ns,
                initial_field=initial,
                initial_time_ns=initial_time_ns,
            )
        near, converted, memory = _near_carrier(model, pulses, initial, initial_time_ns)
        key = _flat_key(near, converted, start_ns, end_ns, memory, initial_time_ns)
        products = self._products.get(key)
        if products is not None:
            self.hits += 1
            products = products.copy()
            n = len(model.state_labels)
            photons = (
                products.reshape(2 * n, 2 * n).diagonal().real.reshape(n, 2).copy()
            )
            return CavityFieldAverage(photons, products)
        self.misses += 1
        result = cavity_field_average(
            model,
            pulses,
            start_ns,
            end_ns,
            initial_field=initial,
            initial_time_ns=initial_time_ns,
        )
        owned = result.field_products.copy()
        candidate = self._products | {key: owned}
        payload = self._payload_bytes + getsizeof(key) + getsizeof(owned)
        resident = payload + getsizeof(candidate)
        if resident <= self.max_bytes:
            self._products = candidate
            self._payload_bytes = payload
            self.resident_bytes = resident
        return result


def cavity_field_average(
    model: ReadoutModel,
    pulses: tuple[ReadoutPulse, ...],
    start_ns: float,
    end_ns: float,
    *,
    initial_field: NDArray[np.complex128] | None = None,
    initial_time_ns: float = 0.0,
) -> CavityFieldAverage:
    """Average intracavity second moments over finite end_ns > start_ns.

    Endpoints are absolute emitted ns, before receiver delay. Pulse history from
    initial_time_ns, including ringdown, is retained. Pulse-local phase is the
    ReadoutPulse convention used by cavity_fields: amplitude*envelope(local)*
    exp(-2*pi*i*(carrier-reference)*local+i*phase). Products use the same-time
    alpha times conjugate-alpha convention and are invariant under a common
    rotating frame. No calibration, scattering mismatch or population-rate
    mixing is applied. initial_field is a finite complex (cavity,) field shared by
    all states at initial_time_ns in sqrt(photons), in model's reference. None
    means zero. initial_time_ns is finite nonnegative absolute emitted ns in the
    same origin as pulses/start/end, default 0. Only subsequent pulse history
    propagates; pulse-local envelopes and phases are never restarted. Omitting
    both equals explicit vacuum at t=0 even for a mid-pulse interval. Empty pulses
    propagate memory. Invalid fields/times raise ValueError. start_ns before
    initial_time_ns raises ValueError naming both times. Inputs are unchanged.

    Flat contents use ConstantLine propagators and singular-safe analytic
    quadratic integrals, including cross-carrier terms. Shaped contents use
    vacuum-field ODEs (rtol=1e-9, atol=1e-11) and quadrature split at pulse/corner
    boundaries and 1/kappa, converging at rtol=1e-7, atol=1e-13. Invalid intervals
    or envelope values raise ValueError; nonconvergence/nonfinite propagation
    raises RuntimeError. Result arrays own their storage and have no time axis.
    """
    if not np.isfinite([start_ns, end_ns]).all() or end_ns <= start_ns:
        raise ValueError("cavity average interval must be finite and positive")
    c = model.cavity_count
    initial = _initial_field(initial_field, c)
    _validate_origin(initial_time_ns, start_ns)
    n = len(model.state_labels)
    products = np.zeros((n, c, n, c), dtype=np.complex128)
    if pulses or np.any(initial):
        near, pulses, initial = _near_carrier(model, pulses, initial, initial_time_ns)
        if all(pulse.envelope is None for pulse in pulses):
            products = _flat_average(
                near, pulses, start_ns, end_ns, initial, initial_time_ns
            )
        else:
            products = _shaped_average(
                near, pulses, start_ns, end_ns, initial, initial_time_ns
            )
    photons = products.reshape(c * n, c * n).diagonal().real.reshape(n, c).copy()
    return CavityFieldAverage(photons, products)


def _near_carrier(
    model: ReadoutModel,
    pulses: tuple[ReadoutPulse, ...],
    initial: NDArray[np.complex128],
    origin: float,
) -> tuple[ReadoutModel, tuple[ReadoutPulse, ...], NDArray[np.complex128]]:
    # Pulse-local phases need +2*pi*delta_f*start so every field rotates together.
    near = replace(
        model,
        reference_frequency_ghz=(
            pulses[0].frequency_ghz if pulses else model.reference_frequency_ghz
        ),
    )
    delta_f = near.reference_frequency_ghz - model.reference_frequency_ghz
    converted = tuple(
        replace(pulse, phase_rad=pulse.phase_rad + 2 * np.pi * delta_f * pulse.start_ns)
        for pulse in pulses
    )
    return near, converted, initial * np.exp(2j * np.pi * delta_f * origin)


def _flat_key(
    model: ReadoutModel,
    pulses: tuple[ReadoutPulse, ...],
    start: float,
    end: float,
    initial: NDArray[np.complex128],
    origin: float,
) -> bytes:
    # Fixed schema for the converted flat solver inputs, including signed zeros.
    parts = [
        b"cavity-flat-average-v1\0",
        pack("!QQ", len(model.state_labels), len(pulses)),
    ]
    for text in (model.identity, *model.state_labels):
        encoded = text.encode("utf-8")
        parts.extend((pack("!Q", len(encoded)), encoded))
    receiver = model.receiver
    parts.append(
        pack(
            "!7d",
            model.reference_frequency_ghz,
            receiver.gain,
            receiver.phase_rad,
            receiver.time_of_flight_ns,
            start,
            end,
            origin,
        )
    )
    for array in (
        model.frequencies_ghz,
        model.kappa_internal_per_ns,
        model.kappa_external_per_ns,
        model.rates_per_ns,
        model.mismatch_phase_rad,
        receiver.offset,
        receiver.noise_covariance,
        initial,
    ):
        dtype = array.dtype.str.encode("ascii")
        parts.extend(
            (
                pack("!QQ", len(dtype), array.ndim),
                dtype,
                pack(f"!{array.ndim}Q", *array.shape),
                array.tobytes(),
            )
        )
    for pulse in pulses:
        parts.append(
            pack(
                "!6dQ",
                pulse.start_ns,
                pulse.end_ns,
                pulse.amplitude.real,
                pulse.amplitude.imag,
                pulse.frequency_ghz,
                pulse.phase_rad,
                len(pulse.breakpoints_ns),
            )
        )
        parts.append(pack(f"!{len(pulse.breakpoints_ns)}d", *pulse.breakpoints_ns))
    return b"".join(parts)


def _initial_field(
    value: NDArray[np.complex128] | None,
    cavities: int,
) -> NDArray[np.complex128]:
    if value is None:
        return np.zeros(cavities, dtype=np.complex128)
    field = np.array(value, dtype=np.complex128, copy=True)
    if field.shape != (cavities,) or not np.isfinite(field).all():
        raise ValueError(f"initial field must be a finite complex ({cavities},) vector")
    return field


def _validate_origin(origin: float, query: float) -> None:
    if not np.isfinite(origin) or origin < 0:
        raise ValueError("initial time must be finite and nonnegative")
    if query < origin:
        raise ValueError(f"query time {query} precedes initial time {origin}")


def _memory_fields(
    model: ReadoutModel,
    times: RealArray,
    initial: NDArray[np.complex128],
    origin: float,
) -> NDArray[np.complex128]:
    if not np.any(initial):
        return np.zeros(
            (len(times), len(model.state_labels), model.cavity_count),
            dtype=np.complex128,
        )
    a, _ = line_parameters(model)
    propagators = expm(-(times - origin)[:, None, None, None] * a[None])
    return np.einsum("tsru,u->tsr", propagators, initial)


def _boundaries(
    pulses: tuple[ReadoutPulse, ...], start: float, end: float, kappa: float = 0.0
) -> RealArray:
    corners = {start, end} | {
        float(t)
        for pulse in pulses
        for t in (
            pulse.start_ns,
            pulse.end_ns,
            *pulse.breakpoints_ns,
            *(
                pulse.start_ns + pulse.envelope.knots_ns
                if isinstance(pulse.envelope, PiecewiseLinearEnvelope)
                else ()
            ),
        )
        if start < t < end
    }
    corners = sorted(corners)
    points = [start]
    for left, right in zip(corners[:-1], corners[1:], strict=True):
        count = max(1, int(np.ceil((right - left) * kappa)))
        points.extend(np.linspace(left, right, count + 1)[1:])
    return np.array(points)


def _flat_fields(
    model: ReadoutModel,
    pulses: tuple[ReadoutPulse, ...],
    times: RealArray,
    initial: NDArray[np.complex128],
    origin: float,
) -> NDArray[np.complex128]:
    # Work near the carrier to avoid exponentiating the absolute GHz rotation.
    near = replace(model, reference_frequency_ghz=pulses[0].frequency_ghz)
    delta = near.reference_frequency_ghz - model.reference_frequency_ghz
    pulses = tuple(
        replace(pulse, phase_rad=pulse.phase_rad + 2 * np.pi * delta * pulse.start_ns)
        for pulse in pulses
    )
    a, v = line_parameters(near)
    n = len(model.state_labels)
    line = ConstantLine(np.zeros((n, n)), a, v, v.astype(np.complex128))
    boundaries = np.unique(
        np.concatenate((_boundaries(pulses, origin, float(times[-1])), times))
    )
    fields = np.tile(initial * np.exp(2j * np.pi * delta * origin), (n, 1))
    output = np.empty((len(times), n, model.cavity_count), dtype=np.complex128)
    cursor = 0
    if times[0] == origin:
        output[0] = fields
        cursor = 1
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
        active = tuple(
            pulse for pulse in pulses if pulse.start_ns <= left < pulse.end_ns
        )
        incidents = np.array(
            [pulse.incident(left, near.reference_frequency_ghz) for pulse in active],
            dtype=np.complex128,
        )
        omegas = np.array(
            [
                2 * np.pi * (pulse.frequency_ghz - near.reference_frequency_ghz)
                for pulse in active
            ]
        )
        fields = line.coherent_step(fields, incidents, omegas, float(right - left))
        if cursor < len(times) and right == times[cursor]:
            output[cursor] = fields
            cursor += 1
    output *= np.exp(-2j * np.pi * delta * times)[:, None, None]
    return output


def _flat_average(
    model: ReadoutModel,
    pulses: tuple[ReadoutPulse, ...],
    start: float,
    end: float,
    initial: NDArray[np.complex128],
    origin: float,
) -> NDArray[np.complex128]:
    n = len(model.state_labels)
    a, v = line_parameters(model)
    line = ConstantLine(np.zeros((n, n)), a, v, v.astype(np.complex128))
    boundaries = _boundaries(pulses, origin, end)
    boundaries = np.unique(np.append(boundaries, start))
    fields = np.tile(initial, (n, 1))
    c = model.cavity_count
    products = np.zeros((n, c, n, c), dtype=np.complex128)
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
        active = tuple(
            pulse for pulse in pulses if pulse.start_ns <= left < pulse.end_ns
        )
        incidents = np.array(
            [pulse.incident(left, model.reference_frequency_ghz) for pulse in active],
            dtype=np.complex128,
        )
        omegas = np.array(
            [
                2 * np.pi * (pulse.frequency_ghz - model.reference_frequency_ghz)
                for pulse in active
            ]
        )
        fields, integral = line.coherent_integral(
            fields, incidents, omegas, float(right - left)
        )
        if left >= start:
            products += integral
    return products / (end - start)


def _shaped_average(
    model: ReadoutModel,
    pulses: tuple[ReadoutPulse, ...],
    start: float,
    end: float,
    initial: NDArray[np.complex128],
    origin: float,
) -> NDArray[np.complex128]:
    # Prepare each dense baseline once, not once per quadrature refinement.
    paths = [
        PathFields(
            replace(model, reference_frequency_ghz=pulse.frequency_ghz),
            pulse,
            np.array([end]),
            1e-9,
            1e-11,
            start_ns=origin,
        )
        for pulse in pulses
    ]
    kappa = float(np.max(model.kappa_internal_per_ns + model.kappa_external_per_ns))
    boundaries = _boundaries(pulses, start, end, kappa)
    half = np.diff(boundaries) / 2
    mid = (boundaries[1:] + boundaries[:-1]) / 2
    previous = None
    for order in (16, 32, 64, 128, 256, 512, 1024):
        nodes, weights = _gauss_rule(order)
        times = (mid[:, None] + half[:, None] * nodes).ravel()
        scaled = (half[:, None] * weights).ravel() / (end - start)
        fields = _memory_fields(model, times, initial, origin)
        for pulse, path in zip(pulses, paths, strict=True):
            phase = np.exp(
                -2j
                * np.pi
                * (pulse.frequency_ghz - model.reference_frequency_ghz)
                * (times - pulse.start_ns)
            )
            fields += path.baseline(times) * phase[:, None, None]
        products = np.einsum("t,tsr,tuq->sruq", scaled, fields, fields.conj())
        if previous is not None and np.allclose(
            products, previous, rtol=1e-7, atol=1e-13
        ):
            return products
        previous = products
    raise RuntimeError("cavity segment average did not converge")


@lru_cache(maxsize=7)
def _gauss_rule(order: int) -> tuple[RealArray, RealArray]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights
