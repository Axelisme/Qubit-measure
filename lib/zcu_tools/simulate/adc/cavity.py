"""State-conditioned coherent fields before scattering and calibration."""

from dataclasses import dataclass, replace
from functools import lru_cache

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
        # Pulse-local phases need +2*pi*delta_f*start so the reference change
        # rotates every field by the same exp(+2*pi*i*delta_f*t).
        near = replace(
            model,
            reference_frequency_ghz=(
                pulses[0].frequency_ghz if pulses else model.reference_frequency_ghz
            ),
        )
        delta_f = near.reference_frequency_ghz - model.reference_frequency_ghz
        pulses = tuple(
            replace(
                pulse,
                phase_rad=pulse.phase_rad + 2 * np.pi * delta_f * pulse.start_ns,
            )
            for pulse in pulses
        )
        initial = initial * np.exp(2j * np.pi * delta_f * initial_time_ns)
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
