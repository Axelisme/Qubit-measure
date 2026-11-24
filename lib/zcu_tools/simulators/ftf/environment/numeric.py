"""FTF numeric input translation and evolution budget policy."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from zcu_tools.simulate.boundary import EnvironmentSnapshot
from zcu_tools.simulate.evolution import DriveSource, TimelineBatch, segment_spans
from zcu_tools.simulators.ftf.config import FTFSettings
from zcu_tools.simulators.ftf.config.schema import Fluxonium, FluxoniumTruncation
from zcu_tools.simulators.ftf.config.schema import Resonator as ResourceResonator
from zcu_tools.simulators.ftf.physics import (
    FluxoniumCircuit,
    JointModel,
    JointParameters,
    Resonator,
    SquidCircuit,
)

_MIN_EVOLUTION_BUDGET = 1e-3
SINGLE_SPAN_BUDGET = 1e-2


@dataclass(frozen=True)
class EvolutionReuse:
    """One production evolve's content digest and per-segment budget for reuse.

    key is the original caller-owned reuse_key. budget is the same positive
    absolute allowance passed to that evolve, not a fresh budget for the acquire.
    """

    key: bytes
    budget: float


def evolution_budget(batches: Sequence[TimelineBatch]) -> float:
    """Return the clamped per-segment allowance for the supplied evolve inputs.

    Count positive-length spans from each point's continuation start (or time
    zero), using the public Engine partition (ADR-0078). Divide 1e-2 by the
    largest point count, capped to [1e-3, 1e-2]. Empty work gets 1e-2.
    """
    longest = 0
    for batch in batches:
        spans = segment_spans(batch)
        if batch.start_ns is None:
            count = max(map(len, spans), default=0)
        else:
            count = max(
                (
                    sum(1 for left, _ in point_spans if left >= start)
                    for point_spans, start in zip(spans, batch.start_ns, strict=True)
                ),
                default=0,
            )
        longest = max(longest, count)
    return max(_MIN_EVOLUTION_BUDGET, SINGLE_SPAN_BUDGET / max(1, longest))


def circuit(value: Fluxonium, truncation: FluxoniumTruncation) -> FluxoniumCircuit:
    """Translate named fluxonium GHz/truncation/noise values without rescaling."""
    return FluxoniumCircuit(
        value.ej_ghz,
        value.ec_ghz,
        value.el_ghz,
        truncation.native_cutoff,
        truncation.levels,
        value.q_cap,
        value.q_ind,
        value.xqp,
        value.aphi,
    )


def joint_parameters(settings: FTFSettings) -> JointParameters:
    """Return pure-numeric F1/C/F2 parameters; no resource type enters simulate."""
    m = settings.chip
    t = settings.numerics.truncation
    c = m.coupler
    coupler = SquidCircuit(
        c.ejmax_ghz,
        c.ec_ghz,
        c.asymmetry,
        t.coupler.charge_cutoff,
        t.coupler.levels,
        c.ng,
        c.q_cap,
        c.resistance_ohm,
        c.flux_resistance_ohm,
        c.mutual_phi0_per_a,
        c.aphi,
    )
    return JointParameters(
        circuit(m.f1, t.f1),
        coupler,
        circuit(m.f2, t.f2),
        m.g_f1_coupler_ghz,
        m.g_f2_coupler_ghz,
        m.g_f1_f2_ghz,
        t.retained_levels,
    )


def resonators(settings: FTFSettings) -> tuple[Resonator, Resonator]:
    """Convert kappa/(2*pi) MHz into photon decay 1/ns exactly once."""

    def convert(r: ResourceResonator) -> Resonator:
        return Resonator(
            r.frequency_ghz,
            r.charge_coefficients_ghz.as_tuple(),
            2 * np.pi * r.kappa_external_mhz / 1000,
            2 * np.pi * r.kappa_internal_mhz / 1000,
            r.thermal_occupation,
        )

    pair = settings.readout.resonators
    return convert(pair.r1), convert(pair.r2)


def dc_flux(
    settings: FTFSettings, snapshot: EnvironmentSnapshot
) -> NDArray[np.float64]:
    """Map validated live round values to reduced F1/C/F2 flux; reject missing/invalid values."""
    values = dict(snapshot.values)
    result = np.array(settings.dc_bias.offset_phi0.as_tuple())
    for name, source in settings.dc_bias.sources.items():
        if (
            name not in values
            or not np.isfinite(values[name])
            or not source.limits[0] <= values[name] <= source.limits[1]
        ):
            raise ValueError(
                f"dc_bias.sources.{name}: missing/nonfinite/out-of-limits round value"
            )
        result += np.array(source.flux_per_unit.as_tuple()) * values[name]
    return result


def drive_sources(settings: FTFSettings, joint: JointModel) -> tuple[DriveSource, ...]:
    """Return named charge/flux quadratures in joint's fixed DC basis, GHz/gain.

    Charge coefficients (GHz/gain) multiply full n. Flux coefficients (Phi0/gain)
    multiply the analytic d(H/h)/df at DC (GHz/Phi0), not a dynamic H(Phi(t)).
    Both use Re[c*z] = Re(c)*Re(z) - Im(c)*Im(z), including signed crosstalk.
    """
    zero = np.zeros_like(joint.hamiltonian(joint.dc_flux))
    sources = []
    for name, source in settings.drive_lines.items():
        coefficients = tuple(
            complex(c.real, c.imag) for c in source.coefficients().as_tuple()
        )
        operator = (
            joint.charge_operator
            if source.operator == "charge"
            else joint.flux_operator
        )
        # Re[c*z] = Re(c)*Re(z)-Im(c)*Im(z); both operators are Hermitian.
        x = sum(
            (c.real * operator(i) for i, c in enumerate(coefficients)),
            zero.copy(),
        )
        y = sum(
            (-c.imag * operator(i) for i, c in enumerate(coefficients)),
            zero.copy(),
        )
        sources.append(DriveSource(name, x, y))
    return tuple(sources)
