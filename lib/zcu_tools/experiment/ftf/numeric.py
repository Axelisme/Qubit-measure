"""Named resource-to-numeric translation; no YAML or solver internals."""

import numpy as np
from numpy.typing import NDArray

from zcu_tools.core.simulation import EnvironmentSnapshot
from zcu_tools.resources.ftf import FTFSettings
from zcu_tools.resources.ftf.schema import Fluxonium, FluxoniumTruncation
from zcu_tools.resources.ftf.schema import Resonator as ResourceResonator
from zcu_tools.simulate.evolution import DriveSource
from zcu_tools.simulate.ftf import (
    FluxoniumCircuit,
    JointModel,
    JointParameters,
    Resonator,
    SquidCircuit,
)


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
    """Translate calibrated microwave quadratures to fixed-DC numeric source operators."""
    zero = np.zeros_like(joint.hamiltonian(joint.dc_flux))
    sources = []
    for name, source in settings.drive_lines.items():
        coefficients = tuple(
            complex(c.real, c.imag) for c in source.charge_drive_ghz_per_gain.as_tuple()
        )
        # Re[c*z] = Re(c)*Re(z)-Im(c)*Im(z); n itself is Hermitian.
        x = sum(
            (c.real * joint.charge_operator(i) for i, c in enumerate(coefficients)),
            zero.copy(),
        )
        y = sum(
            (-c.imag * joint.charge_operator(i) for i, c in enumerate(coefficients)),
            zero.copy(),
        )
        sources.append(DriveSource(name, x, y))
    return tuple(sources)
