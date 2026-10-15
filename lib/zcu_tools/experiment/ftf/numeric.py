"""Named resource-to-numeric translation; no YAML or solver internals."""

import numpy as np

from zcu_tools.resources.ftf import FTFSettings
from zcu_tools.resources.ftf.schema import Fluxonium
from zcu_tools.simulate.ftf import (
    FluxoniumCircuit,
    JointParameters,
    Resonator,
    SquidCircuit,
)


def circuit(value: Fluxonium) -> FluxoniumCircuit:
    """Translate named fluxonium GHz/truncation/noise values without rescaling."""
    return FluxoniumCircuit(
        value.ej_ghz,
        value.ec_ghz,
        value.el_ghz,
        value.native_cutoff,
        value.levels,
        value.q_cap,
        value.q_ind,
        value.xqp,
        value.aphi,
    )


def joint_parameters(settings: FTFSettings) -> JointParameters:
    """Return pure-numeric F1/C/F2 parameters; no resource type enters simulate."""
    m = settings.model
    c = m.coupler
    coupler = SquidCircuit(
        c.ejmax_ghz,
        c.ec_ghz,
        c.asymmetry,
        c.charge_cutoff,
        c.levels,
        c.ng,
        c.q_cap,
        c.resistance_ohm,
        c.flux_resistance_ohm,
        c.mutual_phi0_per_a,
        c.aphi,
    )
    return JointParameters(
        circuit(m.f1),
        coupler,
        circuit(m.f2),
        m.g1c_ghz,
        m.g2c_ghz,
        m.g12_ghz,
        m.retained_levels,
    )


def resonators(settings: FTFSettings) -> tuple[Resonator, Resonator]:
    """Convert kappa/(2*pi) MHz into photon decay 1/ns exactly once."""

    def convert(index: int) -> Resonator:
        r = settings.resonators[index]
        return Resonator(
            r.frequency_ghz,
            r.charge_coefficients_ghz,
            2 * np.pi * r.kappa_external_mhz / 1000,
            2 * np.pi * r.kappa_internal_mhz / 1000,
            r.thermal_occupation,
        )

    return convert(0), convert(1)
