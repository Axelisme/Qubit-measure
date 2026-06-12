"""build_predictor_from_simparams — derive a FluxoniumPredictor from SimParams.

FLUX-AWARE-MOCK: when the offline MockSoc is connected, the session installs a
FluxoniumPredictor whose physics matches the SimEngine, so predicted f01 tracks
the simulated qubit out of the box. This module owns the single mapping from the
mock's :class:`SimParams` (EJ/EC/EL + flux affine) to the predictor's constructor.

Dependency direction
--------------------
``SimParams`` (``program/v2/sim``) and ``FluxoniumPredictor``
(``simulate/fluxonium``) are independent lib leaves — neither imports the other.
Putting the bridge in either leaf would forge a new cross-dependency between them
(and pull scqubits into the program layer). The session layer already sits *above*
both and is the one place that owns the "Use MockSoc" concept, so the builder lives
here: it imports downward into both leaves and creates no upward import. The test
helpers reuse this exact function instead of re-deriving the constructor args.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from zcu_tools.simulate.fluxonium.predict import FluxoniumPredictor

if TYPE_CHECKING:
    from zcu_tools.program.v2.sim import SimParams


def build_predictor_from_simparams(sim: SimParams) -> FluxoniumPredictor:
    """A FluxoniumPredictor matching the mock SimEngine's physics.

    Maps SimParams to FluxoniumPredictor.__init__ per the alignment documented on
    SimParams: ``(EJ, EC, EL)`` as the params tuple, and the shared flux affine
    (``flux_half`` / ``flux_period`` / ``flux_bias``). The predictor then resolves
    f01 through the same reduced-flux mapping (``value_to_flux``) the SimEngine uses,
    so its prediction tracks the simulated qubit.
    """
    return FluxoniumPredictor(
        params=(sim.EJ, sim.EC, sim.EL),
        flux_half=sim.flux_half,
        flux_period=sim.flux_period,
        flux_bias=sim.flux_bias,
    )
