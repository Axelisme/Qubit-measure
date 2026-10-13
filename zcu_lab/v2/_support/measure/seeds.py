"""Laboratory scan-range defaults for measure adapters."""

from zcu_tools.gui.app.measure.seeds import Seed, custom
from zcu_tools.gui.cfg import SweepValue

from zcu_lab.v2._support.measure.ctx_helpers import (
    proper_flux_range,
    proper_qub_freq_range,
    proper_res_freq_range,
)


def res_freq_range(
    *,
    expts: int,
    span_factor: float = 1.5,
) -> Seed[SweepValue]:
    """Deferred resonator frequency range using the established measure policy."""

    return custom(
        lambda ctx: proper_res_freq_range(
            ctx.parameters, expts, span_factor=span_factor
        ),
        description=f"resonator frequency range ({expts} points)",
    )


def qub_freq_range(*, expts: int, span_factor: float = 1.5) -> Seed[SweepValue]:
    return custom(
        lambda ctx: proper_qub_freq_range(
            ctx.parameters, expts, span_factor=span_factor
        ),
        description=f"qubit frequency range ({expts} points)",
    )


def flux_range(*, expts: int) -> Seed[SweepValue]:
    return custom(
        lambda ctx: proper_flux_range(ctx.parameters, expts),
        description=f"flux range ({expts} points)",
    )
