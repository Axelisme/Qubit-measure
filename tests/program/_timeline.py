"""Shared native compilation setup for acquisition and simulator contract tests."""

from collections.abc import Sequence

from qick import QickConfig
from zcu_tools.program.v2 import TimelineBuilder, TimelineProgramV2
from zcu_tools.program.v2.modules import Module
from zcu_tools.program.v2.sweep import SweepCfg


def compile_modules(
    soccfg: QickConfig,
    modules: Sequence[Module | None],
    sweep: list[tuple[str, SweepCfg | int]] | None = None,
    *,
    reps: int = 1,
    initial_delay: float = 1.0,
    relax_delay: float = 1.0,
    **kwargs: object,
) -> TimelineProgramV2:
    """Compile a sequential native timeline with optional outer sweep loops.

    Modules are borrowed declarations. None entries emit nothing. The final
    delay_by_timeline waits for the complete body before shot relaxation.
    """
    builder = TimelineBuilder(
        soccfg,
        reps=reps,
        initial_delay=initial_delay,
        relax_delay=relax_delay,
        **kwargs,
    )
    for name, values in sweep or []:
        builder.add_loop(name, values.expts if isinstance(values, SweepCfg) else values)
    for module in modules:
        if module is not None:
            builder.emit(module)
    builder.delay_by_timeline()
    return builder.build()
