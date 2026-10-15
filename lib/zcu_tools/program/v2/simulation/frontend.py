"""Prepare immutable numeric Module timelines for synchronous acquisition."""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

from zcu_tools.core.simulation import (
    AcquireRequest,
    AcquisitionMode,
)

from .timeline import TimelineInterpreter

if TYPE_CHECKING:
    from zcu_tools.program.acquisition import ImproveAcquireMixin

    from .binding import SimulationBinding


def build_request(
    prog: ImproveAcquireMixin,
    mode: AcquisitionMode,
    binding: SimulationBinding,
    sequence: int,
) -> AcquireRequest:
    """Freeze all compiled loop controls before executing a backend round.

    prog must be a compiled ModularProgramV2 using supported semantic Modules.
    mode selects ADC sums or traces; binding/sequence supply private RNG identity.
    Unsupported operations and inconsistent ADC layouts raise ValueError before
    any backend call. Returned records retain no mutable Module state.
    """
    from ..modular import ModularProgramV2

    if not isinstance(prog, ModularProgramV2):
        raise ValueError("simulation requires a ModularProgramV2 frontend")
    if prog.loop_dims is None or prog.avg_level is None:
        prog.compile()
    if prog.loop_dims is None or prog.avg_level is None:
        raise ValueError("compiled acquisition layout is missing")
    interpreter = TimelineInterpreter(prog)
    loops = prog.loop_dict
    timelines = tuple(
        interpreter.run_point(dict(zip(loops, point, strict=True)))
        for point in product(*(range(n) for n in loops.values()))
    )
    dims = tuple(int(n) for n in prog.loop_dims)
    windows = timelines[0].readouts
    if not windows or any(
        window.reads != prog.ro_chs[window.channel]["trigs"]
        for timeline in timelines
        for window in timeline.readouts
    ):
        raise ValueError("simulation readout layout does not match compiled triggers")
    return AcquireRequest(
        mode,
        dims,
        int(prog.avg_level),
        windows,
        tuple(timeline.events for timeline in timelines),
        binding.root_seed,
        sequence,
        binding.rng_version,
        timelines,
    )
