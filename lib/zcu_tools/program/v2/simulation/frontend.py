"""Prepare immutable numeric Module timelines for synchronous acquisition."""

from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import TYPE_CHECKING

import numpy as np

from zcu_tools.core.simulation import (
    AcquireRequest,
    AcquisitionMode,
    ReadoutWindow,
)

from .vector import VectorInterpreter

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
    prog.validate_module_execution()
    if prog.loop_dims is None or prog.avg_level is None:
        prog.compile()
    if prog.loop_dims is None or prog.avg_level is None:
        raise ValueError("compiled acquisition layout is missing")
    interpreter = VectorInterpreter(prog)
    templates = interpreter.run_templates()
    loops = prog.loop_dict
    names = tuple(n for n in loops if n != "reps")
    point_dims = tuple(loops[n] for n in names)
    windows = tuple(
        ReadoutWindow(
            channel=slot.channel,
            gen_ch=int(slot.generator_channels[0, -1]),
            samples=slot.samples,
            sample_rate_mhz=slot.sample_rate_mhz,
            trigger_us=tuple(float(x) for x in slot.trigger_us[0]),
            frequency_mhz=tuple(float(x) for x in slot.frequency_mhz[0]),
            phase_deg=tuple(float(x) for x in slot.phase_deg[0]),
            generator_channels=tuple(int(x) for x in slot.generator_channels[0]),
        )
        for slot in templates[0].readouts
    )
    if not windows:
        raise ValueError("simulation readout layout does not match compiled triggers")
    request = AcquireRequest(
        mode=mode,
        loop_dims=tuple(int(n) for n in prog.loop_dims),
        avg_level=int(prog.avg_level),
        readouts=windows,
        controls=(),
        root_seed=binding.root_seed,
        sequence=sequence,
        rng_version=binding.rng_version,
        point_dims=point_dims,
        point_names=names,
        reps=loops["reps"],
        templates=templates,
    )
    legacy = replace(request, templates=interpreter.legacy_templates)
    timelines = []
    # Legacy controls/timelines expansion for old backends disappears with those fields.
    for execution in product(*(range(n) for n in loops.values())):
        coordinates = dict(zip(loops, execution, strict=True))
        point = int(
            np.ravel_multi_index(tuple(coordinates[n] for n in names), point_dims)
        )
        timeline = interpreter.flat_tops.restore(legacy.materialize_point(point))
        timelines.append(replace(timeline, coordinates=tuple(coordinates.items())))
    return replace(
        request,
        controls=tuple(t.events for t in timelines),
        timelines=tuple(timelines),
    )
