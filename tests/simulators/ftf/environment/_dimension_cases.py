"""Synthetic FTF sweeps wide enough to exercise the acquire dimension policy."""

from dataclasses import replace
from typing import Literal

import numpy as np
from zcu_tools.simulate.boundary import AcquireRequest, TimelineTemplate
from zcu_tools.simulators.ftf.config import FTFSettings
from zcu_tools.simulators.ftf.environment import FTFReadoutConfig

from tests.core._simulation import tick_array, ticks
from tests.simulators.ftf.environment._templates import (
    herald_template,
    request,
    settings,
    template,
)


def dimension_settings(upper: int) -> FTFSettings:
    """Use a synthetic 18-state product with a caller-selected joint ceiling."""
    base = settings()
    numerics = base.numerics
    truncation = numerics.truncation
    # The 3x2x3 product supports an upper basis larger than the pilot minimum.
    truncation = truncation.model_copy(
        update={
            "retained_levels": upper,
            "f1": truncation.f1.model_copy(update={"levels": 3}),
            "f2": truncation.f2.model_copy(update={"levels": 3}),
        }
    )
    return base.model_copy(
        update={"numerics": numerics.model_copy(update={"truncation": truncation})}
    )


def dimension_request(points: int = 8) -> AcquireRequest:
    """Build a gain sweep with two points per template on the CPU test profile."""
    # The highest-drive point is not first in C order or alone in its template.
    base = template(tones=1)
    templates = []
    for index in range(0, points, 2):
        rows = slice(0, min(2, points - index))
        drives = (
            replace(base.events[0], gain=np.array([0.1, 0.2 + index / 10])),
            base.events[1],
        )
        templates.append(
            replace(
                base,
                points=base.points[rows] + index,
                events=tuple(
                    replace(
                        drive,
                        start_ticks=drive.start_ticks[rows],
                        duration_ticks=drive.duration_ticks[rows],
                        frequency_mhz=drive.frequency_mhz[rows],
                        phase_deg=drive.phase_deg[rows],
                        gain=drive.gain[rows],
                        envelope_index=drive.envelope_index[rows],
                        active=drive.active[rows],
                    )
                    for drive in drives
                ),
                readouts=tuple(
                    replace(
                        read,
                        trigger_ticks=read.trigger_ticks[rows],
                        frequency_mhz=read.frequency_mhz[rows],
                        phase_deg=read.phase_deg[rows],
                        generator_channels=read.generator_channels[rows],
                    )
                    for read in base.readouts
                ),
                duration_ticks=base.duration_ticks[rows],
            )
        )
    return replace(
        request(templates=tuple(templates)), point_dims=(points,), loop_dims=(3, points)
    )


def _delay_terminal_readout(
    plan: TimelineTemplate, delay_ticks: int, terminal_tones: int
) -> TimelineTemplate:
    readouts = []
    for read in plan.readouts:
        triggers = read.trigger_ticks.copy()
        triggers[:, -1] += delay_ticks
        readouts.append(replace(read, trigger_ticks=triggers))
    return replace(
        plan,
        events=(
            *plan.events[:-terminal_tones],
            *(
                replace(event, start_ticks=event.start_ticks + delay_ticks)
                for event in plan.events[-terminal_tones:]
            ),
        ),
        readouts=tuple(readouts),
        duration_ticks=plan.duration_ticks + delay_ticks,
    )


def dimension_bypass(
    kind: Literal["herald", "early_cavity"],
) -> tuple[AcquireRequest, FTFReadoutConfig | None]:
    """Use eight points and require either heralding or an early cavity continuation."""
    acquire = dimension_request()
    if kind == "herald":
        herald, readout = herald_template()
        templates = tuple(
            replace(herald, points=plan.points) for plan in acquire.templates
        )
        terminal_tones = 2
    else:
        # One early cavity template is sufficient to bypass the whole acquire.
        first, *others = acquire.templates
        early = replace(
            first.events[-1],
            start_ticks=np.zeros(2, dtype=np.int64),
            duration_ticks=tick_array(np.full(2, 0.002)),
            envelopes=(None,),
        )
        first = replace(first, events=(early, first.events[-1]))
        templates = (first, *others)
        readout = None
        terminal_tones = 1

    # Distinct terminal starts identify the eight points at Engine.evolve.
    # The 8/10 ns pair repeats every 4 ns, after all earlier pulse edges.
    staggered = tuple(
        _delay_terminal_readout(plan, index * ticks(0.004), terminal_tones)
        for index, plan in enumerate(templates)
    )
    return replace(acquire, templates=staggered), readout
