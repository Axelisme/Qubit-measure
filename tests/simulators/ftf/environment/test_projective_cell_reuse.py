"""Exact projective cell reuse at the FTF acquisition and Engine seams."""

from collections.abc import Sequence
from contextlib import closing
from dataclasses import replace
from typing import override

import numpy as np
import pytest
from zcu_tools.device import DeviceManager
from zcu_tools.program.v2.simulation import IntegratedRound
from zcu_tools.simulate.adc import ADCRawShots, ProjectionChain
from zcu_tools.simulate.boundary import CancellationQuery, TimelineTemplate
from zcu_tools.simulate.evolution import (
    CaptureBuffer,
    EvolutionResult,
    ModelSource,
    SampledRWAProbeMemo,
    TimelineBatch,
)
from zcu_tools.simulators.ftf.environment import FTFNumericUse, FTFSource

from tests.core._simulation import tick_array
from tests.program._profiles import profile_soccfg
from tests.simulators._engine import EngineDouble
from tests.simulators.ftf.environment._templates import (
    request,
    settings,
    snapshot,
    template,
)

pytestmark = pytest.mark.sim


def _read_chain(
    *, phase: float, starts: tuple[float, ...] = (8, 16, 24, 32), support_ns: float = 2
) -> TimelineTemplate:
    base = template()
    pulses = tuple(
        replace(
            base.events[0],
            start_ticks=tick_array(np.full(2, start / 1000)),
            duration_ticks=tick_array(np.full(2, (support_ns if i == 1 else 2) / 1000)),
            # Counteract the local carrier conversion to select the test phase.
            phase_deg=np.full(2, 360 * 6.0 * start + (phase if i == 1 else 0)),
            envelopes=(None,),
        )
        for i, start in enumerate(starts)
    )
    ro = replace(
        base.readouts[0],
        trigger_ticks=tick_array(np.tile(np.array(starts) / 1000, (2, 1))),
        frequency_mhz=np.full((2, len(starts)), 6000.0),
        phase_deg=np.zeros((2, len(starts))),
        generator_channels=np.zeros((2, len(starts)), dtype=np.int64),
    )
    return replace(
        base,
        events=pulses,
        readouts=(ro,),
        duration_ticks=tick_array(np.full(2, starts[-1] / 1000 + 0.008)),
    )


def _run_chain(
    timeline: TimelineTemplate, engine: EngineDouble, *, floor: float = 0.0
) -> tuple[ProjectionChain, FTFNumericUse]:
    numeric = settings().with_telemetry(enabled=False)
    numeric = numeric.model_copy(
        update={
            "numerics": numeric.numerics.model_copy(
                update={"projection_weight_floor": floor}
            )
        }
    )
    with closing(
        FTFSource(numeric, DeviceManager(), soccfg=profile_soccfg("ftf"), engine=engine)
    ) as source:
        result = source.open_acquire(request(templates=(timeline,))).sample_round(
            snapshot(), 0
        )
        (diagnostics,) = source.evaluations
    if not isinstance(result, IntegratedRound) or not isinstance(
        result.raw, ADCRawShots
    ):
        raise AssertionError("expected sampled FTF projective chain")
    chain = result.raw.prepared.readout
    if not isinstance(chain, ProjectionChain):
        raise AssertionError("expected a full chain including the terminal transition")
    return chain, diagnostics


def _continued_spans(engine: EngineDouble) -> set[tuple[float, float]]:
    return {
        (float(start), float(end))
        for _, batches in engine.calls
        for batch in batches
        if batch.start_ns is not None
        for start, end in zip(batch.start_ns, batch.end_ns, strict=True)
        if start > 0
    }


@pytest.mark.parametrize("phase", [0.0, 67.0], ids=["match", "phase-only"])
def test_repeated_projective_cells_handoff_one_source_solve_per_point(phase):
    engine = EngineDouble(backend="cpu")
    chain, diagnostics = _run_chain(_read_chain(phase=phase), engine)
    assert len(chain.transitions) == 3
    for layer in chain.transitions:
        np.testing.assert_array_equal(layer, np.tile(np.eye(2), (2, 1, 1)))
    # Each point solves the first cell, shares its source rows with the
    # second cell, and still evolves to its own terminal event.
    assert _continued_spans(engine) == {(8.0, 16.0), (24.0, 32.0)}
    assert sum(len(batches) for _, batches in engine.calls) == 10
    assert diagnostics.projection_continued_states == ((2, 0, 2), (2, 0, 2))
    assert diagnostics.projection_identity_weight == ((0.0, 0.0, 0.0),) * 2


@pytest.mark.parametrize("other_support", [(24, 2), (20, 0)], ids=["touching", "empty"])
def test_declared_zero_gain_drive_blocks_reuse_only_where_it_overlaps(other_support):
    start, duration = other_support
    drive = replace(
        template(tones=1).events[0],
        start_ticks=tick_array(np.array([16, start]) / 1000),
        duration_ticks=tick_array(np.array([8, duration]) / 1000),
        gain=np.zeros(2),
    )
    timeline = _read_chain(phase=67.0)
    _, diagnostics = _run_chain(
        replace(timeline, events=(*timeline.events, drive)), EngineDouble(backend="cpu")
    )
    assert [
        span.point for span in diagnostics.projective_spans if span.event_index == 1
    ] == [0]
    assert diagnostics.projection_continued_states == ((2, 2, 2), (2, 0, 2))


@pytest.mark.parametrize("difference", ["span", "drive", "support"])
def test_projective_cells_with_different_controls_keep_independent_solves(difference):
    starts = (8, 17, 24, 32) if difference == "span" else (8, 16, 24, 32)
    timeline = _read_chain(
        phase=67.0, starts=starts, support_ns=3 if difference == "support" else 2
    )
    if difference == "drive":
        drive = replace(
            template(tones=1).events[0],
            start_ticks=tick_array(np.full(2, 0.016)),
            duration_ticks=tick_array(np.full(2, 0.008)),
        )
        timeline = replace(timeline, events=(*timeline.events, drive))
    engine = EngineDouble(backend="cpu")
    _run_chain(timeline, engine)
    assert _continued_spans(engine) == set(zip(starts[:-1], starts[1:], strict=True))
    assert sum(len(batches) for _, batches in engine.calls) == 14


class FlipProjectors(EngineDouble):
    """Prescribe a state flip after every cell, leaving the prefix alone."""

    @override
    def evolve(
        self,
        source: ModelSource,
        batches: Sequence[TimelineBatch],
        *,
        budget: float = 1e-3,
        cancel: CancellationQuery | None = None,
        telemetry: bool = False,
        reuse_key: bytes | None = None,
        capture: CaptureBuffer | None = None,
        probe_memo: SampledRWAProbeMemo | None = None,
    ) -> EvolutionResult:
        result = super().evolve(
            source,
            batches,
            budget=budget,
            cancel=cancel,
            telemetry=telemetry,
            reuse_key=reuse_key,
            capture=capture,
            probe_memo=probe_memo,
        )
        if batches[0].start_ns is not None and batches[0].start_ns[0] > 0:
            return replace(result, densities=result.densities[:, ::-1, ::-1])
        return result


def test_each_cell_uses_its_actual_floor_even_with_a_cached_row():
    engine = FlipProjectors(backend="cpu")
    timeline = _read_chain(phase=67.0, starts=(8, 16, 24, 32, 40))
    chain, _ = _run_chain(timeline, engine, floor=0.5)
    # First and third adopt ground. Second and terminal adopt excited.
    # The second cell must neither take the cached ground row nor reuse
    # the first layer's identity placeholder for its newly occupied state.
    for layer, destination in zip(chain.transitions, (1, 0, 1, 0), strict=True):
        np.testing.assert_array_equal(layer, np.eye(2)[np.full((2, 2), destination)])
    assert _continued_spans(engine) == {(8, 16), (16, 24), (32, 40)}
