"""FTF compact marginals and shot-aligned projective readout."""

import json
from collections.abc import Sequence
from contextlib import closing
from dataclasses import replace
from typing import override

import numpy as np
import pytest
from zcu_tools.cfg_model import MuxReadoutTone
from zcu_tools.device import DeviceManager
from zcu_tools.program.v2 import TimelineBuilder
from zcu_tools.program.v2.modules.readout import MuxReadoutCfg
from zcu_tools.program.v2.simulation import IntegratedRound
from zcu_tools.simulate import adc
from zcu_tools.simulate.adc import ADC
from zcu_tools.simulate.boundary import (
    CancellationQuery,
    RoundSignals,
    SamplingIdentity,
    TimelineTemplate,
)
from zcu_tools.simulate.evolution import (
    CaptureBuffer,
    Engine,
    EvolutionResult,
    ModelSource,
    PathDiagnostics,
    TimelineBatch,
)
from zcu_tools.simulators.ftf.config.schema import MuxReadoutSource
from zcu_tools.simulators.ftf.environment import FTFSource, plan_template
from zcu_tools.simulators.telemetry import RoutingTelemetry

from tests.core._simulation import TIME_BASE, tick_array, ticks
from tests.program._profiles import profile_soccfg
from tests.simulators._engine import EngineDouble
from tests.simulators._sources import ftf_environment as ftf
from tests.simulators.ftf.environment._templates import (
    expect_statistics,
    herald_template,
    request,
    settings,
    snapshot,
    template,
)

pytestmark = pytest.mark.sim


class RejectEvolution(Engine):
    """Any solve means request validation missed its pre-evolution boundary."""

    def evolve(self, source, batches, **kwargs):
        raise AssertionError("inconsistent request topology reached evolution")


@pytest.mark.parametrize("difference", ["projective_count", "adc_layout"])
def test_request_rejects_equal_total_reads_with_different_template_event_layout(
    difference,
):
    first, readout = herald_template()
    both = first.readouts[0]
    terminal = replace(
        both,
        trigger_ticks=np.column_stack(
            (both.trigger_ticks[:, 1], both.trigger_ticks[:, 1] + ticks(0.004))
        ),
        frequency_mhz=np.full((2, 2), 6000.0),
    )
    other_adc = first.readouts[1]
    events = first.events[-2:]
    if difference == "adc_layout":
        events = first.events
        other_adc = replace(
            other_adc,
            trigger_ticks=both.trigger_ticks[:, :1],
            frequency_mhz=np.full((2, 1), 5980.0),
        )
    second = replace(
        first, points=first.points + 2, events=events, readouts=(terminal, other_adc)
    )
    acquire = replace(
        request(templates=(first, second)), point_dims=(4,), loop_dims=(3, 4)
    )
    source = FTFSource(
        settings().with_telemetry(enabled=False),
        DeviceManager(),
        soccfg=profile_soccfg("ftf"),
        readout=readout,
        engine=RejectEvolution(1),
    )
    try:
        with pytest.raises(ValueError, match=r"templates 0.*1.*layout"):
            source.open_acquire(acquire).sample_round(snapshot(), 0)
    finally:
        source.close()


@pytest.mark.parametrize("mode", ["raw", "integrated", "decimated"])
def test_trigger_emission_columns_survive_event_grouping_with_partial_adc(mode):
    base, config = herald_template()
    both, partial = base.readouts
    # Point zero emits terminal before herald; point one stays chronological.
    order = np.array([[1, 0], [0, 1]])
    reordered = replace(
        both,
        trigger_ticks=np.take_along_axis(both.trigger_ticks, order, axis=1),
        frequency_mhz=np.take_along_axis(both.frequency_mhz, order, axis=1),
        phase_deg=np.take_along_axis(both.phase_deg, order, axis=1),
    )
    partial = replace(
        partial,
        frequency_mhz=both.frequency_mhz[:, 1:],
        phase_deg=both.phase_deg[:, 1:],
    )
    numeric = settings().with_telemetry(enabled=False)
    canonical = plan_template(
        numeric,
        snapshot(),
        replace(base, readouts=(both, partial)),
        soccfg=profile_soccfg("ftf"),
        readout=config,
        time_base=TIME_BASE,
    )
    emitted = plan_template(
        numeric,
        snapshot(),
        replace(base, readouts=(reordered, partial)),
        soccfg=profile_soccfg("ftf"),
        readout=config,
        time_base=TIME_BASE,
    )
    readout = adc.ProjectionReadout(
        np.array([[0.4, 0.6], [0.7, 0.3]]),
        (
            np.tile(np.eye(2), (2, 1, 1)),
            np.tile(np.eye(2)[::-1], (2, 1, 1)),
        ),
    )
    identity = SamplingIdentity(123, (7, 0, 1))
    reps = 8

    def observe(plan):
        batches = {item.channel: item.batch for item in plan.readouts}
        if mode == "raw":
            return {
                channel: np.moveaxis(values, 0, 1)
                for channel, values in adc.PreparedRawShots(readout, batches, reps=reps)
                .bind(identity)
                .materialize(reps, 0)
                .items()
            }
        if mode == "decimated":
            return {
                item.channel: item.iq.reshape(2, reps, *item.iq.shape[1:])
                for item in adc.sample_traces(
                    readout, batches, reps=reps, identity=identity
                ).channels
            }
        raise AssertionError("shot observation requires raw or decimated mode")

    def statistics(plan):
        batches = {item.channel: item.batch for item in plan.readouts}
        return adc.round_statistics(
            readout,
            tuple(
                adc.ChannelReadout(
                    channel,
                    adc.ConditionalReadout(batch.model).responses(batch),
                    batch.content_index,
                    batch.model.receiver,
                )
                for channel, batch in batches.items()
            ),
            reps=reps,
            seed=123,
            acquire_sequence=7,
            round_index=0,
        )

    if mode == "integrated":
        expected, actual = statistics(canonical), statistics(emitted)
        np.testing.assert_array_equal(actual.state_counts, expected.state_counts)
        old, new = expected.channels[0], actual.channels[0]
        assert not np.allclose(old.conditional_sums[0, 0], old.conditional_sums[0, 1])
        np.testing.assert_allclose(
            new.conditional_sums[0], old.conditional_sums[0, ::-1], rtol=1e-12
        )
        np.testing.assert_allclose(new.conditional_sums[1], old.conditional_sums[1])
        # The direct draw adds the same output-column noise to reordered means.
        old_mean = np.einsum(
            "ps,prsi->pri", expected.state_counts, old.conditional_sums
        )
        new_mean = np.einsum("ps,prsi->pri", actual.state_counts, new.conditional_sums)
        np.testing.assert_allclose(new.sums, new_mean + old.sums - old_mean, atol=1e-10)
        np.testing.assert_array_equal(
            actual.channels[1].sums, expected.channels[1].sums
        )
    else:
        expected_shots, actual_shots = observe(canonical), observe(emitted)
        read_axis = 1
        assert not np.array_equal(
            expected_shots[4][0, :, 0], expected_shots[4][0, :, 1]
        )
        np.testing.assert_array_equal(
            actual_shots[4][0], np.take(expected_shots[4][0], [1, 0], axis=read_axis)
        )
        np.testing.assert_array_equal(actual_shots[4][1], expected_shots[4][1])
        np.testing.assert_array_equal(actual_shots[1], expected_shots[1])
        # The partial ADC observes the same terminal CTMC, not another event.
        np.testing.assert_array_equal(actual_shots[4][0, :, :1], actual_shots[1][0])
        np.testing.assert_array_equal(actual_shots[4][1, :, 1:], actual_shots[1][1])


def thermal_mux_settings():
    numeric = settings().with_telemetry(enabled=False)
    chip = numeric.chip.model_copy(
        update={
            "temperature_k": 1.0,
            "f1": numeric.chip.f1.model_copy(
                update={"q_cap": None, "q_ind": None, "aphi": 0.0, "xqp": 0.0}
            ),
            "f2": numeric.chip.f2.model_copy(
                update={"q_cap": None, "q_ind": None, "aphi": 0.0, "xqp": 0.0}
            ),
            "coupler": numeric.chip.coupler.model_copy(
                update={"q_cap": None, "aphi": 0.0}
            ),
        }
    )
    receiver = numeric.readout.adc.model_copy(update={"gain": 1e12})
    return numeric.model_copy(
        update={
            "chip": chip,
            "numerics": numeric.numerics.model_copy(
                update={
                    "truncation": numeric.numerics.truncation.model_copy(
                        update={"retained_levels": 4}
                    )
                }
            ),
            "readout": numeric.readout.model_copy(
                update={
                    "mux": MuxReadoutSource(
                        gen_ch=1,
                        input_per_gain=1.0,
                        adc=tuple(
                            receiver.model_copy(update={"ro_ch": ch}) for ch in (4, 1)
                        ),
                    )
                }
            ),
        }
    )


def herald_program(env, *, reads=2, reps=256):
    mux = MuxReadoutCfg(
        gen_ch=1,
        nqz=1,
        mixer_freq=6000.0,
        length=0.020,
        ro_length=0.026,
        tones=[
            MuxReadoutTone(ro_ch=ch, freq=freq, gain=0.01)
            for ch, freq in ((4, 6000.0), (1, 6200.0))
        ],
    )
    builder = TimelineBuilder(env.soccfg, reps=reps)
    with builder.repeat("reads", reads):
        builder.emit(mux.build("joint"))
        builder.timeline_inc(0.020)
    builder.delay_by_timeline()
    return builder.build()


@pytest.mark.parametrize("mode", ["integrated", "decimated"])
def test_mux_herald_postselection_retains_joint_ground_and_reports_nodes(
    tmp_path, mode
):
    numeric = thermal_mux_settings().with_telemetry(enabled=True, directory=tmp_path)
    env = ftf(
        numeric,
        soccfg=profile_soccfg("ftf"),
        time_scale=0,
        telemetry=RoutingTelemetry(tmp_path),
    )
    try:
        prog = herald_program(env)
        if mode == "integrated":
            prog.acquire(env.soc, progress=False)
        else:
            prog.acquire_decimated(env.soc, progress=False)
        raw = prog.get_raw()
        assert raw is not None
        captured = env.soc.binding.rounds[0]
        plan = plan_template(
            numeric,
            captured.snapshot,
            captured.acquire.templates[0],
            soccfg=profile_soccfg("ftf"),
            time_base=captured.acquire.time_base,
        )
        labels = []
        for read in range(2):
            observed = np.concatenate([shots[:, read] for shots in raw], axis=-1)
            centres = np.concatenate(
                [
                    ADC(channel.batch.model).responses(channel.batch).sums[0, read]
                    for channel in plan.readouts
                ],
                axis=-1,
            )
            distances = np.linalg.norm(observed[:, None] - centres[None], axis=-1)
            labels.append(np.argmin(distances, axis=-1))
        ground = labels[0] == 0
        assert 0.1 < ground.mean() < 0.5
        assert np.mean(labels[1][ground] == 0) > 0.97
        assert np.mean(labels[1] == labels[0]) > 0.97
        record = json.loads(
            next(tmp_path.glob("routing-*.jsonl")).read_text().splitlines()[-1]
        )
        assert record["extras"]["ftf.projection_continued_states"] == [[4]]
        assert record["extras"]["ftf.projection_identity_weight"] == [[0.0]]
    finally:
        env.devices.close_all_devices()


class FlipAtSecondRead(EngineDouble):
    """Flip populations in the span ending at the fixture's second read."""

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
    ) -> EvolutionResult:
        result = super().evolve(
            source,
            batches,
            budget=budget,
            cancel=cancel,
            telemetry=telemetry,
            reuse_key=reuse_key,
            capture=capture,
        )
        ends = np.concatenate([batch.end_ns for batch in batches])
        flip = np.isin(ends, [8.0, 10.0])
        densities = result.densities.copy()
        densities[flip] = densities[flip, ::-1, ::-1]
        return replace(result, densities=densities)


@pytest.fixture
def flipping_readout():
    base, readout = herald_template()
    numeric = settings().with_telemetry(enabled=False)
    chip = numeric.chip.model_copy(
        update={
            "temperature_k": 1.0,
            "f1": numeric.chip.f1.model_copy(
                update={"q_cap": None, "q_ind": None, "aphi": 0.0, "xqp": 0.0}
            ),
            "f2": numeric.chip.f2.model_copy(
                update={"q_cap": None, "q_ind": None, "aphi": 0.0, "xqp": 0.0}
            ),
            "coupler": numeric.chip.coupler.model_copy(
                update={"q_cap": None, "aphi": 0.0}
            ),
        }
    )
    return base, readout, numeric.model_copy(update={"chip": chip})


def test_compact_bins_partition_reps_and_weight_each_reads_own_flip_marginal(
    flipping_readout,
):
    base, readout, numeric = flipping_readout
    source = FTFSource(
        numeric,
        DeviceManager(),
        soccfg=profile_soccfg("ftf"),
        readout=readout,
        engine=FlipAtSecondRead(),
    )
    acquire = replace(request(templates=(base,)), reps=20000)
    try:
        facts = source.model_facts(snapshot())
        plan = plan_template(
            numeric,
            snapshot(),
            base,
            soccfg=source.soccfg,
            readout=readout,
            facts=facts,
            time_base=acquire.time_base,
        )
        result = source.open_acquire(acquire).sample_round(snapshot(), 0)
        statistics = expect_statistics(result)
        # Two two-state events use at most 1 + (2 - 1) * 2 ordered bins,
        # not four (herald state, terminal state) joint categories.
        assert statistics.state_counts.shape[0] == 2
        assert statistics.state_counts.shape[1] <= 3
        np.testing.assert_array_equal(
            statistics.state_counts.sum(axis=-1), acquire.reps
        )

        initial = facts.source.initial_density.diagonal().real
        expected_occupancies = []
        for item, channel in zip(plan.readouts, statistics.channels, strict=True):
            response = ADC(item.batch.model).responses(item.batch)
            centres = response.sums[item.batch.content_index]
            assert channel.sums.shape[-2] == (2 if item.channel == 4 else 1)
            assert (
                np.linalg.norm(centres[..., 0, :] - centres[..., 1, :], axis=-1) > 1e-5
            ).all()
            is_ground = np.isclose(
                channel.conditional_sums, centres[..., :1, :], rtol=1e-12, atol=1e-9
            ).all(axis=-1)
            is_excited = np.isclose(
                channel.conditional_sums, centres[..., 1:, :], rtol=1e-12, atol=1e-9
            ).all(axis=-1)
            assert (is_ground ^ is_excited).all()
            ground = np.einsum("pb,prb->pr", statistics.state_counts, is_ground)
            expected = initial if item.channel == 4 else initial[1:]
            np.testing.assert_allclose(
                ground / acquire.reps,
                np.broadcast_to(expected, ground.shape),
                atol=0.01,
            )
            expected_occupancies.append(ground)
            weighted = np.einsum(
                "pb,prbi->pri", statistics.state_counts, channel.conditional_sums
            )
            expected_total = (
                ground[..., None] * centres[..., 0, :]
                + (acquire.reps - ground)[..., None] * centres[..., 1, :]
            )
            np.testing.assert_allclose(weighted, expected_total, rtol=1e-12, atol=1e-9)
            # With zero receiver noise, observed totals only add the ADC's
            # integrated quantization noise (variance reps/12 per component).
            np.testing.assert_allclose(
                channel.sums, weighted, atol=6 * np.sqrt(acquire.reps / 12), rtol=0
            )
        both, terminal = expected_occupancies
        np.testing.assert_array_equal(both[:, :1] + terminal, acquire.reps)
        np.testing.assert_array_equal(both[:, 1:], terminal)
    finally:
        source.close()


@pytest.mark.parametrize("mode", ["integrated", "decimated"])
def test_flipping_chain_preserves_raw_and_trace_pairs_with_terminal_only_adc(
    flipping_readout,
    mode,
):
    base, readout, numeric = flipping_readout
    assert readout.mux is not None
    readout = replace(
        readout,
        mux=replace(
            readout.mux,
            receivers=tuple(
                replace(receiver, gain=1e8) for receiver in readout.mux.receivers
            ),
        ),
    )
    source = FTFSource(
        numeric,
        DeviceManager(),
        soccfg=profile_soccfg("ftf"),
        readout=readout,
        engine=FlipAtSecondRead(),
    )
    acquire = replace(request(mode, templates=(base,)), reps=64)
    try:
        plan = plan_template(
            numeric,
            snapshot(),
            base,
            soccfg=source.soccfg,
            readout=readout,
            facts=source.model_facts(snapshot()),
            time_base=acquire.time_base,
        )
        result = source.open_acquire(acquire).sample_round(snapshot(), 0)
        if isinstance(result, IntegratedRound):
            shots = result.raw.materialize(acquire.reps, acquire.avg_level)
        else:
            shots = {
                channel.channel: channel.iq.sum(axis=-2).reshape(acquire.reps, 2, -1, 2)
                for channel in result.channels
            }
        labels = {}
        for item in plan.readouts:
            centres = (
                ADC(item.batch.model)
                .responses(item.batch)
                .sums[item.batch.content_index]
            )
            distances = np.linalg.norm(
                shots[item.channel][..., None, :] - centres[None], axis=-1
            )
            labels[item.channel] = distances.argmin(axis=-1)
        both = labels[4]
        assert set(np.unique(both[..., 0])) == {0, 1}
        np.testing.assert_array_equal(both[..., 0] + both[..., 1], 1)
        np.testing.assert_array_equal(both[..., 1:], labels[1])
    finally:
        source.close()


@pytest.mark.parametrize("floor", [0.0, 0.9])
def test_projection_floor_preserves_reps_even_if_every_row_is_identity(floor):
    template, readout = herald_template()
    numeric = settings().with_telemetry(enabled=False)
    numeric = numeric.model_copy(
        update={
            "numerics": numeric.numerics.model_copy(
                update={"projection_weight_floor": floor}
            ),
            "chip": numeric.chip.model_copy(
                update={"temperature_k": 1.0 if floor else 0.0}
            ),
        }
    )
    source = FTFSource(
        numeric,
        DeviceManager(),
        soccfg=profile_soccfg("ftf"),
        readout=readout,
        engine=EngineDouble(),
    )
    try:
        session = source.open_acquire(request(templates=(template,)))
        statistics = expect_statistics(session.sample_round(snapshot(), 0))
        np.testing.assert_array_equal(statistics.state_counts.sum(axis=-1), 3)
        assert source.evaluations[0].projection_continued_states == (
            ((0,), (0,)) if floor else ((2,), (2,))
        )
        assert source.evaluations[0].projection_identity_weight == (
            ((1.0,), (1.0,)) if floor else ((0.0,), (0.0,))
        )
    finally:
        source.close()


@pytest.mark.parametrize("mode", ["integrated", "decimated"])
def test_unobserved_terminal_still_evolves_without_adding_an_adc_column(mode):
    base, readout = herald_template()
    first = base.readouts[0]
    first = replace(
        first,
        trigger_ticks=first.trigger_ticks[:, :1],
        frequency_mhz=first.frequency_mhz[:, :1],
        phase_deg=first.phase_deg[:, :1],
        generator_channels=first.generator_channels[:, :1],
    )
    base = replace(base, readouts=(first,))
    numeric = settings().with_telemetry(enabled=False)
    numeric = numeric.model_copy(
        update={"chip": numeric.chip.model_copy(update={"temperature_k": 1.0})}
    )
    source = FTFSource(
        numeric,
        DeviceManager(),
        soccfg=profile_soccfg("ftf"),
        readout=readout,
        engine=FlipAtSecondRead(),
    )
    try:
        result = source.open_acquire(request(mode, templates=(base,))).sample_round(
            snapshot(), 0
        )
        if mode == "integrated":
            if not isinstance(result, IntegratedRound) or not isinstance(
                result.raw, adc.ADCRawShots
            ):
                raise AssertionError(
                    "integrated FTF must return ADC statistics and raw"
                )
            chain = result.raw.prepared.readout
            assert isinstance(chain, adc.ProjectionChain)
            (terminal,) = chain.transitions
            np.testing.assert_array_equal(terminal, np.tile(np.eye(2)[::-1], (2, 1, 1)))
            assert not dict(result.raw.prepared.batches)[4].events[0][1].grids
            assert result.statistics.channels[0].sums.shape == (2, 1, 2)
        else:
            if not isinstance(result, RoundSignals):
                raise AssertionError("decimated round must return traces")
            assert result.channels[0].iq.shape == (6, 1, 4, 2)
        assert source.evaluations[0].projection_continued_states == ((2,), (2,))
    finally:
        source.close()


@pytest.mark.parametrize("mode", ["integrated", "decimated"])
def test_terminal_only_runtime_hands_given_density_order_and_seed_to_adc(mode):
    numeric = settings().with_telemetry(enabled=False)
    populations = np.array([[0.8, 0.2], [0.3, 0.7]])
    engine = EngineDouble(
        result=EvolutionResult(
            np.array([1, 0], dtype=np.int64),
            np.array(
                [np.diag(populations[1]), np.diag(populations[0])], dtype=np.complex128
            ),
            PathDiagnostics(),
        )
    )
    source = FTFSource(
        numeric, DeviceManager(), soccfg=profile_soccfg("ftf"), engine=engine
    )
    try:
        acquire = request(mode)
        facts = source.model_facts(snapshot())
        plan = plan_template(
            numeric,
            snapshot(),
            template(),
            soccfg=profile_soccfg("ftf"),
            facts=facts,
            time_base=TIME_BASE,
        )
        readout = adc.ProjectionReadout.unprojected(populations)
        batch = plan.readouts[0].batch
        result = source.open_acquire(acquire).sample_round(snapshot(), 0)
        assert len(engine.calls) == 1
        if mode == "integrated":
            if not isinstance(result, IntegratedRound):
                raise AssertionError("integrated round must return statistics and raw")
            conditional = adc.ADC(
                batch.model,
                rtol=numeric.numerics.solver.rtol,
                atol=numeric.numerics.solver.atol,
            ).responses(batch)
            expected = adc.round_statistics(
                readout,
                (
                    adc.ChannelReadout(
                        0, conditional, batch.content_index, batch.model.receiver
                    ),
                ),
                reps=acquire.reps,
                seed=acquire.root_seed,
                acquire_sequence=acquire.sequence,
                round_index=0,
            )
            np.testing.assert_array_equal(
                result.statistics.state_counts, expected.state_counts
            )
            np.testing.assert_array_equal(
                result.statistics.channels[0].sums, expected.channels[0].sums
            )
            raw = adc.PreparedRawShots(
                readout,
                {0: batch},
                reps=acquire.reps,
                rtol=numeric.numerics.solver.rtol,
                atol=numeric.numerics.solver.atol,
            ).bind(expected.raw_identity)
            np.testing.assert_array_equal(
                result.raw.materialize(acquire.reps, 0)[0],
                raw.materialize(acquire.reps, 0)[0],
            )
        else:
            if not isinstance(result, RoundSignals):
                raise AssertionError("decimated round must return traces")
            expected = adc.sample_traces(
                readout,
                {0: batch},
                reps=acquire.reps,
                identity=SamplingIdentity(acquire.root_seed, (acquire.sequence, 0, 0)),
                rtol=numeric.numerics.solver.rtol,
                atol=numeric.numerics.solver.atol,
            )
            ordered = np.moveaxis(
                expected.channels[0].iq.reshape(2, 3, 1, 4, 2), 1, 0
            ).reshape(6, 1, 4, 2)
            np.testing.assert_array_equal(result.channels[0].iq, ordered)
    finally:
        engine.close()
        source.close()


def test_configured_floor_keeps_low_weight_state_instead_of_redistributing_it():
    template, readout = herald_template()
    numeric = settings().with_telemetry(enabled=False)
    numeric = numeric.model_copy(
        update={
            "chip": numeric.chip.model_copy(update={"temperature_k": 1.0}),
            "numerics": numeric.numerics.model_copy(
                update={"projection_weight_floor": 0.5}
            ),
        }
    )
    source = FTFSource(
        numeric,
        DeviceManager(),
        soccfg=profile_soccfg("ftf"),
        readout=readout,
        engine=FlipAtSecondRead(),
    )
    try:
        initial = source.model_facts(snapshot()).source.initial_density.diagonal().real
        result = source.open_acquire(
            replace(request(templates=(template,)), reps=100)
        ).sample_round(snapshot(), 0)
        if not isinstance(result, IntegratedRound) or not isinstance(
            result.raw, adc.ADCRawShots
        ):
            raise AssertionError("integrated FTF must return ADC statistics and raw")
        chain = result.raw.prepared.readout
        assert isinstance(chain, adc.ProjectionChain)
        (terminal,) = chain.transitions
        np.testing.assert_allclose(chain.initial, np.tile(initial, (2, 1)))
        # The adopted ground row flips, the skipped excited row stays excited.
        np.testing.assert_array_equal(terminal, np.tile([[[0, 1], [0, 1]]], (2, 1, 1)))
        assert source.evaluations[0].projection_continued_states == ((1,), (1,))
        np.testing.assert_allclose(
            source.evaluations[0].projection_identity_weight,
            [[initial[1]], [initial[1]]],
            atol=1e-10,
        )
        np.testing.assert_array_equal(
            result.statistics.state_counts.sum(axis=-1), [100, 100]
        )
    finally:
        source.close()


def _with_third_read(base: TimelineTemplate) -> TimelineTemplate:
    last = tuple(
        replace(event, start_ticks=event.start_ticks + ticks(0.012))
        for event in base.events[-2:]
    )
    windows = tuple(
        replace(
            window,
            trigger_ticks=np.column_stack(
                (window.trigger_ticks, window.trigger_ticks[:, -1] + ticks(0.012))
            ),
            frequency_mhz=np.column_stack(
                (window.frequency_mhz, window.frequency_mhz[:, -1])
            ),
            phase_deg=np.column_stack((window.phase_deg, window.phase_deg[:, -1])),
            generator_channels=np.column_stack(
                (window.generator_channels, window.generator_channels[:, -1])
            ),
        )
        for window in base.readouts
    )
    middle = tuple(
        replace(event, duration_ticks=tick_array(np.full(2, 0.002)))
        for event in base.events[-2:]
    )
    return replace(
        base,
        events=(*base.events[:2], *middle, *last),
        readouts=windows,
        duration_ticks=base.duration_ticks + ticks(0.012),
    )


def test_second_transition_routes_the_state_populated_by_first_transition():
    base, readout = herald_template()
    numeric = settings().with_telemetry(enabled=False)
    numeric = numeric.model_copy(
        update={
            "numerics": numeric.numerics.model_copy(
                update={"projection_weight_floor": 0.5}
            )
        }
    )
    with closing(
        FTFSource(
            numeric,
            DeviceManager(),
            soccfg=profile_soccfg("ftf"),
            readout=readout,
            engine=FlipAtSecondRead(),
        )
    ) as source:
        acquire = request(templates=(_with_third_read(base),))
        source.open_acquire(acquire).sample_round(snapshot(), 0)

        # The first transition flips the ground state into the excited state.
        # Only that flipped state should enter the second transition.
        observed = {
            (span.point, span.event_index, state)
            for span in source.evaluations[0].projective_spans
            if span.event_index is not None
            for _, state in span.nodes
        }
        assert observed == {(0, 0, 0), (1, 0, 0), (0, 1, 1), (1, 1, 1)}


def test_two_projections_preserve_both_herald_states_in_conditional_statistics():
    base, readout = herald_template()
    assert readout.mux is not None
    readout = replace(
        readout,
        mux=replace(
            readout.mux,
            receivers=tuple(
                replace(receiver, gain=1e8) for receiver in readout.mux.receivers
            ),
        ),
    )
    numeric = settings().with_telemetry(enabled=False)
    chip = numeric.chip.model_copy(
        update={
            "temperature_k": 1.0,
            "f1": numeric.chip.f1.model_copy(update={"q_cap": 10.0}),
            "f2": numeric.chip.f2.model_copy(update={"q_cap": 10.0}),
        }
    )
    source = FTFSource(
        numeric.model_copy(
            update={
                "chip": chip,
                "numerics": numeric.numerics.model_copy(
                    update={"projection_weight_floor": 0.0}
                ),
            }
        ),
        DeviceManager(),
        soccfg=profile_soccfg("ftf"),
        readout=readout,
        engine=EngineDouble(),
    )
    try:
        result = source.open_acquire(
            request(templates=(_with_third_read(base),))
        ).sample_round(snapshot(), 0)
        statistics = expect_statistics(result)
        report = source.evaluations[0]
        assert report.projection_continued_states == ((2, 2), (2, 2))
        assert {
            span.event_index
            for span in report.projective_spans
            if span.event_index is not None
        } == {0, 1}
        assert all(len(span.nodes) <= 2 for span in report.projective_spans)
        sums = statistics.channels[0].conditional_sums
        assert sums.shape[:2] == (2, 3)
        # Both herald events still have distinguishable state-conditioned means.
        for event in (0, 1):
            assert np.max(np.ptp(sums[:, event], axis=1)) > 1e-5
    finally:
        source.close()
