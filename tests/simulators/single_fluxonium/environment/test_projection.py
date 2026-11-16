"""Projective reads preserve shot histories through the program/source contract."""

from collections.abc import Generator
from dataclasses import replace
from threading import Event

import numpy as np
import pytest
from scipy.linalg import expm
from zcu_tools.core.simulation import SamplingIdentity
from zcu_tools.experiment.readout_plan import local_readout_pulses
from zcu_tools.experiment.simulation import SimulationEnvironment
from zcu_tools.program.v2.modules.delay import SoftDelay
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.simulation import IntegratedRound
from zcu_tools.simulate import adc
from zcu_tools.simulate.evolution import Cancelled, Engine
from zcu_tools.simulators.single_fluxonium.config import parse_single_fluxonium
from zcu_tools.simulators.single_fluxonium.environment.runtime import (
    SingleFluxoniumRuntimeOptions,
    SingleFluxoniumSource,
    single_fluxonium,
)

from tests.program._profiles import profile_soccfg
from tests.program._timeline import compile_modules
from tests.simulators.single_fluxonium.environment._inputs import settings

pytestmark = pytest.mark.sim


def projection_settings():
    values = settings().model_dump(mode="json")
    values["chip"]["temperature_k"] = 0.05
    values["chip"]["qubit"]["q_cap"] = 3000
    values["numerics"]["truncation"]["levels"] = 2
    values["numerics"]["time_scale"] = 0
    values["readout"]["adc"]["gain"] = 1e6
    values["readout"]["resonator"]["charge_coefficient_ghz"] = 0.03
    return parse_single_fluxonium(values)


@pytest.fixture
def projection_environment() -> Generator[
    tuple[SimulationEnvironment, SingleFluxoniumSource]
]:
    config = projection_settings()
    env = single_fluxonium(config, soccfg=profile_soccfg("single_fluxonium"))
    source = SingleFluxoniumSource(config, env.devices)
    try:
        yield env, source
    finally:
        source.close()
        if env.close_backend is not None:
            env.close_backend()
        env.devices.close_all_devices()


def repeated_read_program(env, reads=2, reps=4096):
    readout = PulseReadoutCfg(
        gen_ch=0,
        ro_ch=0,
        nqz=1,
        freq=7000,
        gain=0.1,
        waveform=ConstWaveformCfg(length=0.008),
        ro_length=0.004,
        trig_offset=0.002,
    )
    modules = [
        module
        for index in range(reads)
        for module in (readout.build(f"ro{index}"), SoftDelay(f"wait{index}", 0.024))
    ]
    return compile_modules(
        env.soccfg,
        modules=modules,
        reps=reps,
        initial_delay=0,
        relax_delay=0,
    )


def expect_chain(
    readout: adc.ProjectionReadout | adc.ProjectionChain,
) -> adc.ProjectionChain:
    """Require a chain at the projective preparation seam."""
    if not isinstance(readout, adc.ProjectionChain):
        raise AssertionError("projective preparation must return a chain")
    return readout


@pytest.mark.parametrize("floor", [0.4, 0.9])
def test_source_floor_keeps_low_probability_states_with_identity_rows(
    projection_environment: tuple[SimulationEnvironment, SingleFluxoniumSource],
    floor: float,
):
    env, source = projection_environment
    values = source.settings.model_dump(mode="json")
    values["numerics"]["projection_prune_floor"] = floor
    source.settings = parse_single_fluxonium(values)
    program = repeated_read_program(env, reps=1)
    program.acquire(env.soc, progress=False)
    request = env.soc.binding.rounds[0].acquire
    prepared = source.prepare(request, source.snapshot())
    chain = expect_chain(prepared.raw.readout)
    diagnostics = prepared.projection_diagnostics
    if diagnostics is None:
        raise AssertionError("projective preparation must report approximation")
    adopted = chain.initial >= floor
    assert diagnostics.continued_states == ((int(np.count_nonzero(adopted)),),)
    assert diagnostics.uncontinued_probability[0][0] == pytest.approx(
        chain.initial[~adopted].sum()
    )
    np.testing.assert_array_equal(chain.transitions[0][~adopted], np.eye(2)[~adopted])
    initial = source.model_facts(source.snapshot()).source.initial_density
    np.testing.assert_allclose(chain.initial, np.diag(initial).real, atol=1e-9)
    assert source.prepare(request, source.snapshot()) is prepared


@pytest.mark.parametrize("projections", [1, 2, 3])
def test_zero_floor_continues_zero_probability_states_and_reuses_preparation(
    projection_environment: tuple[SimulationEnvironment, SingleFluxoniumSource],
    projections: int,
):
    env, source = projection_environment
    values = source.settings.model_dump(mode="json")
    values["chip"]["temperature_k"] = 0
    values["numerics"]["projection_prune_floor"] = 0
    source.settings = parse_single_fluxonium(values)
    program = repeated_read_program(env, reads=projections + 1, reps=1)
    program.acquire(env.soc, progress=False)
    original = env.soc.binding.rounds[0]
    snapshot = source.snapshot()
    prepared = source.prepare(original.acquire, snapshot)
    readout = expect_chain(prepared.raw.readout)
    np.testing.assert_array_equal(readout.initial, [1, 0])
    assert len(readout.transitions) == projections
    diagnostics = prepared.projection_diagnostics
    if diagnostics is None:
        raise AssertionError("projective preparation must report retained histories")
    assert diagnostics.continued_states == ((2,) * projections,)
    np.testing.assert_array_equal(
        diagnostics.uncontinued_probability, [[0] * projections]
    )
    assert diagnostics.evolution_calls == len(prepared.reuse_keys) > 0
    assert source.prepare(original.acquire, snapshot) is prepared
    result = source.execute(replace(original, snapshot=snapshot))
    if not isinstance(result, IntegratedRound):
        raise AssertionError("integrated acquisition must retain joint counts")
    assert result.statistics.state_counts.sum() == 1
    assert result.statistics.state_counts.shape[-1] <= 1 + (projections + 1)
    assert source.prepare(original.acquire, snapshot) is prepared


def test_preparation_rejects_equal_read_counts_with_different_template_events(
    projection_environment: tuple[SimulationEnvironment, SingleFluxoniumSource],
):
    env, source = projection_environment
    program = repeated_read_program(env, reps=1)
    program.acquire(env.soc, progress=False)
    request = env.soc.binding.rounds[0].acquire
    herald = request.templates[0]
    window = herald.readouts[0]
    terminal_start = herald.events[-1].start_ticks[:, None]
    terminal_window = replace(
        window,
        trigger_ticks=terminal_start
        + np.array([[1, 2]], dtype=np.int64) * window.sample_period_ticks,
    )
    terminal = replace(herald, events=(herald.events[-1],), readouts=(terminal_window,))
    herald = replace(herald, points=np.array([1], dtype=np.int64))
    request = replace(
        request,
        point_dims=(2,),
        point_names=("sweep",),
        loop_dims=(1, 2),
        templates=(terminal, herald),
    )
    with pytest.raises(ValueError, match=r"template 1.*layout.*template 0"):
        source.prepare(request, source.snapshot())


def test_two_projections_keep_conditional_markov_transitions(
    projection_environment: tuple[SimulationEnvironment, SingleFluxoniumSource],
):
    env, source = projection_environment
    values = source.settings.model_dump(mode="json")
    values["numerics"]["projection_prune_floor"] = 0
    source.settings = parse_single_fluxonium(values)
    program = repeated_read_program(env, reads=3, reps=1)
    program.acquire(env.soc, progress=False)
    request = env.soc.binding.rounds[0].acquire
    prepared = source.prepare(request, source.snapshot())
    readout = expect_chain(prepared.raw.readout)
    facts = source.model_facts(source.snapshot())
    plan = source.plan_template(
        source.snapshot(), request.templates[0], time_base=request.time_base
    )
    rate = facts.readout.rates_per_ns
    generator = rate - np.diag(rate.sum(axis=0))
    times = [event.end_ns[0] for event in plan.read_events]
    transition = expm(generator * (times[1] - times[0])).T
    final = expm(generator * (times[2] - times[1])).T
    np.testing.assert_allclose(
        readout.initial, np.diag(facts.source.initial_density).real, atol=1e-8
    )
    np.testing.assert_allclose(readout.transitions[0], transition, atol=1e-7)
    np.testing.assert_allclose(readout.transitions[1], final, atol=1e-7)
    np.testing.assert_allclose(
        prepared.populations, readout.initial @ transition @ final, atol=1e-7
    )
    diagnostics = prepared.projection_diagnostics
    if diagnostics is None:
        raise AssertionError("projective preparation must report histories")
    assert diagnostics.continued_states == ((2, 2),)
    np.testing.assert_array_equal(diagnostics.uncontinued_probability, [[0, 0]])
    assert source.prepare(request, source.snapshot()) is prepared


def test_independent_preparation_without_routing_residency_matches_chain(
    projection_environment: tuple[SimulationEnvironment, SingleFluxoniumSource],
):
    env, reference = projection_environment
    program = repeated_read_program(env, reads=4, reps=1)
    program.acquire(env.soc, progress=False)
    request = env.soc.binding.rounds[0].acquire
    expected = reference.prepare(request, reference.snapshot())
    source = SingleFluxoniumSource(
        reference.settings,
        env.devices,
        runtime=SingleFluxoniumRuntimeOptions(
            engine=Engine(workers=1, routing_cache_bytes=0)
        ),
    )
    try:
        actual = source.prepare(request, source.snapshot())
        assert actual is not expected
        actual_chain = expect_chain(actual.raw.readout)
        expected_chain = expect_chain(expected.raw.readout)
        np.testing.assert_allclose(
            actual_chain.initial, expected_chain.initial, rtol=0, atol=1e-8
        )
        for actual_event, expected_event in zip(
            actual_chain.transitions,
            expected_chain.transitions,
            strict=True,
        ):
            np.testing.assert_allclose(actual_event, expected_event, rtol=0, atol=1e-8)
        if (
            actual.projection_diagnostics is None
            or expected.projection_diagnostics is None
        ):
            raise AssertionError("projective preparations must report histories")
        assert (
            actual.projection_diagnostics.continued_states
            == expected.projection_diagnostics.continued_states
        )
        np.testing.assert_allclose(
            actual.projection_diagnostics.uncontinued_probability,
            expected.projection_diagnostics.uncontinued_probability,
            rtol=0,
            atol=1e-8,
        )
    finally:
        source.close()


def test_cancelled_projective_preparation_can_retry_without_partial_results(
    projection_environment: tuple[SimulationEnvironment, SingleFluxoniumSource],
):
    env, source = projection_environment
    program = repeated_read_program(env, reads=4, reps=1)
    program.acquire(env.soc, progress=False)
    request = env.soc.binding.rounds[0].acquire
    snapshot = source.snapshot()
    cancel = Event()
    cancel.set()
    with pytest.raises(Cancelled):
        source.prepare(request, snapshot, cancel=cancel)
    cancel.clear()
    prepared = source.prepare(request, snapshot, cancel=cancel)
    cancel.set()
    with pytest.raises(Cancelled):
        source.prepare(request, snapshot, cancel=cancel)
    cancel.clear()
    retried = source.prepare(request, snapshot, cancel=cancel)
    retried_chain = expect_chain(retried.raw.readout)
    prepared_chain = expect_chain(prepared.raw.readout)
    np.testing.assert_array_equal(retried_chain.initial, prepared_chain.initial)
    for actual, expected in zip(
        retried_chain.transitions, prepared_chain.transitions, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)


def test_unobserved_terminal_keeps_terminal_state_counts_and_evolution(
    projection_environment: tuple[SimulationEnvironment, SingleFluxoniumSource],
):
    env, source = projection_environment
    program = repeated_read_program(env, reps=1)
    program.acquire(env.soc, progress=False)
    original = env.soc.binding.rounds[0]
    template = original.acquire.templates[0]
    window = template.readouts[0]
    window = replace(
        window,
        trigger_ticks=window.trigger_ticks[:, :1],
        frequency_mhz=window.frequency_mhz[:, :1],
        phase_deg=window.phase_deg[:, :1],
        generator_channels=window.generator_channels[:, :1],
    )
    request = replace(
        original.acquire, templates=(replace(template, readouts=(window,)),)
    )
    prepared = source.prepare(request, source.snapshot())
    assert prepared.channel.conditional.event_indices == (0,)
    batch = prepared.raw.batches[0][1]
    assert batch.events[0][-1].grids == ()
    chain = expect_chain(prepared.raw.readout)
    assert len(chain.transitions) == 1
    np.testing.assert_allclose(
        prepared.populations, chain.initial @ chain.transitions[0], atol=1e-14
    )
    result = source.execute(replace(original, acquire=request))
    if not isinstance(result, IntegratedRound):
        raise AssertionError("integrated acquisition must retain terminal joint counts")
    assert result.statistics.state_counts.shape[-1] <= 3
    assert result.statistics.state_counts.sum() == 1
    assert result.raw.materialize(1, 0)[0].shape[-2] == 1


def test_terminal_only_raw_statistics_and_traces_match_unprojected_inputs(
    projection_environment: tuple[SimulationEnvironment, SingleFluxoniumSource],
):
    env, source = projection_environment
    program = repeated_read_program(env, reads=1, reps=3)
    program.acquire(env.soc, progress=False)
    request = env.soc.binding.rounds[0].acquire
    prepared = source.prepare(request, source.snapshot())
    content = source.plan_template(
        source.snapshot(), request.templates[0], time_base=request.time_base
    ).readout
    model = prepared.raw.batches[0][1].model
    local = local_readout_pulses(content.pulses, model.reference_frequency_ghz)
    baseline = adc.ReadoutBatch(
        model,
        tuple(
            (adc.ReadoutEvent(pulses, grids),)
            for pulses, grids in zip(local, content.grids, strict=True)
        ),
        content.inverse,
    )
    identity = SamplingIdentity(123, (7, 0, 1))
    outputs = []
    for raw in (
        prepared.raw,
        adc.PreparedRawShots(
            adc.ProjectionReadout.unprojected(prepared.populations),
            {0: baseline},
            reps=3,
        ),
    ):
        batch = raw.batches[0][1]
        conditional = adc.ConditionalReadout(model).responses(batch)
        channel = adc.ChannelReadout(
            0,
            conditional,
            batch.content_index.reshape(request.point_dims),
            model.receiver,
        )
        outputs.append(
            (
                raw.bind(identity).materialize(3, 0)[0],
                adc.round_statistics(
                    raw.readout,
                    (channel,),
                    reps=3,
                    seed=123,
                    acquire_sequence=7,
                    round_index=0,
                ),
                adc.sample_traces(raw.readout, {0: batch}, reps=3, identity=identity)
                .channels[0]
                .iq,
            )
        )
    actual, expected = outputs
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1].state_counts, expected[1].state_counts)
    np.testing.assert_array_equal(
        actual[1].channels[0].sums, expected[1].channels[0].sums
    )
    np.testing.assert_array_equal(actual[2], expected[2])


def test_program_herald_wait_terminal_matches_conditional_markov_prediction(
    projection_environment: tuple[SimulationEnvironment, SingleFluxoniumSource],
):
    env, source = projection_environment
    program = repeated_read_program(env)
    program.acquire(env.soc, progress=False)
    request = env.soc.binding.rounds[0]
    prepared = source.prepare(request.acquire, source.snapshot())
    projection = expect_chain(prepared.raw.readout)
    facts = source.model_facts(source.snapshot())
    rate = facts.readout.rates_per_ns
    generator = rate - np.diag(rate.sum(axis=0))
    plan = source.plan_template(
        source.snapshot(),
        request.acquire.templates[0],
        time_base=request.acquire.time_base,
    )
    elapsed = plan.batch.end_ns[0] - plan.projective_reads[0][0].start_ns
    transition = expm(generator * elapsed).T
    weights = projection.initial
    np.testing.assert_allclose(
        weights, np.diag(facts.source.initial_density).real, atol=1e-9
    )
    np.testing.assert_allclose(projection.transitions[0], transition, atol=1e-7)
    result = source.execute(request)
    if not isinstance(result, IntegratedRound):
        raise AssertionError("integrated acquisition must retain joint counts")
    centers = prepared.channel.conditional.sums[0]
    channel = result.statistics.channels[0]
    categories = np.argmin(
        np.linalg.norm(
            channel.conditional_sums[:, :, None, :] - centers[:, None, :, :], axis=-1
        ),
        axis=-1,
    )
    for event, marginal in enumerate((weights, weights @ transition)):
        observed = np.bincount(
            categories[event], weights=result.statistics.state_counts, minlength=2
        )
        np.testing.assert_allclose(
            observed / request.acquire.reps, marginal, atol=0.025
        )
    raw = program.get_raw()
    if raw is None:
        raise AssertionError("completed program must retain physical shots")
    shots = np.asarray(raw[0]).reshape(-1, 2, 2)
    states = np.argmin(
        np.linalg.norm(shots[:, :, None, :] - centers[None], axis=-1), axis=-1
    )
    selected = states[:, 0] == 0
    assert selected.sum() > 500
    assert np.mean(states[selected, 1] == 0) == pytest.approx(
        transition[0, 0], abs=0.03
    )
