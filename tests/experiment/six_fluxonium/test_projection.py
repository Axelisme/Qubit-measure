"""Independent cavity histories through the public source and program seams.

Terminal-only numeric pins retire when the versioned ADC/RNG contract changes.
The reference was captured before projective routing and multi-ADC preparation.
"""

from collections.abc import Generator
from dataclasses import replace
from hashlib import sha256
from threading import Event

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.linalg import expm
from zcu_tools.cfg_model import MuxReadoutTone
from zcu_tools.core.simulation import (
    AcquireRequest,
    ChargeDriveRequirement,
    PointTimeline,
    PulseEvent,
    ReadoutWindow,
    RoundRequest,
    RoundSignals,
)
from zcu_tools.experiment.simulation import SimulationEnvironment
from zcu_tools.experiment.six_fluxonium import six_fluxonium
from zcu_tools.experiment.six_fluxonium.routing import route_request
from zcu_tools.experiment.six_fluxonium.runtime import SixFluxoniumSource
from zcu_tools.program.v2 import TimelineBuilder
from zcu_tools.program.v2.modules.readout import MuxReadoutCfg
from zcu_tools.program.v2.simulation import IntegratedRound
from zcu_tools.resources.six_fluxonium import parse_six_fluxonium

from tests.core._simulation import TIME_BASE, single_point_template, tick_tuple, ticks
from tests.program._profiles import profile_soccfg
from tests.resources._six_fluxonium_data import six_document

pytestmark = pytest.mark.sim


def test_routing_rejects_unsupported_charge_requirements_before_preparation():
    settings = parse_six_fluxonium(six_document())
    window = ReadoutWindow(
        0,
        0,
        4,
        TIME_BASE.period_ticks(1000),
        tick_tuple((0.018,)),
        frequency_mhz=(6800,),
        phase_deg=(0,),
        generator_channels=(0,),
    )
    timeline = PointTimeline(
        (),
        (PulseEvent(0, ticks(0.016), ticks(0.008), 6800, 0, 0.1),),
        (window,),
        ticks(0.024),
        time_base=TIME_BASE,
    )
    ordinary = AcquireRequest(
        "integrated",
        (1,),
        0,
        (window,),
        123,
        0,
        "test",
        templates=(single_point_template(timeline),),
        time_base=TIME_BASE,
    )
    assert len(route_request(settings, ordinary)) == 1
    requested = replace(
        ordinary, charge_requirements=(ChargeDriveRequirement("control", 1),)
    )
    with pytest.raises(ValueError, match="six-fluxonium.*charge.*not supported"):
        route_request(settings, requested)


@pytest.fixture
def projection_environment() -> Generator[
    tuple[SimulationEnvironment, SixFluxoniumSource]
]:
    document = six_document()
    for qubit in document["qubits"].values():
        qubit["dc_bias"]["offset"] = -0.5 * qubit["dc_bias"]["period"]
        qubit["dc_bias"]["coil_offset"] = 0
        qubit["chip"]["temperature_k"] = 0.05
        qubit["chip"]["qubit"]["q_cap"] = 30000
        qubit["numerics"]["truncation"] = {"native_cutoff": 12, "levels": 2}
        qubit["numerics"]["projection_prune_floor"] = 0
        qubit["readout"]["adc"]["gain"] = 1e6
        qubit["readout"]["resonator"]["charge_coefficient_ghz"] = 0.03
    settings = parse_six_fluxonium(document)
    env = six_fluxonium(settings, soccfg=profile_soccfg("six_fluxonium"))
    source = SixFluxoniumSource(settings, env.devices)
    try:
        yield env, source
    finally:
        source.close()
        if env.close_backend is not None:
            env.close_backend()
        env.devices.close_all_devices()


def repeated_mux_program(env: SimulationEnvironment, *, reads=2, reps=4096):
    readout = MuxReadoutCfg(
        gen_ch=0,
        nqz=1,
        mixer_freq=6900,
        length=0.064,
        ro_length=0.026,
        trig_offset=0.008,
        tones=[
            MuxReadoutTone(ro_ch=ch, freq=6800 + 125 * ch, gain=0.1) for ch in (0, 1)
        ],
    )
    module = readout.build("readout")
    builder = TimelineBuilder(env.soccfg, reps=reps, initial_delay=0, relax_delay=0)
    with builder.repeat("reads", reads):
        builder.emit(module)
        builder.timeline_inc(0.096)
    builder.delay_by_timeline()
    program = builder.build()
    program.set_reads_per_shot(reads)
    return program


def test_two_cavity_heralds_keep_conditional_and_independent_shot_histories(
    projection_environment,
):
    env, source = projection_environment
    program = repeated_mux_program(env)
    program.acquire(env.soc, progress=False)
    request = env.soc.binding.rounds[0].acquire
    result = source.execute(RoundRequest(request, source.snapshot(), 0))
    if not isinstance(result, IntegratedRound):
        raise AssertionError("integrated source must deliver retained raw shots")
    raw = program.get_raw()
    if raw is None:
        raise AssertionError("program must retain both cavity shot streams")
    selected_ground = []
    for group in route_request(source.settings, request):
        single = source.sources[group.cavity]
        subrequest = replace(
            request,
            templates=group.templates,
            readouts=tuple(w for w in request.readouts if w.channel in group.channels),
        )
        prepared = single.prepare(subrequest, single.snapshot())
        facts = single.model_facts(single.snapshot())
        plan = single.plan_template(
            single.snapshot(), group.templates[0], time_base=request.time_base
        )
        rate = facts.readout.rates_per_ns
        elapsed = plan.batch.end_ns[0] - plan.projective_reads[0][0].start_ns
        transition = expm((rate - np.diag(rate.sum(axis=0))) * elapsed).T
        expected = np.real(np.diag(facts.source.initial_density))[:, None] * transition
        np.testing.assert_allclose(
            prepared.raw.readout.weights.reshape(-1, 2)[0],
            expected.sum(axis=1),
            atol=1e-8,
        )
        centers = prepared.channel.conditional.sums[0]
        shots = np.asarray(raw[group.cavity]).reshape(-1, 2, 2)
        states = np.argmin(
            np.linalg.norm(shots[:, :, None, :] - centers[None], axis=-1), axis=-1
        )
        ground = states[:, 0] == 0
        selected_ground.append(ground)
        assert 500 < ground.sum() < request.reps - 500
        assert np.mean(states[ground, 1] == 0) == pytest.approx(
            transition[0, 0], abs=0.035
        )
        assert np.mean(states[~ground, 1] == 1) == pytest.approx(
            transition[1, 1], abs=0.035
        )
        channel = next(
            c for c in result.statistics.channels if c.channel == group.channels[0]
        )
        np.testing.assert_allclose(
            channel.sums / request.reps, shots.mean(axis=0), atol=1000, rtol=0.03
        )
        bins = np.argmin(
            np.linalg.norm(
                channel.conditional_sums[:, :, None, :] - centers[:, None, :, :],
                axis=-1,
            ),
            axis=-1,
        )
        observed_joint = np.zeros((2, 2), dtype=np.int64)
        np.add.at(observed_joint, (bins[0], bins[1]), result.statistics.state_counts)
        np.testing.assert_allclose(observed_joint / request.reps, expected, atol=0.025)
    first, second = selected_ground
    assert np.mean(first & second) == pytest.approx(
        np.mean(first) * np.mean(second), abs=0.025
    )
    assert np.count_nonzero(first != second) > 500


def shared_cavity_request(env: SimulationEnvironment):
    program = repeated_mux_program(env, reps=256)
    program.acquire(env.soc, progress=False)
    request = env.soc.binding.rounds[0].acquire
    template = request.templates[0]
    first, second = template.readouts
    terminal = replace(
        second,
        trigger_ticks=second.trigger_ticks[:, 1:],
        frequency_mhz=first.frequency_mhz[:, 1:],
        phase_deg=second.phase_deg[:, 1:],
        generator_channels=second.generator_channels[:, 1:],
    )
    events = tuple(
        replace(event, frequency_mhz=np.full_like(event.frequency_mhz, 6800))
        for event in template.events
    )
    return replace(
        request,
        templates=(replace(template, events=events, readouts=(first, terminal)),),
    )


def test_partial_adc_observation_shares_terminal_paths_in_raw_integrated_and_decimated(
    projection_environment,
):
    env, source = projection_environment
    request = shared_cavity_request(env)
    result = source.execute(RoundRequest(request, source.snapshot(), 0))
    if not isinstance(result, IntegratedRound):
        raise AssertionError("integrated source must retain statistics")
    raw = result.raw.materialize(request.reps, 0)
    assert raw[0].shape[-2] == 2
    assert raw[1].shape[-2] == 1
    np.testing.assert_array_equal(raw[0][..., 1, :], raw[1][..., 0, :])
    first, second = result.statistics.channels
    # Integrated draws add independent per-ADC nearest-integer rounding noise.
    np.testing.assert_allclose(
        first.sums[..., 1, :],
        second.sums[..., 0, :],
        atol=6 * np.sqrt(request.reps / 6),
        rtol=0,
    )
    np.testing.assert_array_equal(
        first.conditional_sums[..., 1, :, :], second.conditional_sums[..., 0, :, :]
    )
    traces = source.execute(
        RoundRequest(replace(request, mode="decimated"), source.snapshot(), 0)
    )
    if not isinstance(traces, RoundSignals):
        raise AssertionError("decimated source must return physical traces")
    single = source.sources[0]
    validated = single.preflight(
        request,
        single.snapshot(),
        receivers=tuple(
            single.settings.readout.adc.model_copy(update={"ro_ch": ch})
            for ch in (0, 1)
        ),
    )
    prepared = single.prepare_readouts(validated)
    assert tuple(
        channel.conditional.event_indices for channel in prepared.channels
    ) == ((0, 1), (1,))
    assert len(prepared.raw.readout.populations) == 2
    a, b = traces.channels
    assert a.iq.shape[-3] == 2 and b.iq.shape[-3] == 1
    np.testing.assert_array_equal(a.iq[..., 1, :, :], b.iq[..., 0, :, :])


@pytest.mark.parametrize("same_cavity", [True, False])
def test_all_sources_preflight_second_adc_template_layout_before_any_evolution(
    projection_environment, same_cavity
):
    env, source = projection_environment
    program = repeated_mux_program(env, reps=1)
    program.acquire(env.soc, progress=False)
    request = env.soc.binding.rounds[0].acquire
    original = request.templates[0]
    if same_cavity:
        original = replace(
            original,
            events=tuple(
                replace(event, frequency_mhz=np.full_like(event.frequency_mhz, 6800))
                for event in original.events
            ),
            readouts=tuple(
                replace(window, frequency_mhz=np.full_like(window.frequency_mhz, 6800))
                for window in original.readouts
            ),
        )
    first, second = original.readouts
    conflict = replace(
        second,
        trigger_ticks=np.column_stack(
            (
                second.trigger_ticks[:, 0],
                second.trigger_ticks[:, 0] + 3 * request.time_base.period_ticks(384.0),
            )
        ),
    )
    other = replace(
        original, points=np.array([1], dtype=np.int64), readouts=(first, conflict)
    )
    request = replace(
        request,
        point_dims=(2,),
        point_names=("point",),
        loop_dims=(1, 2),
        templates=(original, other),
    )
    cancelled = Event()
    cancelled.set()
    # Evolution checks this public cancellation query. Layout rejection must win
    # even when a different physical source precedes the conflicting ADC.
    with pytest.raises(ValueError, match=r"ADC 1: template 1.*layout.*template 0"):
        source.execute(RoundRequest(request, source.snapshot(), 0, cancel=cancelled))


def test_observed_cavity_requires_terminal_tone_even_with_a_herald(
    projection_environment,
):
    env, source = projection_environment
    program = repeated_mux_program(env, reps=1)
    program.acquire(env.soc, progress=False)
    request = env.soc.binding.rounds[0].acquire
    original = request.templates[0]
    latest = max(event.start_ticks[0] for event in original.events)
    events = tuple(
        event
        for event in original.events
        if not (event.start_ticks[0] == latest and event.frequency_mhz[0] > 6900)
    )
    request = replace(request, templates=(replace(original, events=events),))
    with pytest.raises(ValueError, match="incident terminal tone"):
        route_request(source.settings, request)


def test_terminal_only_native_integrated_and_decimated_match_fixed_seed_reference(
    projection_environment,
):
    """Keep exact outputs for native ceil-quantized mux ADC windows."""
    env, source = projection_environment
    program = repeated_mux_program(env, reads=1, reps=3)
    program.acquire(env.soc, progress=False)
    request = env.soc.binding.rounds[0].acquire
    result = source.execute(RoundRequest(request, source.snapshot(), 0))
    if not isinstance(result, IntegratedRound):
        raise AssertionError("integrated request must return statistics")
    traces = source.execute(
        RoundRequest(replace(request, mode="decimated"), source.snapshot(), 0)
    )
    if not isinstance(traces, RoundSignals):
        raise AssertionError("decimated request must return traces")
    arrays: dict[str, NDArray[np.float64] | NDArray[np.int64]] = {
        "counts": result.statistics.state_counts
    }
    for channel in result.statistics.channels:
        arrays[f"sums{channel.channel}"] = channel.sums
        arrays[f"conditional{channel.channel}"] = channel.conditional_sums
    for channel, values in result.raw.materialize(3, 0).items():
        arrays[f"raw{channel}"] = values
    for channel in traces.channels:
        arrays[f"traces{channel.channel}"] = channel.iq
    assert {
        name: sha256(values.tobytes()).hexdigest() for name, values in arrays.items()
    } == {
        "counts": "2061cec4721ab2aba8dbf9bd4f7ff16887ca87713b79563650cb52fcd1da75bc",
        "sums0": "c510624582ff4b4b6723d0ede230c1023d19a145fbb8dec541ef13a31cda9e77",
        "conditional0": "1c5e77fbde464dc76e7ab13af4ba6c6dcfaea10ec553a1cb7f57e560c9e19dd6",
        "sums1": "539a7d6b87b498fcdfe216a91235669c36ceb7a0da3ad40ac8470f4ec67ff671",
        "conditional1": "e7ace828750b3a3a3270e91f12144ba86ee77794b5bf0c88debd96a5bee460b4",
        "raw0": "b92e0950ba75fdd126b6ac455a2000c02e928a6db5b395f4942ee8bcc122afaf",
        "raw1": "cc1f7295a55f19ea896a7fd00f814e640300fbab671132d1c2005aeadda71993",
        "traces0": "f4fd3d34ddba2c42e8c32a5e8ef23ba914afd16ec3d16cd865555fd8f4c76ba6",
        "traces1": "6216e8b2d0ee0f2648576eb4a0b169df1931b864cca449415c9ba4cfe138ee53",
    }


@pytest.mark.parametrize("mode", ["integrated", "decimated"])
def test_different_cavity_event_topologies_are_legal(projection_environment, mode):
    env, source = projection_environment
    program = repeated_mux_program(env, reps=16)
    program.acquire(env.soc, progress=False)
    request = env.soc.binding.rounds[0].acquire
    original = request.templates[0]
    first, second = original.readouts
    terminal = replace(
        second,
        trigger_ticks=second.trigger_ticks[:, 1:],
        frequency_mhz=second.frequency_mhz[:, 1:],
        phase_deg=second.phase_deg[:, 1:],
        generator_channels=second.generator_channels[:, 1:],
    )
    # Cavity 1 has no herald pulse or trigger. Cavity 0 still projects once.
    earliest = min(event.start_ticks[0] for event in original.events)
    events = tuple(
        event
        for event in original.events
        if not (event.start_ticks[0] == earliest and event.frequency_mhz[0] > 6900)
    )
    request = replace(
        request,
        mode=mode,
        templates=(replace(original, events=events, readouts=(first, terminal)),),
    )
    result = source.execute(RoundRequest(request, source.snapshot(), 0))
    if isinstance(result, IntegratedRound):
        raw = result.raw.materialize(request.reps, 0)
        assert raw[0].shape[-2] == 2 and raw[1].shape[-2] == 1
        assert result.statistics.state_counts.sum() == request.reps
        assert tuple(
            channel.sums.shape[-2] for channel in result.statistics.channels
        ) == (2, 1)
    else:
        assert tuple(channel.iq.shape[-3] for channel in result.channels) == (2, 1)
