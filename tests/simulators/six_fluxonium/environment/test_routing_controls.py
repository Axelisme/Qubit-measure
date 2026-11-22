"""Source routing diagnostics, six-history ownership and builder controls."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.cfg_model import MuxReadoutTone
from zcu_tools.program.v2 import TimelineBuilder
from zcu_tools.program.v2.modules.readout import MuxReadoutCfg
from zcu_tools.simulate.boundary import RoundRequest
from zcu_tools.simulate.evolution import RoutingHistory
from zcu_tools.simulators.six_fluxonium.config import parse_six_fluxonium
from zcu_tools.simulators.six_fluxonium.environment import runtime
from zcu_tools.simulators.six_fluxonium.environment.routing import route_request

from tests.program._profiles import profile_soccfg
from tests.simulators._engine import controlled_engine
from tests.simulators._sources import six_engines as six_engines
from tests.simulators.six_fluxonium.config._data import six_document

pytestmark = [pytest.mark.sim, pytest.mark.usefixtures("six_engines")]


def routing_settings():
    document = six_document()
    for qubit in document["qubits"].values():
        qubit["chip"]["temperature_k"] = 0.05
        qubit["numerics"]["truncation"] = {"native_cutoff": 12, "levels": 2}
        qubit["readout"]["adc"]["noise_psd_adc2_ns"] = 0.01
    return parse_six_fluxonium(document)


def read_program(env):
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
    builder = TimelineBuilder(env.soccfg, reps=8, initial_delay=0, relax_delay=0)
    with builder.repeat("reads", 2):
        builder.emit(readout.build("readout"))
        builder.timeline_inc(0.096)
    builder.delay_by_timeline()
    return builder.build()


def test_source_passes_cavity_spans_and_honors_engine_reuse_answer():
    config = routing_settings()
    history = RoutingHistory()
    env = runtime.six_fluxonium(config, soccfg=profile_soccfg("six_fluxonium"))
    try:
        source = runtime.SixFluxoniumSource(
            config, env.devices, history=history, families={"t4"}
        )
        try:
            program = read_program(env)
            program.acquire(env.soc, progress=False)
            request = env.soc.binding.rounds[0].acquire
            source.execute(RoundRequest(request, source.snapshot(), 0))
            groups = route_request(config, request)
            assert len(groups) == 2
            for group in groups:
                single = source.sources[group.cavity]
                subrequest = replace(
                    request,
                    templates=group.templates,
                    readouts=tuple(
                        window
                        for window in request.readouts
                        if window.channel in group.channels
                    ),
                )
                snapshot = single.snapshot()
                first = single.prepare(subrequest, snapshot)
                assert len(controlled_engine(single.engine).calls) >= 2
                assert all(
                    model.history is history
                    for model, _ in controlled_engine(single.engine).calls
                )
                diagnostics = first.projection_diagnostics
                assert diagnostics is not None
                assert diagnostics.evolution_calls >= 2
                assert single.prepare(subrequest, snapshot) is first
                controlled_engine(single.engine).reuse_allowed = False
                resolved = single.prepare(subrequest, snapshot)
                assert resolved is not first
                assert (
                    resolved.paths.segments_by_kernel == first.paths.segments_by_kernel
                )
        finally:
            source.close()
    finally:
        if env.close_backend is not None:
            env.close_backend()
        env.devices.close_all_devices()


def test_builder_rejects_unknown_family_before_acquire():
    with pytest.raises(ValueError, match="unknown kernel families"):
        runtime.six_fluxonium(
            routing_settings(),
            soccfg=profile_soccfg("six_fluxonium"),
            families={"unknown"},
        )


def test_acquire_passes_family_policy_and_propagates_engine_failure(six_engines):
    families = {"dop853"}
    env = runtime.six_fluxonium(
        routing_settings(),
        soccfg=profile_soccfg("six_fluxonium"),
        families=families,
    )
    try:
        error = RuntimeError("numeric backend rejected fixture")
        for engine in six_engines:
            assert engine.requested_families == families
            engine.error = error
        with pytest.raises(
            RuntimeError, match="numeric backend rejected fixture"
        ) as caught:
            read_program(env).acquire(env.soc, progress=False)
        assert caught.value is error
    finally:
        if env.close_backend is not None:
            env.close_backend()
        env.devices.close_all_devices()


@pytest.mark.parametrize("borrowed", [False, True])
def test_source_keeps_six_owned_histories_or_borrows_one_without_clearing_on_close(
    borrowed,
):
    config = routing_settings()
    shared = RoutingHistory() if borrowed else None
    env = runtime.six_fluxonium(config, soccfg=profile_soccfg("six_fluxonium"))
    try:
        source = runtime.SixFluxoniumSource(config, env.devices, history=shared)
        try:
            assert len(source.sources) == 6
            histories = tuple(
                single.model_facts(single.snapshot()).source.history
                for single in source.sources
            )
            if borrowed:
                assert all(history is shared for history in histories)
            else:
                assert len({id(history) for history in histories}) == 6
            generations = tuple(history.generation for history in histories)
            source.close()
            assert tuple(history.generation for history in histories) == tuple(
                generation if borrowed else generation + 1 for generation in generations
            )
        finally:
            source.close()
    finally:
        if env.close_backend is not None:
            env.close_backend()
        env.devices.close_all_devices()


def test_builder_close_preserves_borrowed_history():
    history = RoutingHistory()
    env = runtime.six_fluxonium(
        routing_settings(),
        soccfg=profile_soccfg("six_fluxonium"),
        history=history,
    )
    try:
        read_program(env).acquire(env.soc, progress=False)
        generation = history.generation
    finally:
        if env.close_backend is not None:
            env.close_backend()
        env.devices.close_all_devices()
    assert history.generation == generation


@pytest.mark.parametrize("mode", ["integrated", "decimated"])
def test_history_clear_preserves_complete_same_seed_cavity_iq_sequence(mode):
    cleared_history, retained_history = RoutingHistory(), RoutingHistory()
    config = routing_settings()
    cleared = runtime.six_fluxonium(
        config,
        soccfg=profile_soccfg("six_fluxonium"),
        history=cleared_history,
    )
    retained = runtime.six_fluxonium(
        config,
        soccfg=profile_soccfg("six_fluxonium"),
        history=retained_history,
    )
    try:
        cleared_program, retained_program = (
            read_program(cleared),
            read_program(retained),
        )
        for index in range(4):
            if index == 2:
                cleared_history.clear()
            if mode == "integrated":
                actual = cleared_program.acquire(cleared.soc, progress=False)
                expected = retained_program.acquire(retained.soc, progress=False)
            else:
                actual = cleared_program.acquire_decimated(cleared.soc, progress=False)
                expected = retained_program.acquire_decimated(
                    retained.soc, progress=False
                )
            for actual_iq, expected_iq in zip(actual, expected, strict=True):
                np.testing.assert_array_equal(actual_iq, expected_iq)
            actual_raw, expected_raw = (
                cleared_program.get_raw(),
                retained_program.get_raw(),
            )
            assert actual_raw is not None and expected_raw is not None
            for actual_iq, expected_iq in zip(actual_raw, expected_raw, strict=True):
                np.testing.assert_array_equal(actual_iq, expected_iq)
    finally:
        for env in (cleared, retained):
            if env.close_backend is not None:
                env.close_backend()
            env.devices.close_all_devices()
