"""Source routing diagnostics and builder-visible acquisition controls."""

import numpy as np
import pytest
from zcu_tools.program.v2.modules.delay import SoftDelay
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.simulate.evolution import RoutingHistory
from zcu_tools.simulators.single_fluxonium.config import parse_single_fluxonium
from zcu_tools.simulators.single_fluxonium.environment import runtime

from tests.program._profiles import profile_soccfg
from tests.program._timeline import compile_modules
from tests.simulators._engine import EngineDouble, controlled_engine
from tests.simulators._sources import single_environment
from tests.simulators.single_fluxonium.environment._inputs import settings

pytestmark = pytest.mark.sim


def routing_settings():
    values = settings().model_dump(mode="json")
    values["chip"]["temperature_k"] = 0.05
    values["numerics"]["truncation"]["levels"] = 2
    values["numerics"]["time_scale"] = 0
    values["readout"]["adc"]["noise_psd_adc2_ns"] = [[0.01, 0], [0, 0.01]]
    return parse_single_fluxonium(values)


def read_program(env, *, post_herald_drive=False):
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
    drive = (
        PulseCfg(
            ch=1,
            nqz=1,
            freq=1000,
            gain=0.1,
            waveform=ConstWaveformCfg(length=0.008),
        ).build("drive")
        if post_herald_drive
        else None
    )
    return compile_modules(
        env.soccfg,
        modules=[
            SoftDelay("prefix", 0.008),
            readout.build("herald"),
            drive,
            SoftDelay("wait", 0.024),
            readout.build("terminal"),
        ],
        reps=8,
        initial_delay=0,
        relax_delay=0,
    )


def test_source_passes_projective_spans_and_honors_engine_reuse_answer():
    config = routing_settings()
    history = RoutingHistory()
    env = single_environment(config, soccfg=profile_soccfg("single_fluxonium"))
    try:
        source = runtime.SingleFluxoniumSource(
            config,
            env.devices,
            runtime=runtime.SingleFluxoniumRuntimeOptions(
                engine=EngineDouble(workers=1, families={"t4"}), history=history
            ),
        )
        try:
            program = read_program(env)
            program.acquire(env.soc, progress=False)
            request = env.soc.binding.rounds[0].acquire
            snapshot = source.snapshot()
            first = source.prepare(request, snapshot)
            assert len(controlled_engine(source.engine).calls) >= 2
            assert all(
                model.history is history
                for model, _ in controlled_engine(source.engine).calls
            )
            diagnostics = first.projection_diagnostics
            assert diagnostics is not None
            assert diagnostics.evolution_calls >= 2
            assert source.prepare(request, snapshot) is first
            controlled_engine(source.engine).reuse_allowed = False
            resolved = source.prepare(request, snapshot)
            assert resolved is not first
            assert resolved.paths.segments_by_kernel == first.paths.segments_by_kernel
        finally:
            source.close()
    finally:
        if env.close_backend is not None:
            env.close_backend()
        env.devices.close_all_devices()


def test_builder_close_preserves_borrowed_history():
    history = RoutingHistory()
    env = single_environment(
        routing_settings(),
        soccfg=profile_soccfg("single_fluxonium"),
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


def test_builder_rejects_unknown_family_before_acquire():
    with pytest.raises(ValueError, match="unknown kernel families"):
        runtime.single_fluxonium(
            routing_settings(),
            soccfg=profile_soccfg("single_fluxonium"),
            families={"unknown"},
        )


@pytest.mark.parametrize(
    "post_herald_drive, families, domain",
    [
        (False, {"dop853"}, "idle"),
        (True, {"t4"}, "plateau"),
    ],
)
def test_acquire_propagates_engine_failure(
    post_herald_drive,
    families,
    domain,
):
    engine = EngineDouble()
    error = RuntimeError("numeric backend rejected fixture")
    engine.error = error
    env = single_environment(
        routing_settings(),
        engine=engine,
        soccfg=profile_soccfg("single_fluxonium"),
        families=families,
    )
    try:
        program = read_program(env, post_herald_drive=post_herald_drive)
        with pytest.raises(
            RuntimeError, match="numeric backend rejected fixture"
        ) as caught:
            program.acquire(env.soc, progress=False)
        assert caught.value is error
    finally:
        if env.close_backend is not None:
            env.close_backend()
        env.devices.close_all_devices()


@pytest.mark.parametrize("mode", ["integrated", "decimated"])
def test_history_clear_preserves_complete_same_seed_acquire_iq_sequence(mode):
    cleared_history, retained_history = RoutingHistory(), RoutingHistory()
    config = routing_settings()
    cleared = single_environment(
        config,
        soccfg=profile_soccfg("single_fluxonium"),
        history=cleared_history,
    )
    retained = single_environment(
        config,
        soccfg=profile_soccfg("single_fluxonium"),
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
