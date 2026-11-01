"""Request-clock interpretation at public physical planning/acquisition seams."""

from dataclasses import replace

import numpy as np
from zcu_tools.core.simulation import AcquireRequest, RoundRequest, RoundSignals
from zcu_tools.experiment.single_fluxonium.runtime import (
    SingleFluxoniumSource,
    single_fluxonium,
)
from zcu_tools.program.v2.simulation.compiled import build_compiled_request

from tests.core._simulation import TIME_BASE
from tests.program._profiles import profile_soccfg
from tests.program.v2.simulation._compiled_support import clock_invariance_program

from ._inputs import settings, template


def test_source_plan_and_preparation_reinterpret_ticks_when_clock_changes():
    env = single_fluxonium(settings(), soccfg=profile_soccfg("single_fluxonium"))
    source = SingleFluxoniumSource(settings(), env.devices)
    try:
        snapshot = source.snapshot()
        compiled = replace(template(), points=np.array([0, 1], dtype=np.int64))
        faster = replace(TIME_BASE, ticks_per_us=TIME_BASE.ticks_per_us * 2)
        original = source.plan_template(snapshot, compiled, time_base=TIME_BASE)
        changed = source.plan_template(snapshot, compiled, time_base=faster)
        np.testing.assert_array_equal(changed.batch.end_ns, original.batch.end_ns / 2)
        request = AcquireRequest(
            "integrated",
            (1, 2),
            0,
            (),
            123,
            0,
            "test",
            time_base=TIME_BASE,
            point_dims=(2,),
            point_names=("point",),
            templates=(compiled,),
        )
        # preflight requires the same declared receiver set as the templates.
        from zcu_tools.core.simulation import ReadoutWindow

        request = replace(request, readouts=(ReadoutWindow(0, 0, 4, 1344, (24192,)),))
        a = source.prepare(request, snapshot)
        b = source.prepare(replace(request, time_base=faster), snapshot)
        assert not np.array_equal(
            a.channel.conditional.sums, b.channel.conditional.sums
        )
    finally:
        source.close()
        if env.close_backend is not None:
            env.close_backend()
        env.devices.close_all_devices()


def test_extra_declared_clock_preserves_original_channel_physical_iq_bits():
    env = single_fluxonium(settings(), soccfg=profile_soccfg("single_fluxonium"))
    env.soccfg["gens"][1]["f_fabric"] = 512.0

    def simulate(extra):
        program = clock_invariance_program(env.soccfg, extra_generator=extra)
        request = build_compiled_request(
            program, "decimated", env.soc.binding, 7, routes=env.soc.routes
        )
        env.soc.install_request(request)
        try:
            result = env.soc.binding.execute(
                RoundRequest(request, env.soc.binding.capture_snapshot(), 0)
            )
            if not isinstance(result, RoundSignals):
                raise AssertionError(
                    "physical decimated acquisition returns RoundSignals"
                )
            assert tuple(channel.channel for channel in result.channels) == (0,)
            return request, result.channels[0].iq.copy()
        finally:
            env.soc.release_request()

    try:
        original, a = simulate(False)
        expanded, b = simulate(True)
        assert original.time_base.ticks_per_us == 43008
        assert expanded.time_base.ticks_per_us == 172032
        assert {event.gen_ch for event in expanded.templates[0].events} == {0}
        assert np.linalg.norm(a) > 0
        np.testing.assert_array_equal(a.view(np.uint64), b.view(np.uint64))
    finally:
        if env.close_backend is not None:
            env.close_backend()
        env.devices.close_all_devices()
