"""Mux frontend emits all coherent carriers and preserves explicit ADC order."""

import numpy as np
import pytest
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.program.v2 import ModularProgramV2, ProgramV2Cfg
from zcu_tools.program.v2.modules.control import Repeat
from zcu_tools.program.v2.simulation.timeline import TimelineInterpreter

from tests.program.v2._mux_support import mux_soccfg, native_mux


@pytest.mark.parametrize("placement", ["direct", "repeat"])
def test_mux_frontend_agrees_with_scalar_timing_and_declared_tone_order(placement):
    mux = native_mux().build("mux_probe")
    module = Repeat("container", 1).add_content(mux) if placement == "repeat" else mux
    prog = ModularProgramV2(mux_soccfg(), ProgramV2Cfg(reps=2), modules=[module])
    env = white_noise(std_adc=0, n_gens=2, n_readouts=9, routes=((0, 4), (0, 1)))
    try:
        try:
            prog.acquire(env.soc, progress=False)
        except ValueError as error:
            pytest.fail(f"mux must lower to coherent carriers and ADC windows: {error}")
        request = env.soc.binding.last_request
        assert request is not None
        point = request.materialize_point(0)
        scalar = TimelineInterpreter(prog).run_point({})
        assert point == scalar
        count = 1
        assert len(point.events) == count * 2
        assert tuple(w.channel for w in point.readouts) == (4, 1)
        for window in point.readouts:
            assert len(window.trigger_us) == count
        for start in range(0, len(point.events), 2):
            first, second = point.events[start : start + 2]
            assert first.gen_ch == second.gen_ch == 0
            assert first.start_us == second.start_us
            assert first.duration_us == second.duration_us
            assert first.frequency_mhz == pytest.approx(1000, abs=1e-5)
            assert second.frequency_mhz == pytest.approx(1080, abs=1e-5)
            assert first.phase_deg == 0
            assert second.phase_deg == pytest.approx(45, abs=1e-5)
            assert first.gain == pytest.approx(0.1, abs=1e-4)
        raw = prog.get_raw()
        assert raw is not None
        assert len(raw) == 2
        np.testing.assert_array_equal(raw[0], raw[1])
    finally:
        env.devices.close_all_devices()


@pytest.mark.parametrize("channels", [(4, 1), (1,)])
def test_fixed_mux_compile_profile_runs_real_mux_with_every_adc(channels):
    from zcu_tools.program.v2 import mocksoc

    factory = getattr(mocksoc, "make_mux_mock_soccfg", None)
    assert factory is not None, "offline mux acquisition needs a fixed compile profile"
    soccfg = factory(3, 5, gen_ch=0, ro_chs=channels)
    prog = ModularProgramV2(
        soccfg, ProgramV2Cfg(reps=3), modules=[native_mux(channels).build("mux")]
    )
    env = white_noise(n_gens=3, n_readouts=5, routes=tuple((0, ch) for ch in channels))
    try:
        prog.acquire_decimated(env.soc, progress=False)
        raw = prog.get_raw()
        assert raw is not None
        assert len(raw) == len(channels)
        assert all(shots.shape[-1] == 2 for shots in raw)
    finally:
        env.devices.close_all_devices()
