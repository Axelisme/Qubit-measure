"""Trusted environment authors can use ordinary Python dataclasses."""

from dataclasses import replace

import pytest
from zcu_tools.experiment.simulation import load_environment, white_noise

from tests.program._profiles import load_soccfg


def test_environment_builder_can_resolve_its_own_postponed_annotations(tmp_path):
    path = tmp_path / "authored.py"
    path.write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass, replace\n"
        "from zcu_tools.experiment.simulation import white_noise\n"
        "from tests.program._profiles import load_soccfg\n"
        "@dataclass\n"
        "class Settings:\n"
        "    label: str = 'authored environment'\n"
        "def build_environment():\n"
        "    return replace(white_noise(load_soccfg(), std_adc=1.0, seed=None, "
        "routes=((0, 0),)), description=Settings().label)\n"
    )
    environment = load_environment(path)
    try:
        assert environment.description == "authored environment"
    finally:
        environment.devices.close_all_devices()


def test_environment_bundle_rejects_a_different_compile_profile():
    environment = white_noise(load_soccfg(), std_adc=1.0, seed=None, routes=((0, 0),))
    try:
        with pytest.raises(ValueError, match="matching|match"):
            replace(environment, soccfg=load_soccfg(n_gens=3))
    finally:
        environment.devices.close_all_devices()


@pytest.mark.parametrize("gen_ch", range(3))
@pytest.mark.parametrize("ro_ch", range(2))
def test_white_noise_all_routes_acquires_on_any_generator_and_adc(gen_ch, ro_ch):
    import numpy as np
    from zcu_tools.program.v2 import TimelineBuilder
    from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
    from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg

    env = white_noise(
        load_soccfg(n_gens=3, n_readouts=2), std_adc=0, seed=17, routes="all"
    )
    try:
        builder = TimelineBuilder(env.soccfg, reps=2)
        builder.emit(
            PulseReadoutCfg(
                gen_ch=gen_ch,
                ro_ch=ro_ch,
                nqz=1,
                freq=100.0,
                gain=0.1,
                waveform=ConstWaveformCfg(length=0.1),
                ro_length=0.1,
            ).build("readout")
        )
        prog = builder.build()
        averaged = prog.acquire(env.soc, progress=False)
        np.testing.assert_array_equal(averaged[0], np.zeros((1, 2)))
    finally:
        env.devices.close_all_devices()


def test_white_noise_integrated_round_is_compact_and_decimated_remains_signals():
    import numpy as np
    from zcu_tools.core.simulation import (
        AcquireRequest,
        ReadoutWindow,
        RoundRequest,
        RoundSignals,
        TimeBase,
    )
    from zcu_tools.program.v2.simulation import IntegratedRound

    env = white_noise(load_soccfg(), std_adc=3, seed=17, routes=((0, 0),))
    try:
        time_base = TimeBase.from_clocks(
            (1000.0, env.soccfg["readouts"][0]["f_output"])
        )
        request = AcquireRequest(
            mode="integrated",
            loop_dims=(100,),
            avg_level=0,
            reps=100,
            readouts=(
                ReadoutWindow(
                    0,
                    0,
                    8,
                    time_base.period_ticks(env.soccfg["readouts"][0]["f_output"]),
                    (0,),
                ),
            ),
            root_seed=17,
            sequence=0,
            rng_version=env.soc.binding.rng_version,
            time_base=time_base,
        )
        delivery = env.soc.binding.execute(
            RoundRequest(request, env.soc.binding.capture_snapshot(), 0)
        )
        assert isinstance(delivery, IntegratedRound)
        result = delivery.statistics
        np.testing.assert_array_equal(result.state_counts, [100])
        assert result.channels[0].sums.shape == (1, 2)
        np.testing.assert_array_equal(result.channels[0].conditional_sums, 0)
        np.testing.assert_array_equal(
            result.channels[0].shot_covariance,
            np.eye(2) * 9 * request.readouts[0].samples,
        )
        assert result.raw_identity.stream == (request.sequence, 0, 1)
        request = replace(request, mode="decimated")
        traces = env.soc.binding.execute(
            RoundRequest(request, env.soc.binding.capture_snapshot(), 0)
        )
        assert isinstance(traces, RoundSignals)
        assert traces.channels[0].iq.shape == (100, 1, request.readouts[0].samples, 2)
    finally:
        env.devices.close_all_devices()
