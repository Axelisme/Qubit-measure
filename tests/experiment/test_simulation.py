"""Trusted environment authors can use ordinary Python dataclasses."""

from dataclasses import replace

import pytest
from zcu_tools.experiment.simulation import load_environment, white_noise
from zcu_tools.program.v2.mocksoc import make_mock_soccfg


def test_environment_builder_can_resolve_its_own_postponed_annotations(tmp_path):
    path = tmp_path / "authored.py"
    path.write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass, replace\n"
        "from zcu_tools.experiment.simulation import white_noise\n"
        "@dataclass\n"
        "class Settings:\n"
        "    label: str = 'authored environment'\n"
        "def build_environment():\n"
        "    return replace(white_noise(), description=Settings().label)\n"
    )
    environment = load_environment(path)
    try:
        assert environment.description == "authored environment"
    finally:
        environment.devices.close_all_devices()


def test_environment_bundle_rejects_a_different_compile_profile():
    environment = white_noise()
    try:
        with pytest.raises(ValueError, match="matching|match"):
            replace(environment, soccfg=make_mock_soccfg(n_gens=3))
    finally:
        environment.devices.close_all_devices()


def test_white_noise_integrated_round_is_compact_and_decimated_remains_signals():
    import numpy as np
    from zcu_tools.core.simulation import (
        AcquireRequest,
        ReadoutWindow,
        RoundRequest,
        RoundSignals,
    )
    from zcu_tools.program.v2.simulation import IntegratedRound

    env = white_noise(std_adc=3, seed=17)
    try:
        request = AcquireRequest(
            mode="integrated",
            loop_dims=(100,),
            avg_level=0,
            reps=100,
            readouts=(ReadoutWindow(0, 0, 8, 307.2, (0.0,)),),
            root_seed=17,
            sequence=0,
            rng_version=env.soc.binding.rng_version,
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
