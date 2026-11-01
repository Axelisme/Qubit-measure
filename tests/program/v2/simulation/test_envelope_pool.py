"""Cross-preparation waveform ownership through the public frontend seam."""

from tests.program._profiles import load_soccfg
import math
import sys

import numpy as np
import pytest
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg, GaussWaveformCfg
from zcu_tools.program.v2.simulation.compiled import build_compiled_request

from tests.program._timeline import compile_modules


@pytest.fixture
def environment():
    env = white_noise(load_soccfg(n_gens=2), std_adc=0, seed=4, routes=((0, 0), (1, 0)))
    yield env
    env.devices.close_all_devices()


def program(env, *, length=0.1, name="drive"):
    return compile_modules(
        env.soccfg,
        modules=[
            PulseCfg(
                ch=1,
                nqz=1,
                freq=80,
                gain=0.2,
                waveform=GaussWaveformCfg(length=length, sigma=length / 4),
            ).build(name),
            PulseReadoutCfg(
                gen_ch=0,
                ro_ch=0,
                nqz=1,
                freq=100,
                gain=0.1,
                waveform=ConstWaveformCfg(length=0.1),
                ro_length=0.1,
            ).build("readout"),
        ],
        reps=1,
    )


def envelope(prog, soc, sequence):
    request = build_compiled_request(
        prog, "integrated", soc.binding, sequence, routes=soc.routes
    )
    result = request.materialize_point(0).events[0].envelope
    assert result is not None
    return result


def test_new_preparations_share_equal_content_not_program_or_name(environment):
    soc = environment.soc
    first = envelope(program(environment), soc, 0)
    second = envelope(program(environment, name="renamed"), soc, 1)
    assert second is first
    # The environment, not a global cache, owns the pool.
    other = white_noise(load_soccfg(n_gens=2), std_adc=0, seed=4, routes=((0, 0), (1, 0)))
    try:
        independent = envelope(program(other), other.soc, 0)
        assert independent == first
        assert independent is not first
    finally:
        other.devices.close_all_devices()


@pytest.mark.parametrize("difference", ["samples", "clock", "maxv"])
def test_new_preparation_checks_current_compiled_content(environment, difference):
    prog = program(environment)
    soc = environment.soc
    first = envelope(prog, soc, 0)
    assert envelope(prog, soc, 1) is first
    old_times, old_values = first.sample_period_ticks, first.values
    table = prog.envelopes[1]["envs"]
    assert isinstance(table, dict)
    if difference == "samples":
        next(iter(table.values()))["data"][:, 1] = 7
    elif difference == "clock":
        prog.soccfg["gens"][1]["f_fabric"] *= 2
    else:
        prog.soccfg["gens"][1]["maxv"] //= 2
    changed = envelope(prog, soc, 2)
    assert changed is not first
    if difference == "clock":
        # Doubling the common frequency preserves period ticks, but halves us.
        changed_request = build_compiled_request(
            prog, "integrated", soc.binding, 2, routes=soc.routes
        )
        assert changed_request.time_base.ticks_per_us == 86016
        assert changed.sample_period_ticks == first.sample_period_ticks == 7
        assert changed.values == first.values
    else:
        assert changed != first
    assert first.sample_period_ticks == old_times
    assert first.values == old_values
    assert envelope(prog, soc, 3) is changed


def test_byte_bounded_eviction_keeps_borrowed_envelope_immutable(environment):
    prog = program(environment, length=4.0)
    soc = environment.soc
    first = envelope(prog, soc, 0)
    assert envelope(prog, soc, 1) is first
    original_times, original_values = first.sample_period_ticks, first.values
    # Exceed 32 MiB using the immutable samples alone, independently of the
    # pool's additional key/container accounting. Only retain the first borrow.
    sample_bytes = sum(sys.getsizeof(v) for v in first.values)
    count = math.ceil(32 * 1024**2 / sample_bytes) + 2
    table = prog.envelopes[1]["envs"]
    assert isinstance(table, dict)
    data = next(iter(table.values()))["data"]
    original_data = data.copy()
    for index in range(count):
        data[:, 1] = index + 1
        envelope(prog, soc, index + 2)
    data[:] = original_data
    rebuilt = envelope(prog, soc, count + 2)
    assert rebuilt is not first
    assert rebuilt == first
    assert first.sample_period_ticks == original_times
    assert first.values == original_values
    np.testing.assert_array_equal(data, original_data)
    assert envelope(prog, soc, count + 3) is rebuilt
