"""Single-cavity acquisition at the public source and program boundaries.

Small retained spaces isolate composition. Production-sized cost is measured by
an author probe, not by weakening the environment defaults in these tests.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from zcu_tools.core.simulation import AcquireRequest, ReadoutSlot, ReadoutWindow, RoundRequest, RoundSignals, TimelineTemplate
from zcu_tools.device import DeviceManager
from zcu_tools.experiment.single_fluxonium.runtime import SingleFluxoniumSource, single_fluxonium
from zcu_tools.program.v2 import ModularProgramV2, ProgramV2Cfg
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.simulation import IntegratedRound
from zcu_tools.resources.simulation_replay import ReplayAcquire, read_replay, replay_paths, save_replay
from zcu_tools.resources.single_fluxonium import parse_single_fluxonium

from ._inputs import settings, slot

pytestmark = pytest.mark.sim


def physical_settings():
    values = settings().model_dump(mode="json")
    values["chip"]["qubit"]["q_cap"] = 3000
    values["readout"]["resonator"]["charge_coefficient_ghz"] = 0.03
    values["readout"]["adc"]["gain"] = 1000
    values["numerics"]["min_round_duration_s"] = 0
    return parse_single_fluxonium(values)


def request_for(source, experiment="onetone", mode="integrated", reps=100_000):
    facts = source.model_facts(source.snapshot())
    frequency = float(np.diff(facts.model.energies_ghz[:2])[0]) * 1000
    ends = np.array([40.0, 60.0]) if experiment in ("t1", "fastflux_t1") else np.full(2, 40.0)
    pulses = []
    if experiment != "onetone":
        drive = slot(1, 0, 32, frequency_mhz=frequency, gain=1.0)
        if experiment == "rabi":
            drive.gain[:] = [0, 1]
        if experiment == "twotone":
            drive.frequency_mhz[:] = [frequency, frequency + 500]
        pulses.append(drive)
    if experiment == "fastflux_t1":
        pulses.append(slot(2, 32, 8, gain=0.01))
    readout = replace(slot(0, 40, 24, frequency_mhz=7000, gain=0.1), start_us=ends / 1000)
    pulses.append(readout)
    window = ReadoutSlot(0, 8, 1000, ends[:, None] / 1000 + 0.002,
                         np.full((2, 1), 7000.0), np.zeros((2, 1)), np.zeros((2, 1), dtype=np.int64))
    compiled = TimelineTemplate(np.array([0, 1], dtype=np.int64), tuple(pulses), (window,), (ends + 24) / 1000)
    return AcquireRequest(mode, (reps, 2), 0, (ReadoutWindow(0, 0, 8, 1000, (0.042,)),),
                          123, 0, f"PCG64/SeedSequence-v2/numpy-{np.__version__}",
                          point_dims=(2,), point_names=("sweep",), reps=reps, templates=(compiled,))


def expect_integrated(result):
    if not isinstance(result, IntegratedRound):
        raise AssertionError("integrated source must deliver statistics and deferred raw")
    return result


def independent_populations(source, request, point):
    """Direct fixed-coordinate Lindblad ODE, independent of evolution dispatch."""
    facts = source.model_facts(source.snapshot())
    model = facts.source.model((0,))
    energy = model.energies_ghz
    rho0 = facts.source.initial_density
    charge = model.drives[0]
    template = request.templates[0]
    end = template.events[-1].start_us[point] * 1000
    rate = model.rates_per_ns
    outgoing = rate.sum(axis=0)
    decay = (outgoing[:, None] + outgoing[None, :]) / 2 + model.dephasing_per_ns
    def derivative(time, flattened):
        rho = flattened.reshape(rho0.shape)
        hamiltonian = np.diag(energy).astype(complex)
        for event in template.events:
            if event.gen_ch != 1 or not event.start_us[point] * 1000 <= time < (event.start_us[point] + event.duration_us[point]) * 1000:
                continue
            angle = 2 * np.pi * event.frequency_mhz[point] / 1000 * time + np.deg2rad(event.phase_deg[point])
            hamiltonian += event.gain[point] * (np.cos(angle) * charge.in_phase_ghz + np.sin(angle) * charge.quadrature_ghz)
        result = -2j * np.pi * (hamiltonian @ rho - rho @ hamiltonian) - decay * rho
        result[np.diag_indices_from(result)] += rate @ rho.diagonal()
        return result.ravel()
    solution = solve_ivp(derivative, (0, end), rho0.ravel(), method="DOP853", rtol=1e-9, atol=1e-11)
    assert solution.success
    return solution.y[:, -1].reshape(rho0.shape).diagonal().real


@pytest.mark.parametrize("experiment", ["onetone", "twotone", "rabi", "t1", "fastflux_t1"])
def test_representative_single_points_match_population_oracle(experiment):
    env = single_fluxonium(physical_settings())
    source = SingleFluxoniumSource(physical_settings(), env.devices)
    try:
        request = request_for(source, experiment)
        try:
            result = expect_integrated(source.execute(RoundRequest(request, source.snapshot(), 0)))
        except NotImplementedError:
            pytest.fail("single-cavity execute must evolve and sample a real integrated round")
        populations = result.statistics.state_counts / request.reps
        for point in range(2):
            # The tiny fast-flux pulse is a composition smoke, not an independent
            # continuous-ramp accuracy test (owned by evolution).
            expected = independent_populations(source, request, point)
            np.testing.assert_allclose(populations[point], expected, atol=0.015, rtol=0)
        sums = result.statistics.channels[0].conditional_sums
        assert sums.shape == (2, 1, 4, 2)
        assert np.linalg.norm(sums[0, 0, 0] - sums[0, 0, 1]) > 0.01
        assert result.statistics.raw_identity.stream == (0, 0, 1)
        if experiment == "rabi":
            assert populations[1, 1] > populations[0, 1] + 0.1
        if experiment == "twotone":
            assert populations[0, 1] > populations[1, 1] + 0.1
        if experiment in ("t1", "fastflux_t1"):
            assert populations[1, 1] < populations[0, 1]
    finally:
        source.close()
        env.devices.close_all_devices()


@pytest.mark.parametrize("mode", ["integrated", "decimated"])
def test_round_replay_and_raw_outlive_source(tmp_path: Path, mode):
    from zcu_tools.experiment.single_fluxonium import replay

    env = single_fluxonium(physical_settings())
    source = SingleFluxoniumSource(physical_settings(), env.devices)
    try:
        source.private_replay = tmp_path / "archive"
        request = request_for(source, "rabi", mode, reps=8)
        snapshot = source.snapshot()
        try:
            result = source.execute(RoundRequest(request, snapshot, 2))
        except NotImplementedError:
            pytest.fail("single-cavity execute must produce replayable physical signals")
        if isinstance(result, IntegratedRound):
            expected = result.raw.materialize(request.reps, request.avg_level)
        else:
            expected = {c.channel: c.iq.copy() for c in result.channels}
        paths = replay_paths(source.private_replay)
        assert len(paths) == 1
        later = source.execute(RoundRequest(request, snapshot, 3))
        if isinstance(later, IntegratedRound):
            assert not np.array_equal(later.statistics.channels[0].sums, result.statistics.channels[0].sums)
    finally:
        source.close()
        env.devices.close_all_devices()
    if isinstance(result, IntegratedRound):
        for ch, raw in result.raw.materialize(request.reps, request.avg_level).items():
            np.testing.assert_array_equal(raw, expected[ch])
    replayed = replay.replay_round(paths[0])
    actual = replayed.raw.materialize(request.reps, request.avg_level) if isinstance(replayed, IntegratedRound) else {c.channel: c.iq for c in replayed.channels}
    for ch in expected:
        np.testing.assert_array_equal(actual[ch], expected[ch])
    if isinstance(result, IntegratedRound):
        np.testing.assert_array_equal(replayed.statistics.channels[0].sums, result.statistics.channels[0].sums)
    else:
        assert isinstance(result, RoundSignals)
        assert result.channels[0].iq.shape == (16, 1, 8, 2)
    saved = read_replay(paths[0])
    incompatible = saved.acquire.model_copy(update={"versions": (("backend", "obsolete"),)})
    path = save_replay(tmp_path / "obsolete", incompatible, saved.round.snapshot, 2)
    with pytest.raises(ValueError, match="version mismatch"):
        replay.replay_round(path)


def test_program_acquire_raw_decimated_and_live_round_refresh():
    env = single_fluxonium(physical_settings())
    try:
        readout = PulseReadoutCfg(gen_ch=0, ro_ch=0, nqz=1, freq=7000, gain=0.1,
                                 waveform=ConstWaveformCfg(length=0.024), ro_length=0.016, trig_offset=0.002)
        pulse = PulseCfg(ch=1, nqz=1, freq=580, gain=0.2, waveform=ConstWaveformCfg(length=0.032))
        program = ModularProgramV2(env.soccfg, ProgramV2Cfg(reps=8, rounds=2), modules=[pulse.build("mw"), readout.build("ro")])
        bias = env.create_device("FakeDevice", physical_settings().dc_bias.source.name)
        def change_bias(*_args, **_kwargs):
            bias.set_value(0.2)
        try:
            average = program.acquire(env.soc, progress=False, round_hook=change_bias)
        except NotImplementedError:
            pytest.fail("bound program acquire must return real single-cavity IQ")
        assert average is not None
        assert env.soc.binding.rounds[0].snapshot.values[0][1] == 0
        assert env.soc.binding.rounds[1].snapshot.values[0][1] == 0.2
        raw = program.get_raw()
        assert raw is not None and np.asarray(raw[0]).shape == (8, 1, 2)
        assert np.linalg.norm(average[0]) > 1
        traces = program.acquire_decimated(env.soc, progress=False)
        assert traces is not None
        raw = program.get_raw()
        assert raw is not None
        trace = np.asarray(traces[0])
        np.testing.assert_allclose(np.asarray(raw[0]).reshape(-1, 2), trace.reshape(-1, trace.shape[-2], 2).sum(axis=1), atol=(trace.shape[-2] + 1) / 2, rtol=0)
    finally:
        env.devices.close_all_devices()


def test_source_rejects_nonterminal_cavity_pulses_before_evolution():
    env = single_fluxonium(physical_settings())
    source = SingleFluxoniumSource(physical_settings(), env.devices)
    try:
        request = request_for(source, reps=1)
        original = request.templates[0]
        earlier = slot(0, 0, 8, frequency_mhz=7000, gain=0.1)
        request = replace(request, templates=(replace(original, events=(earlier, *original.events)),))
        with pytest.raises(ValueError, match="only terminal readout|terminal readout only"):
            source.execute(RoundRequest(request, source.snapshot(), 0))
    finally:
        source.close()
        env.devices.close_all_devices()


def test_source_rejects_invalidated_session_and_refreshes_mutated_controls():
    env = single_fluxonium(physical_settings())
    source = SingleFluxoniumSource(physical_settings(), env.devices)
    try:
        request = request_for(source, "rabi", reps=100_000)
        try:
            session = source.open_acquire(request)
        except NotImplementedError:
            pytest.fail("open_acquire must return a live single-cavity session")
        first = expect_integrated(session.sample_round(source.snapshot(), 0))
        request.templates[0].events[0].gain[:] = 0
        second = expect_integrated(session.sample_round(source.snapshot(), 0))
        assert first.statistics.state_counts[1, 1] > 10_000
        assert second.statistics.state_counts[1, 1] == 0
        source.open_acquire(request)
        with pytest.raises(RuntimeError, match="active"):
            session.sample_round(source.snapshot(), 1)
        source.close()
        with pytest.raises(RuntimeError, match="closed|active"):
            source.execute(RoundRequest(request, env.soc.binding.capture_snapshot(), 0))
    finally:
        source.close()
        env.devices.close_all_devices()
