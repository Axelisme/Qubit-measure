"""Single-cavity acquisition at the public source and program boundaries.

Small retained spaces isolate composition. Production-sized cost is measured by
an author probe, not by weakening the environment defaults in these tests.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from zcu_tools.core.simulation import (
    AcquireRequest,
    AcquisitionMode,
    ReadoutSlot,
    ReadoutWindow,
    RoundRequest,
    RoundSignals,
    TimelineTemplate,
)
from zcu_tools.experiment.single_fluxonium.runtime import (
    SingleFluxoniumSource,
    single_fluxonium,
)
from zcu_tools.program.v2 import ModularProgramV2, ProgramV2Cfg
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.simulation import IntegratedRound
from zcu_tools.resources.simulation_replay import (
    ReplayAcquire,
    read_replay,
    replay_paths,
    save_replay,
)
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


def request_for(
    source, experiment="onetone", mode: AcquisitionMode = "integrated", reps=100_000
):
    facts = source.model_facts(source.snapshot())
    frequency = float(np.diff(facts.model.energies_ghz[:2])[0]) * 1000
    ends = (
        np.array([40.0, 60.0])
        if experiment in ("t1", "fastflux_t1")
        else np.full(2, 40.0)
    )
    pulses = []
    if experiment != "onetone":
        drive = slot(1, 0, 32, frequency_mhz=frequency, phase_deg=0.0, gain=1.0)
        if experiment == "rabi":
            drive.gain[:] = [0, 1]
        if experiment == "twotone":
            drive.frequency_mhz[:] = [frequency, frequency + 500]
        pulses.append(drive)
    if experiment == "fastflux_t1":
        pulses.append(slot(2, 32, 8, frequency_mhz=0.0, phase_deg=0.0, gain=0.1))
    readout = replace(
        slot(0, 40, 24, frequency_mhz=7000.0, phase_deg=0.0, gain=0.1),
        start_us=ends / 1000,
    )
    pulses.append(readout)
    window = ReadoutSlot(
        0,
        8,
        1000,
        ends[:, None] / 1000 + 0.002,
        np.full((2, 1), 7000.0),
        np.zeros((2, 1)),
        np.zeros((2, 1), dtype=np.int64),
    )
    compiled = TimelineTemplate(
        np.array([0, 1], dtype=np.int64), tuple(pulses), (window,), (ends + 24) / 1000
    )
    return AcquireRequest(
        mode,
        (reps, 2),
        0,
        (ReadoutWindow(0, 0, 8, 1000, (0.042,)),),
        123,
        0,
        f"PCG64/SeedSequence-v2/numpy-{np.__version__}",
        point_dims=(2,),
        point_names=("sweep",),
        reps=reps,
        templates=(compiled,),
    )


def expect_integrated(result):
    if not isinstance(result, IntegratedRound):
        raise AssertionError(
            "integrated source must deliver statistics and deferred raw"
        )
    return result


def independent_populations(source, request, point):
    """Direct fixed-coordinate Lindblad ODE, independent of evolution dispatch."""
    facts = source.model_facts(source.snapshot())
    model = facts.source.model((0,))
    rho0 = facts.source.initial_density
    charge = model.drives[0]
    template = request.templates[0]
    end = template.events[-1].start_us[point] * 1000
    boundaries = sorted(
        {
            0.0,
            end,
            *(
                float(value)
                for event in template.events
                if event.gen_ch != 0
                for value in (
                    event.start_us[point] * 1000,
                    (event.start_us[point] + event.duration_us[point]) * 1000,
                )
                if 0 < value < end
            ),
        }
    )
    state = rho0.ravel()
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        active = [
            event
            for event in template.events
            if event.gen_ch != 0
            and event.start_us[point] * 1000
            <= (start + stop) / 2
            < (event.start_us[point] + event.duration_us[point]) * 1000
        ]
        flux = facts.model.dc_flux + sum(
            event.gain[point] * source.settings.flux_line.flux_phi0_per_gain
            for event in active
            if event.gen_ch == 2
        )
        level = int(round((flux - facts.model.dc_flux) / 1e-3))
        bath = facts.source.model((level,))
        transform = model.eigenvectors.conj().T @ bath.eigenvectors
        h0 = (
            model.eigenvectors.conj().T
            @ facts.model.hamiltonian(flux)
            @ model.eigenvectors
        )
        rate = bath.rates_per_ns
        outgoing = rate.sum(axis=0)
        decay = (outgoing[:, None] + outgoing[None, :]) / 2 + bath.dephasing_per_ns

        interval = (h0, active, transform, decay, rate)

        def derivative(time, flattened, interval=interval):
            h0, active, transform, decay, rate = interval
            rho = flattened.reshape(rho0.shape)
            hamiltonian = h0.copy()
            for event in active:
                if event.gen_ch != 1:
                    continue
                angle = 2 * np.pi * event.frequency_mhz[
                    point
                ] / 1000 * time + np.deg2rad(event.phase_deg[point])
                hamiltonian += event.gain[point] * (
                    np.cos(angle) * charge.in_phase_ghz
                    + np.sin(angle) * charge.quadrature_ghz
                )
            local = transform.conj().T @ rho @ transform
            dissipated = -decay * local
            dissipated[np.diag_indices_from(local)] += rate @ local.diagonal()
            result = (
                -2j * np.pi * (hamiltonian @ rho - rho @ hamiltonian)
                + transform @ dissipated @ transform.conj().T
            )
            return result.ravel()

        solution = solve_ivp(
            derivative, (start, stop), state, method="DOP853", rtol=1e-9, atol=1e-11
        )
        assert solution.success
        state = solution.y[:, -1]
    return state.reshape(rho0.shape).diagonal().real


@pytest.mark.parametrize(
    "experiment", ["onetone", "twotone", "rabi", "t1", "fastflux_t1"]
)
def test_representative_single_points_match_population_oracle(experiment):
    config = physical_settings()
    if experiment == "fastflux_t1":
        config = config.model_copy(
            update={
                "flux_line": config.flux_line.model_copy(
                    update={"tau_ns": 0.0, "delay_ns": 0.0}
                )
            }
        )
    env = single_fluxonium(config)
    source = SingleFluxoniumSource(config, env.devices)
    try:
        request = request_for(source, experiment)
        try:
            result = expect_integrated(
                source.execute(RoundRequest(request, source.snapshot(), 0))
            )
        except NotImplementedError:
            pytest.fail(
                "single-cavity execute must evolve and sample a real integrated round"
            )
        populations = result.statistics.state_counts / request.reps
        for point in range(2):
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
        if experiment == "fastflux_t1":
            dc_only = replace(
                request.templates[0],
                events=tuple(
                    event for event in request.templates[0].events if event.gen_ch != 2
                ),
            )
            ignored = independent_populations(
                source, replace(request, templates=(dc_only,)), 0
            )
            assert abs(populations[0, 1] - ignored[1]) > 0.015
    finally:
        source.close()
        env.devices.close_all_devices()


def test_onetone_mean_matches_causal_scalar_cavity_solution():
    config = physical_settings()
    env = single_fluxonium(config)
    source = SingleFluxoniumSource(config, env.devices)
    try:
        request = request_for(source)
        try:
            result = expect_integrated(
                source.execute(RoundRequest(request, source.snapshot(), 0))
            )
        except NotImplementedError:
            pytest.fail("onetone acquire must compute physical cavity means")
        facts = source.model_facts(source.snapshot())
        frequency = facts.readout.frequencies_ghz[0, 0]
        external = 2 * np.pi * config.readout.resonator.kappa_external_mhz / 1000
        internal = 2 * np.pi * config.readout.resonator.kappa_internal_mhz / 1000
        pole = (external + internal) / 2 + 2j * np.pi * (frequency - 7)
        # The ADC samples at 2..9 ns after switch-on. The cavity starts in vacuum.
        emitted = 1 - external / (2 * pole) * (1 - np.exp(-pole * np.arange(2, 10)))
        mean = 1000 * np.conj(emitted).sum()
        expected = [mean.real, mean.imag]
        channel = result.statistics.channels[0]
        np.testing.assert_allclose(
            channel.conditional_sums[0, 0, 0], expected, atol=1e-6, rtol=1e-9
        )
        # Ground state is absorbing at zero temperature. Only ADC rounding noise
        # remains in the direct sum, with standard error sqrt(1/(12*reps)).
        np.testing.assert_allclose(
            channel.sums[0, 0] / request.reps, expected, atol=0.01, rtol=0
        )
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
            pytest.fail(
                "single-cavity execute must produce replayable physical signals"
            )
        if isinstance(result, IntegratedRound):
            expected = result.raw.materialize(request.reps, request.avg_level)
        else:
            expected = {c.channel: c.iq.copy() for c in result.channels}
        paths = replay_paths(source.private_replay)
        assert len(paths) == 1
        later = source.execute(RoundRequest(request, snapshot, 3))
        if isinstance(later, IntegratedRound):
            assert not np.array_equal(
                later.statistics.channels[0].sums,
                expect_integrated(result).statistics.channels[0].sums,
            )
    finally:
        source.close()
        env.devices.close_all_devices()
    if isinstance(result, IntegratedRound):
        for ch, raw in result.raw.materialize(request.reps, request.avg_level).items():
            np.testing.assert_array_equal(raw, expected[ch])
    replayed = replay.replay_round(paths[0])
    actual = (
        replayed.raw.materialize(request.reps, request.avg_level)
        if isinstance(replayed, IntegratedRound)
        else {c.channel: c.iq for c in replayed.channels}
    )
    for ch in expected:
        np.testing.assert_array_equal(actual[ch], expected[ch])
    if isinstance(result, IntegratedRound):
        np.testing.assert_array_equal(
            expect_integrated(replayed).statistics.channels[0].sums,
            result.statistics.channels[0].sums,
        )
    else:
        assert isinstance(result, RoundSignals)
        assert result.channels[0].iq.shape == (16, 1, 8, 2)
    saved = read_replay(paths[0])
    incompatible = saved.acquire.model_copy(
        update={"versions": (("backend", "obsolete"),)}
    )
    path = save_replay(tmp_path / "obsolete", incompatible, saved.round.snapshot, 2)
    with pytest.raises(ValueError, match="version mismatch"):
        replay.replay_round(path)


def test_program_acquire_raw_decimated_and_live_round_refresh():
    env = single_fluxonium(physical_settings())
    try:
        readout = PulseReadoutCfg(
            gen_ch=0,
            ro_ch=0,
            nqz=1,
            freq=7000,
            gain=0.1,
            waveform=ConstWaveformCfg(length=0.024),
            ro_length=0.016,
            trig_offset=0.002,
        )
        pulse = PulseCfg(
            ch=1, nqz=1, freq=580, gain=0.2, waveform=ConstWaveformCfg(length=0.032)
        )
        program = ModularProgramV2(
            env.soccfg,
            ProgramV2Cfg(reps=8, rounds=2),
            modules=[pulse.build("mw"), readout.build("ro")],
        )
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
        np.testing.assert_allclose(
            np.asarray(raw[0]).reshape(-1, 2),
            trace.reshape(-1, trace.shape[-2], 2).sum(axis=1),
            atol=(trace.shape[-2] + 1) / 2,
            rtol=0,
        )
    finally:
        env.devices.close_all_devices()


@pytest.mark.parametrize("target", ["single", "ftf"])
def test_replay_rejects_other_backend_settings_before_numeric_work(
    tmp_path: Path, target
):
    from zcu_tools.core.simulation import EnvironmentSnapshot
    from zcu_tools.experiment.ftf import replay as ftf_replay
    from zcu_tools.experiment.single_fluxonium import replay as single_replay
    from zcu_tools.resources.ftf import parse_ftf

    from tests.resources._ftf_data import ftf_document

    wrong = parse_ftf(ftf_document()) if target == "single" else physical_settings()
    replay = single_replay if target == "single" else ftf_replay
    fixed = ReplayAcquire(
        settings=wrong,
        request=AcquireRequest("integrated", (1,), 0, (), 123, 0, "test"),
        versions=replay.replay_versions(),
    )
    path = save_replay(tmp_path, fixed, EnvironmentSnapshot("wrong", "wrong"), 0)
    try:
        replay.replay_round(path)
    except AttributeError:
        pytest.fail(
            f"{target} replay must reject other-backend settings with ValueError before numeric work"
        )
    except ValueError as error:
        assert (
            "single-fluxonium replay requires single-fluxonium settings"
            if target == "single"
            else "FTF replay requires FTF settings"
        ) in str(error)
    else:
        pytest.fail("replay must reject settings belonging to another backend")


def test_changed_settings_refresh_snapshot_identity_and_replay_inputs(tmp_path: Path):
    from zcu_tools.experiment.single_fluxonium import replay_round

    config = physical_settings()
    env = single_fluxonium(config)
    source = SingleFluxoniumSource(
        config, env.devices, private_replay=tmp_path / "archive"
    )
    try:
        request = request_for(source, reps=8)
        before = source.snapshot()
        old_facts = source.model_facts(before)
        source.settings = config.model_copy(
            update={
                "readout": config.readout.model_copy(
                    update={
                        "adc": config.readout.adc.model_copy(update={"gain": 2000.0})
                    }
                )
            }
        )
        after = source.snapshot()
        assert after.model_identity != before.model_identity
        assert after.values == before.values
        assert source.model_facts(after) is not old_facts
        result = expect_integrated(source.execute(RoundRequest(request, after, 0)))
        archived = replay_paths(tmp_path / "archive")[0]
    finally:
        source.close()
        env.devices.close_all_devices()
    replayed = expect_integrated(replay_round(archived))
    np.testing.assert_array_equal(
        replayed.statistics.channels[0].sums, result.statistics.channels[0].sums
    )
    for channel, values in result.raw.materialize(
        request.reps, request.avg_level
    ).items():
        np.testing.assert_array_equal(
            replayed.raw.materialize(request.reps, request.avg_level)[channel], values
        )


def test_builder_backend_cleanup_rejects_later_work_but_leaves_drivers_owned():
    env = single_fluxonium(physical_settings())
    driver = env.create_device("FakeDevice", physical_settings().dc_bias.source.name)
    try:
        try:
            cleanup = env.close_backend
        except AttributeError:
            pytest.fail(
                "single-fluxonium builder must expose backend cleanup to its host"
            )
        if cleanup is None:
            pytest.fail("single-fluxonium builder must install its source cleanup")
        cleanup()
        with pytest.raises(RuntimeError, match="closed"):
            env.soc.binding.capture_snapshot()
        assert driver.get_value() == 0
        driver.set_value(0.2)
        assert driver.get_value() == 0.2
    finally:
        env.devices.close_all_devices()


def test_dc_only_single_source_acquires_without_a_fast_flux_route():
    values = physical_settings().model_dump(mode="json")
    values["flux_line"] = None
    try:
        config = parse_single_fluxonium(values)
    except ValueError as error:
        pytest.fail(f"DC-only material must admit an explicitly unbound flux line: {error}")
    env = single_fluxonium(config)
    try:
        readout = PulseReadoutCfg(
            gen_ch=0, ro_ch=0, nqz=1, freq=7000, gain=0.1,
            waveform=ConstWaveformCfg(length=0.024), ro_length=0.016,
        )
        program = ModularProgramV2(
            env.soccfg, ProgramV2Cfg(reps=8), modules=[readout.build("ro")]
        )
        average = program.acquire(env.soc, progress=False)
        assert average is not None and np.linalg.norm(average[0]) > 1
        assert program.get_raw() is not None
    finally:
        if env.close_backend is not None:
            env.close_backend()
        env.devices.close_all_devices()


def test_source_rejects_nonterminal_cavity_pulses_before_evolution():
    env = single_fluxonium(physical_settings())
    source = SingleFluxoniumSource(physical_settings(), env.devices)
    try:
        request = request_for(source, reps=1)
        original = request.templates[0]
        earlier = slot(0, 0, 8, frequency_mhz=7000.0, phase_deg=0.0, gain=0.1)
        request = replace(
            request, templates=(replace(original, events=(earlier, *original.events)),)
        )
        with pytest.raises(
            ValueError, match="only terminal readout|terminal readout only"
        ):
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
