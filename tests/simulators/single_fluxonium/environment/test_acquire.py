"""Single-cavity acquisition at the public source and program boundaries.

Small retained spaces isolate composition. Production-sized cost is measured by
an author probe, not by weakening the environment defaults in these tests.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from zcu_tools.core.simulation import (
    AcquireRequest,
    AcquisitionMode,
    ChargeDriveRequirement,
    ReadoutSlot,
    ReadoutWindow,
    RoundRequest,
    TimelineTemplate,
)
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.simulation import IntegratedRound
from zcu_tools.simulate.evolution import EvolutionResult, PathDiagnostics
from zcu_tools.simulators.replay import (
    ReplayAcquire,
    read_replay,
    replay_paths,
    save_replay,
)
from zcu_tools.simulators.single_fluxonium.config import parse_single_fluxonium
from zcu_tools.simulators.single_fluxonium.environment.runtime import (
    SingleFluxoniumRuntimeOptions,
    SingleFluxoniumSource,
)

from tests.core._simulation import TIME_BASE, tick_array, tick_tuple
from tests.program._profiles import profile_soccfg
from tests.program._timeline import compile_modules
from tests.simulators._engine import EngineDouble, controlled_engine
from tests.simulators._sources import single_environment as single_fluxonium
from tests.simulators.single_fluxonium.environment._inputs import settings, slot

pytestmark = pytest.mark.sim


def physical_settings():
    values = settings().model_dump(mode="json")
    values["chip"]["qubit"]["q_cap"] = 3000
    values["readout"]["resonator"]["charge_coefficient_ghz"] = 0.03
    values["readout"]["adc"]["gain"] = 1000
    values["numerics"]["time_scale"] = 0
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
        start_ticks=tick_array(ends / 1000),
    )
    pulses.append(readout)
    window = ReadoutSlot(
        0,
        8,
        TIME_BASE.period_ticks(1000),
        tick_array(ends[:, None] / 1000 + 0.002),
        np.full((2, 1), 7000.0),
        np.zeros((2, 1)),
        np.zeros((2, 1), dtype=np.int64),
    )
    compiled = TimelineTemplate(
        np.array([0, 1], dtype=np.int64),
        tuple(pulses),
        (window,),
        tick_array((ends + 24) / 1000),
    )
    return AcquireRequest(
        mode,
        (reps, 2),
        0,
        (ReadoutWindow(0, 0, 8, TIME_BASE.period_ticks(1000), tick_tuple((0.042,))),),
        123,
        0,
        f"PCG64/SeedSequence-v2/numpy-{np.__version__}",
        point_dims=(2,),
        point_names=("sweep",),
        reps=reps,
        templates=(compiled,),
        time_base=TIME_BASE,
    )


def expect_integrated(result):
    if not isinstance(result, IntegratedRound):
        raise AssertionError(
            "integrated source must deliver statistics and deferred raw"
        )
    return result


@pytest.mark.parametrize(
    "experiment", ["onetone", "twotone", "rabi", "t1", "fastflux_t1"]
)
def test_representative_single_points_deliver_given_populations_to_adc(experiment):
    config = physical_settings()
    env = single_fluxonium(config, soccfg=profile_soccfg("single_fluxonium"))
    # Deliberately return reverse point order with distinct supplied populations.
    expected = np.array([[0.8, 0.2, 0, 0], [0.3, 0.7, 0, 0]])
    engine = EngineDouble(
        result=EvolutionResult(
            np.array([1, 0], dtype=np.int64),
            np.array([np.diag(expected[1]), np.diag(expected[0])], dtype=np.complex128),
            PathDiagnostics(),
        )
    )
    source = SingleFluxoniumSource(
        config, env.devices, runtime=SingleFluxoniumRuntimeOptions(engine=engine)
    )
    try:
        request = request_for(source, experiment)
        result = expect_integrated(
            source.execute(RoundRequest(request, source.snapshot(), 0))
        )
        np.testing.assert_allclose(
            result.statistics.state_counts / request.reps, expected, atol=0.015, rtol=0
        )
        assert len(engine.calls) == 1
        assert result.statistics.channels[0].conditional_sums.shape == (2, 1, 4, 2)
        assert result.statistics.raw_identity.stream == (0, 0, 1)
    finally:
        source.close()
        if env.close_backend is not None:
            env.close_backend()
        env.devices.close_all_devices()


def test_onetone_mean_matches_causal_scalar_cavity_solution():
    config = physical_settings()
    env = single_fluxonium(config, soccfg=profile_soccfg("single_fluxonium"))
    source = SingleFluxoniumSource(
        config,
        env.devices,
        runtime=SingleFluxoniumRuntimeOptions(engine=EngineDouble()),
    )
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
    from zcu_tools.simulators.single_fluxonium.environment import replay

    env = single_fluxonium(
        physical_settings(), soccfg=profile_soccfg("single_fluxonium")
    )
    source = SingleFluxoniumSource(
        physical_settings(),
        env.devices,
        runtime=SingleFluxoniumRuntimeOptions(engine=EngineDouble()),
    )
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
    saved = read_replay(paths[0])
    assert saved.round.index == 2
    assert saved.acquire.request.point_dims == request.point_dims
    incompatible = saved.acquire.model_copy(
        update={"versions": (("backend", "obsolete"),)}
    )
    path = save_replay(tmp_path / "obsolete", incompatible, saved.round.snapshot, 2)
    with pytest.raises(ValueError, match="version mismatch"):
        replay.replay_round(path)


def named_drive_settings():
    values = physical_settings().model_dump(mode="json")
    values["flux_line"] = None
    values["drive_lines"] = {
        "first": {
            "name": "primary",
            "gen_ch": 1,
            "charge_drive_ghz_per_gain": {"real": 0.1, "imag": -0.02},
        },
        "second": {
            "name": "auxiliary",
            "gen_ch": 5,
            "charge_drive_ghz_per_gain": {"real": -0.05, "imag": 0.1},
        },
    }
    return parse_single_fluxonium(values)


def test_preflight_routes_overlapping_events_to_their_own_named_source():
    config = named_drive_settings()
    env = single_fluxonium(config, soccfg=profile_soccfg("single_fluxonium"))
    source = SingleFluxoniumSource(
        config,
        env.devices,
        runtime=SingleFluxoniumRuntimeOptions(engine=EngineDouble()),
    )
    try:
        request = request_for(source, "rabi")
        compiled = request.templates[0]
        second = slot(5, 8, 24, frequency_mhz=820, phase_deg=60, gain=-0.35)
        compiled = replace(compiled, events=(*compiled.events, second))
        request = replace(request, templates=(compiled,))
        validated = source.preflight(request, source.snapshot())
        tracks = validated.plans[0][0].batch.tracks
        assert [track.port for track in tracks] == ["primary", "auxiliary"]
        facts = source.model_facts(source.snapshot())
        model = facts.source.model((0,))
        charge = (
            model.eigenvectors.conj().T
            @ facts.model.charge_operator()
            @ model.eigenvectors
        )
        assert [drive.name for drive in model.drives] == ["primary", "auxiliary"]
        # Caller supplies distinct complex calibrations for the two named lines.
        for drive, calibration in zip(
            model.drives, (0.1 - 0.02j, -0.05 + 0.1j), strict=True
        ):
            np.testing.assert_allclose(
                drive.in_phase_ghz, calibration.real * charge, atol=1e-12
            )
            np.testing.assert_allclose(
                drive.quadrature_ghz, -calibration.imag * charge, atol=1e-12
            )
        np.testing.assert_array_equal(tracks[0].start_ns, [0, 0])
        np.testing.assert_array_equal(tracks[1].start_ns, [8, 8])
        np.testing.assert_array_equal(tracks[1].duration_ns, [24, 24])
        np.testing.assert_allclose(tracks[1].frequency_ghz, [0.82, 0.82])
        np.testing.assert_allclose(tracks[1].phase_rad, [np.pi / 3, np.pi / 3])
        np.testing.assert_allclose(tracks[1].amplitude, [-0.35, -0.35])
    finally:
        source.close()
        if env.close_backend is not None:
            env.close_backend()
        env.devices.close_all_devices()


@pytest.mark.parametrize("channel", [3, 6])
def test_preflight_rejects_unknown_generator_with_event_path(channel):
    config = named_drive_settings()
    env = single_fluxonium(config, soccfg=profile_soccfg("single_fluxonium"))
    source = SingleFluxoniumSource(
        config,
        env.devices,
        runtime=SingleFluxoniumRuntimeOptions(engine=EngineDouble()),
    )
    try:
        request = request_for(source, "rabi")
        compiled = request.templates[0]
        events = (replace(compiled.events[0], gen_ch=channel), *compiled.events[1:])
        request = replace(request, templates=(replace(compiled, events=events),))
        with pytest.raises(
            ValueError,
            match=rf"template.events.0.gen_ch.*unbound.*generator {channel}",
        ):
            source.preflight(request, source.snapshot())
    finally:
        source.close()
        if env.close_backend is not None:
            env.close_backend()
        env.devices.close_all_devices()


def test_program_acquire_raw_decimated_and_live_round_refresh():
    env = single_fluxonium(
        physical_settings(), soccfg=profile_soccfg("single_fluxonium")
    )
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
        program = compile_modules(
            env.soccfg,
            modules=[pulse.build("mw"), readout.build("ro")],
            reps=8,
        )
        bias = env.create_device("FakeDevice", physical_settings().dc_bias.source.name)

        def change_bias(*_args, **_kwargs):
            bias.set_value(0.2)

        try:
            average = program.acquire(
                env.soc,
                progress=False,
                round_hook=change_bias,
                rounds=2,
            )
        except NotImplementedError:
            pytest.fail("bound program acquire must return real single-cavity IQ")
        assert average is not None
        assert env.soc.binding.rounds[0].snapshot.values[0][1] == 0
        assert env.soc.binding.rounds[1].snapshot.values[0][1] == 0.2
        raw = program.get_raw()
        assert raw is not None and np.asarray(raw[0]).shape == (8, 1, 2)
        assert np.linalg.norm(average[0]) > 1
        traces = program.acquire_decimated(
            env.soc,
            progress=False,
            rounds=2,
        )
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
    from zcu_tools.simulators.ftf.config import parse_ftf
    from zcu_tools.simulators.ftf.environment import replay as ftf_replay
    from zcu_tools.simulators.single_fluxonium.environment import (
        replay as single_replay,
    )

    from tests.simulators.ftf.config._data import ftf_document

    wrong = parse_ftf(ftf_document()) if target == "single" else physical_settings()
    replay = single_replay if target == "single" else ftf_replay
    fixed = ReplayAcquire(
        settings=wrong,
        request=AcquireRequest(
            "integrated", (1,), 0, (), 123, 0, "test", time_base=TIME_BASE
        ),
        versions=replay.replay_versions(),
    )
    path = save_replay(tmp_path, fixed, EnvironmentSnapshot("wrong", "wrong"), 0)
    try:
        if target == "single":
            single_replay.replay_round(path)
        else:
            ftf_replay.replay_round(path, soccfg=profile_soccfg("ftf"))
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
    config = physical_settings()
    env = single_fluxonium(config, soccfg=profile_soccfg("single_fluxonium"))
    source = SingleFluxoniumSource(
        config,
        env.devices,
        private_replay=tmp_path / "archive",
        runtime=SingleFluxoniumRuntimeOptions(engine=EngineDouble()),
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
    saved = read_replay(archived)
    assert saved.round.snapshot.model_identity == after.model_identity
    assert result.statistics.channels[0].sums.shape == (2, 1, 2)


def test_builder_backend_cleanup_rejects_later_work_but_leaves_drivers_owned():
    env = single_fluxonium(
        physical_settings(), soccfg=profile_soccfg("single_fluxonium")
    )
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
        pytest.fail(
            f"DC-only material must admit an explicitly unbound flux line: {error}"
        )
    env = single_fluxonium(config, soccfg=profile_soccfg("single_fluxonium"))
    try:
        readout = PulseReadoutCfg(
            gen_ch=0,
            ro_ch=0,
            nqz=1,
            freq=7000,
            gain=0.1,
            waveform=ConstWaveformCfg(length=0.024),
            ro_length=0.016,
        )
        program = compile_modules(
            env.soccfg,
            modules=[readout.build("ro")],
            reps=8,
        )
        average = program.acquire(env.soc, progress=False)
        assert average is not None and np.linalg.norm(average[0]) > 1
        assert program.get_raw() is not None
    finally:
        if env.close_backend is not None:
            env.close_backend()
        env.devices.close_all_devices()


def test_source_rejects_nonterminal_cavity_pulses_before_evolution():
    env = single_fluxonium(
        physical_settings(), soccfg=profile_soccfg("single_fluxonium")
    )
    source = SingleFluxoniumSource(physical_settings(), env.devices)
    try:
        request = request_for(source, reps=1)
        original = request.templates[0]
        earlier = slot(0, 0, 8, frequency_mhz=7000.0, phase_deg=0.0, gain=0.1)
        request = replace(
            request, templates=(replace(original, events=(earlier, *original.events)),)
        )
        with pytest.raises(ValueError, match="untriggered cavity drive"):
            source.execute(RoundRequest(request, source.snapshot(), 0))
    finally:
        source.close()
        env.devices.close_all_devices()


class EvolutionObserver(EngineDouble):
    """Count calls at the injected evolution boundary without replacing admission."""

    def __init__(self):
        super().__init__(workers=1)
        self.evolutions = 0

    def evolve(self, source, batches, **kwargs):
        self.evolutions += 1
        return super().evolve(source, batches, **kwargs)


def test_new_acquire_rechecks_charge_admission_before_reusing_numeric_preparation():
    config = physical_settings()
    env = single_fluxonium(config, soccfg=profile_soccfg("single_fluxonium"))
    engine = EvolutionObserver()
    source = SingleFluxoniumSource(
        config, env.devices, runtime=SingleFluxoniumRuntimeOptions(engine=engine)
    )
    try:
        ordinary = replace(
            request_for(source, "rabi", reps=2),
            charge_requirements=(ChargeDriveRequirement("Stokes", 1),),
        )
        snapshot = source.snapshot()
        first = expect_integrated(
            source.open_acquire(ordinary).sample_round(snapshot, 0)
        )
        np.testing.assert_array_equal(
            first.statistics.state_counts.sum(axis=-1), [2, 2]
        )
        completed_evolutions = engine.evolutions
        assert completed_evolutions > 0
        if config.flux_line is None:
            raise AssertionError("reuse acceptance requires a bound flux generator")
        wrong_role = replace(
            ordinary,
            charge_requirements=(
                ChargeDriveRequirement("Stokes", config.flux_line.gen_ch),
            ),
        )
        with pytest.raises(
            ValueError, match=rf"Stokes.*{config.flux_line.gen_ch}.*charge drive line"
        ):
            source.open_acquire(wrong_role).sample_round(snapshot, 0)
        assert engine.evolutions == completed_evolutions
    finally:
        source.close()
        if env.close_backend is not None:
            env.close_backend()
        env.devices.close_all_devices()


def test_source_rejects_invalidated_session_and_refreshes_mutated_controls():
    env = single_fluxonium(
        physical_settings(), soccfg=profile_soccfg("single_fluxonium")
    )
    source = SingleFluxoniumSource(
        physical_settings(),
        env.devices,
        runtime=SingleFluxoniumRuntimeOptions(engine=EngineDouble()),
    )
    try:
        request = request_for(source, "rabi", reps=100_000)
        try:
            session = source.open_acquire(request)
        except NotImplementedError:
            pytest.fail("open_acquire must return a live single-cavity session")
        first = expect_integrated(session.sample_round(source.snapshot(), 0))
        request.templates[0].events[0].gain[:] = 0
        second = expect_integrated(session.sample_round(source.snapshot(), 0))
        calls = controlled_engine(source.engine).calls
        assert len(calls) == 2
        (_, first_batches), (_, second_batches) = calls
        (first_batch,), (second_batch,) = first_batches, second_batches
        assert len(first_batch.tracks) == len(second_batch.tracks) == 1
        first_track, second_track = first_batch.tracks[0], second_batch.tracks[0]
        assert np.any(first_track.amplitude != 0)
        np.testing.assert_array_equal(second_track.amplitude, 0)
        np.testing.assert_array_equal(second_batch.points, first_batch.points)
        np.testing.assert_array_equal(second_track.start_ns, first_track.start_ns)
        np.testing.assert_array_equal(second_track.duration_ns, first_track.duration_ns)
        assert (
            first.statistics.channels[0].sums.shape
            == second.statistics.channels[0].sums.shape
        )
        source.open_acquire(request)
        with pytest.raises(RuntimeError, match="active"):
            session.sample_round(source.snapshot(), 1)
        source.close()
        with pytest.raises(RuntimeError, match="closed|active"):
            source.execute(RoundRequest(request, env.soc.binding.capture_snapshot(), 0))
    finally:
        source.close()
        env.devices.close_all_devices()
