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
    ChargeDriveRequirement,
    ReadoutSlot,
    ReadoutWindow,
    RoundRequest,
    RoundSignals,
    TimelineTemplate,
)
from zcu_tools.experiment.single_fluxonium.runtime import (
    SingleFluxoniumRuntimeOptions,
    SingleFluxoniumSource,
    single_fluxonium,
)
from zcu_tools.program.v2 import LoadValue, TimelineBuilder
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
from zcu_tools.simulate.evolution import Engine

from tests.core._simulation import TIME_BASE, tick_array, tick_tuple
from tests.program._timeline import compile_modules

from ._inputs import settings, slot

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


def independent_populations(source, request, point):
    """Direct fixed-coordinate Lindblad ODE, independent of evolution dispatch."""
    facts = source.model_facts(source.snapshot())
    model = facts.source.model((0,))
    rho0 = facts.source.initial_density
    charge = model.drives[0]
    template = request.templates[0]
    end = request.time_base.ns(template.events[-1].start_ticks[point])
    boundaries = sorted(
        {
            0.0,
            end,
            *(
                float(value)
                for event in template.events
                if event.gen_ch != 0
                for value in (
                    request.time_base.ns(event.start_ticks[point]),
                    request.time_base.ns(
                        event.start_ticks[point] + event.duration_ticks[point]
                    ),
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
            and request.time_base.ns(event.start_ticks[point])
            <= (start + stop) / 2
            < request.time_base.ns(
                event.start_ticks[point] + event.duration_ticks[point]
            )
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
        if config.flux_line is None:
            raise AssertionError("fast-flux oracle fixture requires a bound line")
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
    env = single_fluxonium(config)
    source = SingleFluxoniumSource(config, env.devices)
    try:
        request = request_for(source, "rabi")
        compiled = request.templates[0]
        second = slot(5, 8, 24, frequency_mhz=820, phase_deg=60, gain=-0.35)
        compiled = replace(compiled, events=(*compiled.events, second))
        request = replace(request, templates=(compiled,))
        validated = source.preflight(request, source.snapshot())
        tracks = validated.plans[0][0].batch.tracks
        assert [track.port for track in tracks] == ["primary", "auxiliary"]
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
    env = single_fluxonium(config)
    source = SingleFluxoniumSource(config, env.devices)
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


def test_overlapping_named_drives_acquire_and_respond_to_each_calibration():
    means = []
    for disabled in (None, "first", "second"):
        configured = named_drive_settings().model_dump(mode="json")
        if disabled is not None:
            configured["drive_lines"][disabled]["charge_drive_ghz_per_gain"] = {
                "real": 0.0,
                "imag": 0.0,
            }
        env = single_fluxonium(parse_single_fluxonium(configured))
        try:
            first = PulseCfg(
                ch=1,
                nqz=1,
                freq=580,
                gain=0.2,
                waveform=ConstWaveformCfg(length=0.032),
            )
            second = PulseCfg(
                ch=5,
                nqz=1,
                freq=580,
                phase=60,
                gain=0.35,
                waveform=ConstWaveformCfg(length=0.024),
            )
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
            builder = TimelineBuilder(env.soccfg, reps=100_000)
            builder.fork("default", into="second")
            builder.emit(first.build("first"))
            builder.emit(second.build("second"), timeline="second")
            builder.join("default", "second", into="default")
            builder.emit(readout.build("ro"))
            builder.delay_by_timeline()
            program = builder.build()
            average = program.acquire(
                env.soc,
                progress=False,
                charge_requirements=(
                    ChargeDriveRequirement("first drive", 1),
                    ChargeDriveRequirement("second drive", 5),
                ),
            )
            assert average is not None
            means.append(np.asarray(average))
        finally:
            if env.close_backend is not None:
                env.close_backend()
            env.devices.close_all_devices()
    assert np.max(np.abs(means[0] - means[1])) > 1e-3
    assert np.max(np.abs(means[0] - means[2])) > 1e-3


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
    from zcu_tools.experiment.ftf import replay as ftf_replay
    from zcu_tools.experiment.single_fluxonium import replay as single_replay
    from zcu_tools.resources.ftf import parse_ftf

    from tests.resources._ftf_data import ftf_document

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
        pytest.fail(
            f"DC-only material must admit an explicitly unbound flux line: {error}"
        )
    env = single_fluxonium(config)
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


@pytest.mark.parametrize(
    "case, expected",
    [
        ("absent", [[972.5758438077908, -0.25173123177691964]]),
        (
            "inactive",
            [
                [972.5535136297327, -0.23690658344665044],
                [972.5616134942782, -0.30040715262673],
            ],
        ),
        (
            "mixed",
            [
                [972.5535136297327, -0.23690658344665044],
                [972.5616134942782, -0.30040715262664125],
            ],
        ),
        ("active", [[972.5755645388051, -0.22000655695588697]]),
        ("delayed", [[972.5752852698197, -0.18828188213495886]]),
        ("distorted", [[972.5755645388051, -0.22000655695588697]]),
        ("tail", [[972.5423259988822, -0.2611257285396458]]),
    ],
)
def test_fixed_seed_acquire_preserves_flux_controls(case, expected):
    # Known-good public acquire means, including ADC noise, from the unchanged
    # planner. Exact equality protects this planning-only optimization.
    values = physical_settings().model_dump(mode="json")
    if case == "active":
        values["flux_line"].update(delay_ns=0.0, tau_ns=0.0)
    elif case == "delayed":
        values["flux_line"]["delay_ns"] = 6.0
    elif case == "distorted":
        values["flux_line"]["distortion_terms"] = [{"amplitude": 0.3, "tau_ns": 12.0}]
    elif case == "tail":
        values["flux_line"]["tau_ns"] = 20.0
    env = single_fluxonium(parse_single_fluxonium(values))
    try:
        microwave = PulseCfg(
            ch=1,
            nqz=1,
            freq=580,
            gain=0.2,
            waveform=ConstWaveformCfg(length=0.032),
        )
        fast_flux = PulseCfg(
            ch=2,
            nqz=1,
            freq=0,
            gain=0.4,
            waveform=ConstWaveformCfg(length=0.008),
        )
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
        builder = TimelineBuilder(
            env.soccfg, reps=8, initial_delay=0.0, relax_delay=0.0
        )
        if case in ("inactive", "mixed"):
            builder.add_loop("axis", 2)
        builder.emit(microwave.build("mw"))
        if case in ("inactive", "mixed"):
            counts = [0, 0] if case == "inactive" else [0, 1]
            builder.emit(LoadValue("count", counts, "axis", "n"))
            with builder.repeat("ff_repeat", "n", range_hint=(0, 1)):
                builder.emit(fast_flux.build("ff"))
        elif case != "absent":
            builder.emit(fast_flux.build("ff"))
        builder.timeline_inc(0.004)
        builder.emit(readout.build("ro"))
        builder.delay_by_timeline()
        program = builder.build()
        average = program.acquire(env.soc, rounds=1, progress=False)
        assert average is not None
        np.testing.assert_array_equal(np.asarray(average).reshape(-1, 2), expected)
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
        with pytest.raises(ValueError, match="untriggered cavity drive"):
            source.execute(RoundRequest(request, source.snapshot(), 0))
    finally:
        source.close()
        env.devices.close_all_devices()


class EvolutionObserver(Engine):
    """Count calls at the injected evolution boundary without replacing admission."""

    def __init__(self):
        super().__init__(workers=1)
        self.evolutions = 0

    def evolve(self, source, batches, **kwargs):
        self.evolutions += 1
        return super().evolve(source, batches, **kwargs)


def test_new_acquire_rechecks_charge_admission_before_reusing_numeric_preparation():
    config = physical_settings()
    env = single_fluxonium(config)
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
