"""Public fast-flux template conversion and engine family selection.

The no-fast-flux compatibility pin retires if that batch contract changes.
"""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.core.simulation import EventSlot, PulseEnvelope, TimelineTemplate
from zcu_tools.simulators.ftf.environment import FTFModelFacts
from zcu_tools.simulators.ftf.environment import ftf
from zcu_tools.simulators.ftf.environment import plan_template
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.simulators.ftf.config import FTFSettings
from zcu_tools.simulators.ftf.config import parse_ftf
from zcu_tools.simulate.evolution import Engine, segment_spans

from tests.core._simulation import TIME_BASE, tick_array, ticks
from tests.simulators.ftf.environment._templates import herald_template
from tests.simulators.ftf.environment._templates import settings
from tests.simulators.ftf.environment._templates import snapshot
from tests.simulators.ftf.environment._templates import template
from tests.program._profiles import profile_soccfg
from tests.program._timeline import compile_modules

pytestmark = pytest.mark.sim


def flux_settings(*, tau: float = 0.0, distortion: bool = False) -> FTFSettings:
    """Three separately bound loops, deliberately sharing a display name."""
    return parse_ftf(
        {
            **settings().model_dump(mode="json"),
            "flux_lines": {
                "f1": {
                    "name": "flux",
                    "gen_ch": 4,
                    "flux_phi0_per_gain": 0.1,
                    "delay_ns": 2.0,
                    "tau_ns": tau,
                    "distortion_terms": [{"amplitude": 0.4, "tau_ns": 2.0}]
                    if distortion
                    else [],
                },
                "coupler": {"name": "flux", "gen_ch": 5, "flux_phi0_per_gain": -0.2},
                "f2": {"name": "flux", "gen_ch": 6, "flux_phi0_per_gain": 0.3},
            },
        }
    )


def flux_slot(channel: int, gain: float) -> EventSlot:
    """A four-ns baseband platform on both points."""
    original = template().events[-1]
    return replace(
        original,
        gen_ch=channel,
        start_ticks=tick_array(np.zeros(2)),
        duration_ticks=tick_array(np.full(2, 0.004)),
        frequency_mhz=np.zeros(2),
        phase_deg=np.zeros(2),
        gain=np.full(2, gain),
        envelopes=(None,),
    )


def flux_template() -> TimelineTemplate:
    """Three flux events, with f1 inactive on the second point."""
    f1 = replace(flux_slot(4, 0.2), active=np.array([True, False]))
    return replace(
        template(),
        events=(f1, flux_slot(5, 0.1), flux_slot(6, 0.04), *template().events),
    )


def test_three_loop_steps_preserve_snapshot_dc_and_delayed_boundaries() -> None:
    config = flux_settings()
    facts = FTFModelFacts(config, snapshot(0.4))
    plan = plan_template(
        config,
        snapshot(0.4),
        flux_template(),
        soccfg=profile_soccfg("ftf"),
        facts=facts,
        time_base=TIME_BASE,
    )
    np.testing.assert_allclose(
        facts.joint.dc_flux, [0.3, 0.24, 0.4], atol=1e-16, rtol=0
    )
    flux = plan.batch.flux
    np.testing.assert_array_equal(flux.offsets, [0, 4, 6])
    np.testing.assert_array_equal(flux.start_ns, [0, 2, 4, 6, 0, 4])
    np.testing.assert_array_equal(
        flux.level,
        [[0, -20, 12], [20, -20, 12], [20, 0, 0], [0, 0, 0], [0, -20, 12], [0, 0, 0]],
    )
    assert plan.batch.tracks == ()
    assert plan.batch.flux_waveform is None
    assert plan.batch.segment_boundaries == ((0.0, 2.0, 4.0, 6.0), (0.0, 4.0))
    np.testing.assert_array_equal(plan.batch.end_ns, [8.0, 10.0])
    # Fast flux never changes the terminal ADC model's round-DC frequencies.
    dc_plan = plan_template(
        config,
        snapshot(0.4),
        template(),
        soccfg=profile_soccfg("ftf"),
        facts=facts,
        time_base=TIME_BASE,
    )
    np.testing.assert_array_equal(
        plan.readouts[0].batch.model.frequencies_ghz,
        dc_plan.readouts[0].batch.model.frequencies_ghz,
    )


@pytest.mark.parametrize("distortion", [False, True], ids=["base-tau", "distortion"])
def test_filtered_three_loop_waveform_retains_delays_and_tail_until_readout(
    distortion: bool,
) -> None:
    config = flux_settings(tau=0.0 if distortion else 2.0, distortion=distortion)
    plan = plan_template(
        config,
        snapshot(),
        flux_template(),
        soccfg=profile_soccfg("ftf"),
        time_base=TIME_BASE,
    )
    waveform = plan.batch.flux_waveform
    assert waveform is not None
    lo, hi = waveform.offsets[:2]
    times, values = waveform.time_ns[lo:hi], waveform.flux[lo:hi]
    at_four = values[np.flatnonzero(times == 4)[0]]
    weight = 0.4 if distortion else 1.0
    np.testing.assert_allclose(
        at_four, [0.3 + 0.02 * (1 - weight * np.exp(-1)), 0.22, 0.4], atol=1e-15
    )
    at_eight = values[np.flatnonzero(times == 8)[0]]
    assert at_eight[0] == pytest.approx(
        0.3 + 0.02 * weight * (1 - np.exp(-2)) * np.exp(-1)
    )
    # The inactive f1 source on point 1 contributes neither a pulse nor a tail.
    np.testing.assert_array_equal(waveform.flux[waveform.offsets[1] :, 0], 0.3)
    np.testing.assert_array_equal(plan.batch.flux.level[-1], [0, 0, 0])


@pytest.mark.parametrize(
    "kind,family",
    [
        ("step", "t4"),
        ("filtered", "adiabatic_map"),
        ("sampled", "adiabatic_map"),
        ("microwave", "flux_dop853"),
    ],
)
def test_flux_pulse_envelope_selects_existing_engine_family(
    kind: str, family: str
) -> None:
    config = flux_settings(tau=2.0 if kind in ("filtered", "microwave") else 0.0)
    pulse = flux_slot(4, 0.02)
    if kind == "sampled":
        pulse = replace(pulse, envelopes=(PulseEnvelope(ticks(0.002), (0, 1, 0)),))
    base = template(tones=1 if kind == "microwave" else 0)
    original = replace(base, events=(pulse, *base.events))
    plan = plan_template(
        config, snapshot(), original, soccfg=profile_soccfg("ftf"), time_base=TIME_BASE
    )
    assert (plan.batch.flux_waveform is None) == (kind == "step")
    if kind == "sampled":
        assert plan.batch.segment_boundaries == ((2.0, 4.0, 6.0), (2.0, 4.0, 6.0))
    engine = Engine(1)
    try:
        result = engine.evolve(plan.source, (plan.batch,))
    finally:
        engine.close()
    assert result.paths.segments_by_kernel[family] > 0


def test_public_environment_admits_bound_flux_generator_through_acquire() -> None:
    env = ftf(
        flux_settings().with_telemetry(enabled=False),
        soccfg=profile_soccfg("ftf"),
        time_scale=0,
    )
    try:
        pulse = PulseCfg(
            ch=6,
            nqz=1,
            freq=0.0,
            gain=0.04,
            waveform=ConstWaveformCfg(length=0.008),
            post_delay=0.004,
        )
        readout = PulseReadoutCfg(
            gen_ch=0,
            ro_ch=0,
            nqz=1,
            freq=6000.0,
            gain=0.1,
            waveform=ConstWaveformCfg(length=0.008),
            ro_length=0.008,
            trig_offset=0.0,
        )
        program = compile_modules(
            env.soccfg,
            modules=[pulse.build("flux"), readout.build("ro")],
            reps=1,
        )
        acquired = program.acquire(env.soc, progress=False)
        assert acquired is not None
        assert np.isfinite(acquired[0]).all()
        assert env.soc.binding.rounds
    finally:
        if env.close_backend is not None:
            env.close_backend()
        env.devices.close_all_devices()


def test_cavity_boundaries_merge_with_delayed_flux_events() -> None:
    base = template()
    earlier = replace(
        base.events[-1],
        start_ticks=tick_array(np.zeros(2)),
        duration_ticks=tick_array(np.full(2, 0.002)),
        envelopes=(None,),
    )
    original = replace(base, events=(flux_slot(4, 0.02), earlier, *base.events))
    plan = plan_template(
        flux_settings(),
        snapshot(),
        original,
        soccfg=profile_soccfg("ftf"),
        time_base=TIME_BASE,
    )
    assert plan.batch.segment_boundaries == ((0.0, 2.0, 6.0), (0.0, 2.0, 6.0))


def test_projective_and_cavity_boundaries_preserve_filtered_flux_splits() -> None:
    herald, readout = herald_template()
    shifted = replace(
        herald,
        events=tuple(
            replace(event, start_ticks=event.start_ticks + ticks(0.020))
            for event in herald.events
        ),
        readouts=tuple(
            replace(window, trigger_ticks=window.trigger_ticks + ticks(0.020))
            for window in herald.readouts
        ),
    )
    cavity = replace(
        shifted.events[0],
        start_ticks=tick_array(np.array([0.010, 0.012])),
        envelopes=(None,),
    )
    observed = replace(shifted, events=(cavity, *shifted.events))
    config = flux_settings(tau=2.0)
    facts = FTFModelFacts(config, snapshot())
    flux_only = plan_template(
        config,
        snapshot(),
        replace(
            template(),
            events=(flux_slot(4, 0.2), *shifted.events[-2:]),
            readouts=tuple(
                replace(
                    window,
                    trigger_ticks=window.trigger_ticks[:, -1:],
                    frequency_mhz=window.frequency_mhz[:, -1:],
                    phase_deg=window.phase_deg[:, -1:],
                    generator_channels=window.generator_channels[:, -1:],
                )
                for window in shifted.readouts
            ),
        ),
        soccfg=profile_soccfg("ftf"),
        readout=readout,
        facts=facts,
        time_base=TIME_BASE,
    )
    projection_only = plan_template(
        config,
        snapshot(),
        observed,
        soccfg=profile_soccfg("ftf"),
        readout=readout,
        facts=facts,
        time_base=TIME_BASE,
    )
    combined = plan_template(
        config,
        snapshot(),
        replace(observed, events=(flux_slot(4, 0.2), *observed.events)),
        soccfg=profile_soccfg("ftf"),
        readout=readout,
        facts=facts,
        time_base=TIME_BASE,
    )
    # The delayed 0..4 ns flux pulse must still split at 2 and 6 ns.
    assert all({2.0, 6.0} <= set(nodes) for nodes in flux_only.batch.segment_boundaries)
    expected = tuple(
        tuple(sorted(set(flux_nodes) | set(read_nodes)))
        for flux_nodes, read_nodes in zip(
            flux_only.batch.segment_boundaries,
            ((10.0, 12.0, 20.0), (12.0, 14.0, 22.0)),
            strict=True,
        )
    )
    assert combined.batch.segment_boundaries == expected
    spans = segment_spans(combined.batch)
    for point, projection_start in enumerate((20.0, 22.0)):
        assert {2.0, 6.0, projection_start} <= {start for start, _ in spans[point]}
        assert [read.start_ns for read in combined.projective_reads[point]] == [
            projection_start
        ]
    assert combined.projective_reads == projection_only.projective_reads
    assert combined.cavity_drives == projection_only.cavity_drives
    for actual, original in zip(
        combined.readouts, projection_only.readouts, strict=True
    ):
        for events, expected_events in zip(
            actual.batch.events, original.batch.events, strict=True
        ):
            for event, expected_event in zip(events, expected_events, strict=True):
                assert event.pulses == expected_event.pulses
                for grid, expected_grid in zip(
                    event.grids, expected_event.grids, strict=True
                ):
                    np.testing.assert_array_equal(grid.times_ns, expected_grid.times_ns)
                    assert (
                        grid.demodulation_frequency_ghz
                        == expected_grid.demodulation_frequency_ghz
                    )
                    assert grid.phase_rad == expected_grid.phase_rad
    assert [
        tuple(len(event.grids) for event in channel.batch.events[0])
        for channel in combined.readouts
    ] == [(1, 1), (0, 1)]
    np.testing.assert_array_equal(
        combined.batch.flux.offsets, flux_only.batch.flux.offsets
    )
    np.testing.assert_array_equal(
        combined.batch.flux.start_ns, flux_only.batch.flux.start_ns
    )
    np.testing.assert_array_equal(combined.batch.flux.level, flux_only.batch.flux.level)
    waveform = combined.batch.flux_waveform
    original_waveform = flux_only.batch.flux_waveform
    assert waveform is not None and original_waveform is not None
    np.testing.assert_array_equal(waveform.time_ns, original_waveform.time_ns)
    np.testing.assert_array_equal(waveform.flux, original_waveform.flux)


def test_unbound_flux_configuration_preserves_elementwise_batch_contract() -> None:
    plan = plan_template(
        settings(),
        snapshot(),
        template(tones=1),
        soccfg=profile_soccfg("ftf"),
        time_base=TIME_BASE,
    )
    batch = plan.batch
    np.testing.assert_array_equal(batch.points, [0, 1])
    np.testing.assert_array_equal(batch.end_ns, [8.0, 10.0])
    np.testing.assert_array_equal(batch.flux.offsets, [0, 1, 2])
    np.testing.assert_array_equal(batch.flux.start_ns, [0.0, 0.0])
    np.testing.assert_array_equal(batch.flux.level, np.zeros((2, 3), dtype=np.int64))
    assert batch.flux_waveform is None
    assert batch.segment_boundaries == batch.segment_shifts == ()
    assert batch.start_ns is batch.initial_densities is None
    assert len(batch.tracks) == 1
    track = batch.tracks[0]
    assert track.port == "mw"
    np.testing.assert_array_equal(track.start_ns, [0.0, 2.0])
    np.testing.assert_array_equal(track.duration_ns, [2.0, 2.0])
    np.testing.assert_array_equal(track.frequency_ghz, [0.25, 0.25])
    np.testing.assert_array_equal(track.phase_rad, [0.0, 0.0])
    np.testing.assert_array_equal(track.amplitude, [-0.2, -0.2])
    np.testing.assert_array_equal(track.envelope_index, [0, 0])
    np.testing.assert_array_equal(track.active, [True, True])
    assert len(track.envelopes) == 1
    assert track.envelopes[0].kind == "constant"
