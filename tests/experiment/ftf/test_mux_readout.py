"""Author-configured mux planning and FTF joint shots, independent of YAML shape."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.device import DeviceManager
from zcu_tools.experiment import ftf as ftf_module
from zcu_tools.program.v2.simulation import IntegratedRound
from zcu_tools.simulate.adc import ADC
from zcu_tools.simulate.evolution import Engine, EvolutionResult, PathDiagnostics

from tests.experiment._ftf_templates import request, settings, snapshot, template

pytestmark = pytest.mark.sim


def configuration(channels=(4, 1)):
    config_type = getattr(ftf_module, "FTFReadoutConfig", None)
    receiver_type = getattr(ftf_module, "FTFReceiverConfig", None)
    assert config_type is not None, "FTF must accept a detached readout configuration"
    assert receiver_type is not None, "FTF must expose per-channel receiver values"
    return config_type(
        "mux",
        tuple(
            receiver_type(
                ro_ch=ch,
                gain=1000.0 + i * 100,
                noise_psd_adc2_ns=((0.0, 0.0), (0.0, 0.0)),
            )
            for i, ch in enumerate(channels)
        ),
    )


def mux_template():
    base = template()
    pulse = base.events[0]
    second = replace(pulse, frequency_mhz=pulse.frequency_mhz + 80)
    first_window = replace(base.readouts[0], channel=4)
    second_window = replace(
        base.readouts[0],
        channel=1,
        frequency_mhz=base.readouts[0].frequency_mhz + 80,
        samples=3,
    )
    return replace(base, events=(pulse, second), readouts=(first_window, second_window))


def test_mux_planning_keeps_every_incident_tone_and_independent_adc_grid():
    plan = ftf_module.plan_template(
        settings(), snapshot(), mux_template(), readout=configuration()
    )
    assert tuple(item.channel for item in plan.readouts) == (4, 1)
    np.testing.assert_array_equal(plan.batch.end_ns, [8.0, 10.0])
    for item, samples, gain in zip(plan.readouts, (4, 3), (1000, 1100), strict=True):
        assert len(item.batch.pulses[0]) == 2
        assert item.batch.pulses[0][0].frequency_ghz == 6.0
        assert item.batch.pulses[0][1].frequency_ghz == 6.08
        assert len(item.batch.grids[0][0].times_ns) == samples
        assert item.batch.model.receiver.gain == gain
    np.testing.assert_array_equal(
        plan.readouts[0].batch.model.frequencies_ghz,
        plan.readouts[1].batch.model.frequencies_ghz,
    )


def test_mux_planner_rejects_unbound_receiver_and_different_pulse_start():
    base = mux_template()
    unbound = replace(base.readouts[0], channel=2)
    with pytest.raises(ValueError, match="unbound.*receiver"):
        ftf_module.plan_template(
            settings(),
            snapshot(),
            replace(base, readouts=(unbound,)),
            readout=configuration(),
        )
    delayed = replace(base.events[1], start_us=base.events[1].start_us + 0.001)
    with pytest.raises(ValueError, match="common.*start|same.*start"):
        ftf_module.plan_template(
            settings(),
            snapshot(),
            replace(base, events=(base.events[0], delayed)),
            readout=configuration(),
        )


class MixedEngine(Engine):
    """Supply a nontrivial joint mixture without performing evolution."""

    def __init__(self):
        super().__init__(1, 0)

    def evolve(
        self, source, batches, *, budget=1e-3, cancel, telemetry=False, reuse_key=None
    ):
        points = np.concatenate([batch.points for batch in batches])
        densities = np.tile(np.diag([0.4, 0.6])[None], (len(points), 1, 1))
        return EvolutionResult(points, densities, PathDiagnostics())


@pytest.mark.parametrize("mode", ["integrated", "decimated"])
@pytest.mark.parametrize("jumping", [False, True])
def test_mux_ftf_channels_share_physical_shots_and_initial_mixture(mode, jumping):
    config = configuration()
    config = replace(
        config, receivers=tuple(replace(r, gain=1e7) for r in config.receivers)
    )
    numeric = settings()
    # A hot, lossy synthetic bath makes repeated paths observable in a short
    # readout. The low-loss case independently protects the initial mixture.
    chip = numeric.chip
    if jumping:
        chip = chip.model_copy(
            update={
                "temperature_k": 1.0,
                "f1": chip.f1.model_copy(update={"q_cap": 10.0}),
                "f2": chip.f2.model_copy(update={"q_cap": 10.0}),
            }
        )
    else:
        chip = chip.model_copy(
            update={
                "coupler": chip.coupler.model_copy(update={"q_cap": 1e30}),
            }
        )
    numeric = numeric.model_copy(update={"chip": chip})
    content = mux_template()
    window = content.readouts[0]
    content = replace(content, readouts=(window, replace(window, channel=1)))
    source = ftf_module.FTFSource(
        numeric, DeviceManager(), engine=MixedEngine(), readout=config
    )
    acquire = replace(request(mode), reps=128, templates=(content,))
    try:
        result = source.open_acquire(acquire).sample_round(snapshot(), 0)
        if isinstance(result, IntegratedRound):
            raw = result.raw.materialize(acquire.reps, acquire.avg_level)
            channel_order = tuple(ch.channel for ch in result.statistics.channels)
            first, second = raw[4], raw[1]
        else:
            channel_order = tuple(ch.channel for ch in result.channels)
            first, second = (ch.iq for ch in result.channels)
        assert channel_order == (4, 1)
        np.testing.assert_array_equal(first, second)
        plan = ftf_module.plan_template(numeric, snapshot(), content, readout=config)
        batch = plan.readouts[0].batch
        if jumping:
            # A fixed conditional centre or shared initial states alone cannot
            # produce this many distinct shots at the same point.
            point_shots = first[:, 0] if mode == "integrated" else first[::2]
            assert np.unique(point_shots.reshape(128, -1), axis=0).shape[0] > 10
        else:
            model = replace(batch.model, rates_per_ns=np.zeros((2, 2)))
            expected = ADC(model).responses(replace(batch, include_traces=True))
            if mode == "integrated":
                centres = np.rint(expected.sums[0, 0])
                observed = first[:, 0, 0]
            else:
                assert expected.traces is not None
                centres = expected.traces[0, 0]
                observed = first[::2, 0]
            distance = np.max(
                np.abs(observed[:, None] - centres[None]),
                axis=tuple(range(2, observed.ndim + 1)),
            )
            nearest = distance.argmin(axis=1)
            assert set(nearest) == {0, 1}
            assert 0.25 < (nearest == 0).mean() < 0.55
    finally:
        source.close()
