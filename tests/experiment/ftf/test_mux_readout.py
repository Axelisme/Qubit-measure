"""Author-configured mux planning and FTF joint shots, independent of YAML shape."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.core.simulation import RoundStatistics
from zcu_tools.device import DeviceManager
from zcu_tools.experiment import ftf as ftf_module
from zcu_tools.simulate.adc import ConditionalReadout, ConditionalSums
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
def test_mux_ftf_channels_decode_to_the_same_joint_state_per_shot(mode, monkeypatch):
    def distinguishable_response(self, batch):
        # Each receiver encodes the joint state differently. A per-channel
        # sampling loop would produce mismatched decoded states.
        code = 10 if batch.model.receiver.gain == 1000 else 100
        samples = len(batch.grids[0][0].times_ns)
        traces = np.zeros((len(batch.pulses), 1, 2, samples, 2))
        traces[:, :, 0, :, 0] = code
        traces[:, :, 1, :, 0] = 2 * code
        return ConditionalSums(
            traces.sum(axis=-2),
            np.array([samples], dtype=np.int64),
            traces if batch.include_traces else None,
        )

    monkeypatch.setattr(ConditionalReadout, "responses", distinguishable_response)
    source = ftf_module.FTFSource(
        settings(), DeviceManager(), engine=MixedEngine(), readout=configuration()
    )
    acquire = replace(request(mode), reps=128, templates=(mux_template(),))
    try:
        result = source.open_acquire(acquire).sample_round(snapshot(), 0)
        if isinstance(result, RoundStatistics):
            raw = result.materialize_raw(acquire.reps, acquire.avg_level)
            first = raw[4].reshape(-1, 2)[:, 0] / 40
            second = raw[1].reshape(-1, 2)[:, 0] / 300
        else:
            first = result.channels[0].iq[:, 0, 0, 0] / 10
            second = result.channels[1].iq[:, 0, 0, 0] / 100
        assert tuple(ch.channel for ch in result.channels) == (4, 1)
        assert set(first) == {1, 2}, "the test must exercise both joint states"
        np.testing.assert_array_equal(first, second)
    finally:
        source.close()
