"""Common-origin FTF carriers retain their IQ phase across readout delays."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.experiment.ftf import (
    FTFReadoutConfig,
    FTFReceiverConfig,
    plan_template,
)
from zcu_tools.simulate.adc import ConditionalReadout, ReadoutModel, Receiver

from tests.experiment._ftf_templates import settings, snapshot, template


def isolated_carrier_plan(addressed_tone: int, phase_reference_us: float):
    """Keep two flat carriers in the plan, with only the addressed one incident."""
    base = template()
    start = np.array([0.0, 0.302269345238])
    frequencies = (6000.0, 6080.0)
    pulses = tuple(
        replace(
            base.events[0],
            start_us=start,
            duration_us=np.full(2, 1.0),
            frequency_mhz=np.full(2, frequency),
            phase_deg=np.full(2, 23.0),
            phase_reference_us=phase_reference_us,
            gain=np.full(2, 0.2 if tone == addressed_tone else 0.0),
            envelopes=(None,),
        )
        for tone, frequency in enumerate(frequencies)
    )
    windows = tuple(
        replace(
            base.readouts[0],
            channel=tone,
            trigger_us=(start + 0.05115327380952381)[:, None],
            frequency_mhz=np.full((2, 1), frequency),
            phase_deg=np.full((2, 1), 11.0),
            samples=69,
            sample_rate_mhz=76.8,
        )
        for tone, frequency in enumerate(frequencies)
    )
    config = FTFReadoutConfig(
        "mux",
        tuple(
            FTFReceiverConfig(
                ro_ch=tone,
                gain=100,
                noise_psd_adc2_ns=((0.0, 0.0), (0.0, 0.0)),
            )
            for tone in range(2)
        ),
    )
    return plan_template(
        settings(),
        snapshot(),
        replace(base, events=pulses, readouts=windows),
        readout=config,
    )


def classical_batch(plan, tone):
    """Use one classical state and two resonators, without joint evolution."""
    batch = plan.readouts[tone].batch
    model = ReadoutModel(
        "phase-origin",
        ("g",),
        np.array([[5.9998, 6.0801]]),
        batch.model.reference_frequency_ghz,
        np.full(2, 2 * np.pi * 0.0001),
        np.full(2, 2 * np.pi * 0.001),
        np.zeros((1, 1)),
        Receiver(gain=100),
    )
    return replace(batch, model=model)


def centers(batch, route):
    solver = ConditionalReadout(batch.model, rtol=1e-9, atol=1e-11)
    if route == "integrated":
        return solver.responses(batch).sums[:, 0, 0] / 69
    return np.array(
        [
            solver.response(pulses, grids[0]).traces[0].mean(axis=0)
            for pulses, grids in zip(batch.pulses, batch.grids, strict=True)
        ]
    )


@pytest.mark.parametrize("route", ["integrated", "ode"])
@pytest.mark.parametrize("phase_reference_us", [0.0, 0.001])
@pytest.mark.parametrize("tone", [0, 1])
def test_isolated_mux_carrier_center_is_invariant_under_readout_translation(
    route, phase_reference_us, tone
):
    batch = classical_batch(isolated_carrier_plan(tone, phase_reference_us), tone)
    values = centers(batch, route)
    assert np.linalg.norm(values[0]) > 1, "exercise a nonzero conditional center"
    # The other carrier is zero: finite-window cross-tone leakage is not an
    # invariant under translation, but the addressed carrier's phase must be.
    np.testing.assert_allclose(values[1], values[0], rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("route", ["integrated", "ode"])
def test_reference_carrier_output_is_bitwise_equal_to_unconverted_phase(route):
    plan = isolated_carrier_plan(0, 0.001)
    batch = classical_batch(plan, 0)
    # Drop the zero-gain companion for a genuine single-tone response. The
    # existing common-origin phase expression must remain bit-for-bit intact.
    original_phase = float(np.deg2rad(23.0 - 360 * 6000.0 * 0.001))
    single = replace(batch, pulses=tuple((pulses[0],) for pulses in batch.pulses))
    unconverted = replace(
        single,
        pulses=tuple(
            (replace(pulses[0], phase_rad=original_phase),) for pulses in single.pulses
        ),
    )
    np.testing.assert_array_equal(centers(single, route), centers(unconverted, route))
