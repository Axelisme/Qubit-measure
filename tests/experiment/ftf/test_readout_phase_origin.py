"""Common-origin FTF carriers retain their IQ phase across readout delays."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.experiment.ftf import (
    FTFMuxReadoutConfig,
    FTFReadoutConfig,
    FTFReceiverConfig,
    plan_template,
)
from zcu_tools.simulate.adc import ConditionalReadout, ReadoutModel, Receiver

from tests.core._simulation import TIME_BASE, tick_array, ticks
from tests.experiment._ftf_templates import settings, snapshot, template

pytestmark = pytest.mark.sim


def isolated_carrier_plan(
    addressed_tone: int | None,
    phase_reference_us: float,
    *,
    frequencies: tuple[float, ...] = (6000.0, 6080.0),
):
    """Use flat carriers; None gives every carrier a nonzero incident gain."""
    base = template()
    start = np.array([0, 92 * TIME_BASE.period_ticks(430.08)], dtype=np.int64)
    pulses = tuple(
        replace(
            base.events[0],
            gen_ch=1,
            start_ticks=start,
            duration_ticks=tick_array(np.full(2, 1.0)),
            frequency_mhz=np.full(2, frequency),
            phase_deg=np.full(2, 23.0),
            phase_reference_ticks=ticks(phase_reference_us),
            gain=np.full(2, 0.2 if addressed_tone is None or tone == addressed_tone else 0.0),
            envelopes=(None,),
        )
        for tone, frequency in enumerate(frequencies)
    )
    windows = tuple(
        replace(
            base.readouts[0],
            channel=tone + 1,
            generator_channels=np.ones((2, 1), dtype=np.int64),
            trigger_ticks=(start + 68750)[:, None],  # 22 tProc cycles.
            frequency_mhz=np.full((2, 1), frequency),
            phase_deg=np.full((2, 1), 11.0),
            samples=69,
            sample_period_ticks=TIME_BASE.period_ticks(76.8),
        )
        for tone, frequency in enumerate(frequencies)
    )
    config = FTFReadoutConfig(
        pulse=FTFReceiverConfig(0, 100, ((0.0, 0.0), (0.0, 0.0))),
        mux=FTFMuxReadoutConfig(
            gen_ch=1,
            input_per_gain=1.0,
            receivers=tuple(
                FTFReceiverConfig(
                    ro_ch=tone + 1,
                    gain=100,
                    noise_psd_adc2_ns=((0.0, 0.0), (0.0, 0.0)),
                )
                for tone in range(len(frequencies))
            ),
        ),
    )
    return plan_template(
        settings(),
        snapshot(),
        replace(base, events=pulses, readouts=windows),
        readout=config,
        time_base=TIME_BASE,
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
            for events in batch.events
            for pulses, grids in ((events[0].pulses, events[0].grids),)
        ]
    )


@pytest.mark.parametrize("route", ["integrated", "ode"])
@pytest.mark.parametrize("tone", [0, 1])
def test_distinct_pfb_channels_are_invariant_under_92_cycle_translation(route, tone):
    batch = classical_batch(isolated_carrier_plan(None, 0.001), tone)
    values = centers(batch, route)
    assert np.linalg.norm(values[0]) > 1
    np.testing.assert_allclose(values[1], values[0], rtol=1e-7, atol=1e-7)


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
    single = replace(
        batch,
        events=tuple(
            (replace(events[0], pulses=(events[0].pulses[0],)),)
            for events in batch.events
        ),
    )
    unconverted = replace(
        single,
        events=tuple(
            (
                replace(
                    events[0],
                    pulses=(replace(events[0].pulses[0], phase_rad=original_phase),),
                ),
            )
            for events in single.events
        ),
    )
    np.testing.assert_array_equal(centers(single, route), centers(unconverted, route))
