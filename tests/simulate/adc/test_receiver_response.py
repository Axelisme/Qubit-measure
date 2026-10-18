"""Microwave IQ, causal line delay and per-resonator scattering at public seams."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import expm
from zcu_tools.core.simulation import SamplingIdentity
from zcu_tools.simulate import adc


def line_model() -> adc.ReadoutModel:
    return adc.ReadoutModel(
        "receiver-contract",
        ("g",),
        np.array([[6.001, 6.013]]),
        6.0,
        np.array([0.03, 0.05]),
        np.array([0.08, 0.06]),
        np.zeros((1, 1)),
    )


def traces(model, pulse, grid, path):
    reader = adc.ConditionalReadout(model)
    if path == "single":
        return reader.response((pulse,), grid).traces
    if path == "shaped":
        pulse = replace(pulse, envelope=lambda local: 1.0)
    result = reader.responses(
        adc.ReadoutBatch(
            model, ((pulse,),), ((grid,),), np.array([0], dtype=np.int64), True
        )
    )
    if result.traces is None:
        raise AssertionError("requested conditional traces must be returned")
    assert_allclose(result.sums[0, 0], result.traces[0, 0].sum(axis=-2))
    return result.traces[0, 0]


@pytest.mark.parametrize("path", ["single", "constant", "shaped"])
def test_microwave_iq_conjugates_field_before_receiver_calibration(path):
    model = replace(
        line_model(),
        receiver=adc.Receiver(gain=2.3, phase_rad=0.31, offset=np.array([0.4, -0.2])),
    )
    pulse = adc.ReadoutPulse(0.0, 20.0, 1.2 + 0.3j, 6.0, 0.2)
    grid = adc.SampleGrid(np.array([-1.0, 0.0, 4.0, 12.0, 20.0, 28.0]), 6.003, 0.1)
    # Closed-form driven two-cavity field, with reciprocal shared-line damping.
    v = np.sqrt(np.array([0.08, 0.06]) / 2)
    a = np.diag([0.015 + 2j * np.pi * 0.001, 0.025 + 2j * np.pi * 0.013])
    a += np.outer(v, v)
    amplitude = (1.2 + 0.3j) * np.exp(0.2j)
    expected = []
    for time in grid.times_ns:
        duration = max(0.0, min(time, 20.0))
        cavity = np.linalg.solve(a, (np.eye(2) - expm(-a * duration)) @ (v * amplitude))
        if time > 20:
            cavity = expm(-a * (time - 20)) @ cavity
        outgoing = (amplitude if 0 <= time < 20 else 0j) - v @ cavity
        microwave = np.conj(outgoing * np.exp(2j * np.pi * 0.003 * time - 0.1j))
        value = 2.3 * np.exp(0.31j) * microwave
        expected.append([value.real + 0.4, value.imag - 0.2])
    assert_allclose(traces(model, pulse, grid, path)[0], expected, atol=2e-9, rtol=0)


@pytest.mark.parametrize("path", ["single", "constant", "shaped"])
@pytest.mark.parametrize("phases", [(0.6, 0.0), (0.0, -0.4), (0.6, -0.4)])
def test_mismatch_rotates_each_scattered_field_without_changing_dynamics(path, phases):
    model = replace(line_model(), mismatch_phase_rad=np.array(phases))
    pulse = adc.ReadoutPulse(0, 70, 1.0, 6.0)
    grid = adc.SampleGrid(np.array([0.0, 12.0, 35.0, 70.0, 80.0]), 6.0)
    v = np.sqrt(np.array([0.08, 0.06]) / 2)
    a = np.diag([0.015 + 2j * np.pi * 0.001, 0.025 + 2j * np.pi * 0.013])
    a += np.outer(v, v)
    expected = []
    for time in grid.times_ns:
        cavity = np.linalg.solve(a, (np.eye(2) - expm(-a * min(time, 70))) @ v)
        if time > 70:
            cavity = expm(-a * (time - 70)) @ cavity
        # Hanger uses +phi in microwave IQ; internal fields use its conjugate.
        microwave = (1.0 if time < 70 else 0.0) - np.sum(
            v * np.exp(1j * np.array(phases)) * cavity.conj()
        )
        expected.append([microwave.real, microwave.imag])
    assert_allclose(traces(model, pulse, grid, path)[0], expected, atol=2e-9, rtol=0)


@pytest.mark.parametrize("path", ["single", "constant", "shaped"])
@pytest.mark.parametrize("reference", [6.0, 6.008])
def test_time_of_flight_delays_onset_ringdown_and_absolute_demodulated_phase(
    path, reference
):
    base = replace(line_model(), reference_frequency_ghz=reference)
    delay = 3.25
    delayed = replace(base, receiver=adc.Receiver(time_of_flight_ns=delay))
    pulse = adc.ReadoutPulse(0, 20, 1.0 + 0.2j, 6.004, 0.3)
    times = np.array([-1.0, 0.0, 2.0, 3.25, 7.0, 20.0, 23.25, 30.0])
    grid = adc.SampleGrid(times, 6.007, 0.2)
    source_grid = replace(grid, times_ns=times - delay)
    source_iq = traces(base, pulse, source_grid, path)[0]
    source = source_iq[:, 0] + 1j * source_iq[:, 1]
    expected = source * np.exp(-2j * np.pi * 6.007 * delay)
    actual = traces(delayed, pulse, grid, path)[0]
    assert_allclose(actual[:, 0] + 1j * actual[:, 1], expected, atol=2e-9, rtol=0)
    assert_allclose(actual[:3], 0.0, atol=0, rtol=0)
    assert np.linalg.norm(actual[-1]) > 0.01, "ringdown must also arrive late"


@pytest.mark.parametrize("reference", [6.0, 6.12])
def test_time_of_flight_has_negative_absolute_carrier_slope(reference):
    base = replace(
        line_model(),
        reference_frequency_ghz=reference,
        kappa_external_per_ns=np.zeros(2),
    )
    delayed = replace(base, receiver=adc.Receiver(time_of_flight_ns=400.0))
    values = []
    for frequency in (6.121, 6.122):
        pulse = adc.ReadoutPulse(0, 450, 1.0, frequency)
        grid = adc.SampleGrid(np.array([425.0]), frequency)
        iq = traces(delayed, pulse, grid, "constant")[0, 0]
        value = iq[0] + 1j * iq[1]
        assert_allclose(value, np.exp(-2j * np.pi * frequency * 400), atol=3e-11)
        values.append(value)
    assert_allclose(np.angle(values[1] / values[0]), -2.5132741228718345, atol=3e-11)


@pytest.mark.parametrize(
    ("path", "shaped"),
    [("single", False), ("batch", False), ("single", True), ("batch", True)],
)
def test_same_physical_pulse_has_reference_independent_iq_and_delay_factor(
    path, shaped
):
    start = 13.25
    delay = 3.25
    times = start + np.array([-1.0, 0.0, 2.0, 3.25, 7.0, 20.0, 23.25, 30.0])
    grid = adc.SampleGrid(times, 6.007, 0.2)
    observed = []
    for reference in (6.0, 6.008):
        model = replace(line_model(), reference_frequency_ghz=reference)
        # The rotating phase at pulse start changes with the reference so the
        # physical carrier phase (phase - 2*pi*f_ref*start) stays fixed.
        pulse = adc.ReadoutPulse(
            start,
            start + 20.0,
            1.0 + 0.2j,
            6.004,
            0.3 + 2 * np.pi * (reference - 6.0) * start,
            envelope=(lambda local: 1 + 0.01j * local) if shaped else None,
        )
        delayed = replace(model, receiver=adc.Receiver(time_of_flight_ns=delay))
        source_grid = replace(grid, times_ns=times - delay)
        source_iq = traces(model, pulse, source_grid, path)[0]
        delayed_iq = traces(delayed, pulse, grid, path)[0]
        source = source_iq[:, 0] + 1j * source_iq[:, 1]
        actual = delayed_iq[:, 0] + 1j * delayed_iq[:, 1]
        assert_allclose(
            actual, source * np.exp(-2j * np.pi * 6.007 * delay), atol=2e-9, rtol=0
        )
        assert_allclose(actual[:3], 0.0, atol=0, rtol=0)
        observed.append((source, actual))
    assert_allclose(observed[0], observed[1], atol=2e-9, rtol=0)


@pytest.mark.parametrize("path", ["single", "constant"])
def test_nonzero_start_delay_ratio_has_absolute_slope_in_changed_reference(path):
    start = 13.25
    ratios_by_reference = []
    for reference in (6.0, 6.12):
        base = replace(
            line_model(),
            reference_frequency_ghz=reference,
            kappa_external_per_ns=np.zeros(2),
        )
        delayed = replace(base, receiver=adc.Receiver(time_of_flight_ns=400.0))
        ratios = []
        for frequency in (6.121, 6.122):
            pulse = adc.ReadoutPulse(
                start,
                start + 450,
                1.0,
                frequency,
                0.3 + 2 * np.pi * (reference - 6.0) * start,
            )
            grid = adc.SampleGrid(np.array([start + 425]), frequency)
            original = traces(base, pulse, grid, path)[0, 0]
            shifted = traces(delayed, pulse, grid, path)[0, 0]
            ratio = (shifted[0] + 1j * shifted[1]) / (original[0] + 1j * original[1])
            assert_allclose(ratio, np.exp(-2j * np.pi * frequency * 400), atol=3e-11)
            ratios.append(ratio)
        assert_allclose(
            np.angle(ratios[1] / ratios[0]), -2.5132741228718345, atol=3e-11
        )
        ratios_by_reference.append(ratios)
    assert_allclose(ratios_by_reference[0], ratios_by_reference[1], atol=3e-11)


def test_shaped_envelope_is_delayed_before_receiver_offset_and_noise():
    receiver = adc.Receiver(
        phase_rad=0.4,
        offset=np.array([0.2, -0.3]),
        noise_covariance=np.full((2, 2), 0.04),
        time_of_flight_ns=4.0,
    )
    model = replace(line_model(), receiver=receiver, kappa_external_per_ns=np.zeros(2))
    pulse = adc.ReadoutPulse(0, 12, 1.0, 6.0, envelope=lambda local: 1 + 0.1j * local)
    grid = adc.SampleGrid(np.array([0.0, 2.0, 4.0, 7.0, 15.0, 16.0]), 6.0)
    result = adc.ConditionalReadout(model).response((pulse,), grid)
    expected = np.array([0j, 0j, 1 + 0j, 1 + 0.3j, 1 + 1.1j, 0j]).conj() * np.exp(0.4j)
    assert_allclose(
        result.traces[0], np.column_stack((expected.real, expected.imag)) + [0.2, -0.3]
    )
    inputs = adc.ReadoutBatch(
        model, ((pulse,),), ((grid,),), np.array([0], dtype=np.int64), True
    )
    reader = adc.ConditionalReadout(model)
    batched = reader.responses(inputs)
    assert batched.traces is not None
    assert_allclose(batched.traces[0, 0], result.traces)
    sampled = adc.ADC(model).sample(
        adc.PopulationInput(("g",), np.array([1.0])),
        (pulse,),
        grid,
        SamplingIdentity(123, (0,)),
        shots=8,
    )
    residual = sampled.traces - result.traces
    assert_allclose(residual[..., 0], residual[..., 1], atol=2e-15)


@pytest.mark.parametrize("parameter", ["delay", "r1", "r2"])
def test_response_cache_distinguishes_receiver_delay_and_each_mismatch(parameter):
    original = line_model()
    reader = adc.ConditionalReadout(original)
    pulse = adc.ReadoutPulse(0, 20, 1.0, 6.0)
    grid = adc.SampleGrid(np.array([1.0, 8.0, 25.0]), 6.0)
    if parameter == "delay":
        changed = replace(original, receiver=adc.Receiver(time_of_flight_ns=3.0))
    else:
        changed = replace(
            original,
            mismatch_phase_rad=np.array(
                [0.6, 0.0] if parameter == "r1" else [0.0, 0.6]
            ),
        )
    inputs = adc.ReadoutBatch(
        original, ((pulse,),), ((grid,),), np.array([0], dtype=np.int64), True
    )
    before = reader.responses(inputs)
    after = reader.responses(replace(inputs, model=changed))
    fresh = adc.ConditionalReadout(changed).responses(replace(inputs, model=changed))
    assert not np.allclose(before.sums, after.sums)
    assert_allclose(after.sums, fresh.sums, atol=0, rtol=0)
    assert_allclose(reader.responses(inputs).sums, before.sums, atol=0, rtol=0)


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_numeric_receiver_rejects_invalid_time_of_flight(value):
    with pytest.raises(ValueError, match="time_of_flight_ns"):
        adc.Receiver(time_of_flight_ns=value)


@pytest.mark.parametrize("value", [np.array([0.0]), np.array([0.0, np.inf])])
def test_numeric_model_rejects_invalid_mismatch(value):
    with pytest.raises(ValueError, match="mismatch_phase_rad"):
        replace(line_model(), mismatch_phase_rad=value)
