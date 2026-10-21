"""One-cavity readout contracts across deterministic and physical-shot APIs."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp, trapezoid
from zcu_tools.core.simulation import SamplingIdentity
from zcu_tools.simulate import adc


def _model(*, jumping=False, identical=False, mismatch=True):
    frequencies = np.array([[6.0], [6.003], [5.997]])
    if identical:
        frequencies[:] = 6.0
    rates = np.array([[0.0, 0.03, 0.0], [0.02, 0.0, 0.04], [0.0, 0.01, 0.0]])
    phases = np.array([0.37]) if mismatch else np.zeros(1)
    try:
        return adc.ReadoutModel(
            "one-cavity",
            ("g", "e", "f"),
            frequencies,
            6.01,
            np.array([0.02]),
            np.array([0.06]),
            rates if jumping else np.zeros((3, 3)),
            adc.Receiver(
                gain=17.0,
                phase_rad=0.23,
                offset=np.array([0.2, -0.4]),
                time_of_flight_ns=1.5,
            ),
            mismatch_phase_rad=phases,
        )
    except ValueError as error:
        if str(error) == "frequencies_ghz must have shape (3, 2)":
            pytest.fail(f"public readout rejects a one-cavity model: {error}")
        raise


def _pulses(kind):
    knots = np.array([0.0, 5.0, 13.0, 20.0])
    values = np.array([0.1, 0.8, 0.3j, 0.1])
    envelope = adc.PiecewiseLinearEnvelope(knots, values)
    if kind == "flat":
        envelope = None
    elif kind == "opaque":

        def opaque(t):
            return complex(np.interp(t, knots, values))

        envelope = opaque
    return (
        adc.ReadoutPulse(1.0, 21.0, 0.3j, 6.01, 0.2, envelope=envelope),
        adc.ReadoutPulse(1.0, 29.0, 0.2, 6.015, -0.3),
    )


def _batch(model, pulses, *, include_traces):
    grids = (
        adc.SampleGrid(np.arange(0.0, 40.0, 2.0), 6.008, 0.17),
        adc.SampleGrid(np.arange(20.0, 60.0, 2.0), 6.013, -0.21),
    )
    return adc.ReadoutBatch(
        model, (pulses,), (grids,), np.array([0], dtype=np.int64), include_traces
    )


def _reference_fields(model, pulses, times, initial=0j, origin=0.0):
    # Independent scalar oscillator equations, with pulse corners split exactly.
    decay = 0.5 * (model.kappa_internal_per_ns[0] + model.kappa_external_per_ns[0])
    generator = decay + 2j * np.pi * (
        model.frequencies_ghz[:, 0] - model.reference_frequency_ghz
    )
    coupling = np.sqrt(model.kappa_external_per_ns[0] / 2)
    corners = sorted(
        {origin, float(times[-1])}
        | {t for p in pulses for t in (p.start_ns, p.end_ns) if origin < t < times[-1]}
    )
    fields = np.empty((len(times), len(generator)), dtype=complex)
    current = np.full(len(generator), initial, dtype=complex)
    fields[times == origin] = current
    for left, right in zip(corners[:-1], corners[1:], strict=True):
        stop = np.nextafter(right, left)

        def rhs(t, field, stop=stop):
            t = min(t, stop)
            incident = sum(p.incident(t, model.reference_frequency_ghz) for p in pulses)
            return -generator * field + coupling * incident

        solution = solve_ivp(
            rhs,
            (left, right),
            current,
            method="DOP853",
            dense_output=True,
            rtol=1e-11,
            atol=1e-13,
            max_step=0.1,
        )
        assert solution.success
        selected = (times > left) & (times <= right)
        if np.any(selected):
            fields[selected] = solution.sol(times[selected]).T
        current = solution.y[:, -1]
    return fields


@pytest.mark.parametrize("kind", ["flat", "sampled", "opaque"])
def test_one_cavity_conditional_mean_and_decimated_traces_follow_scalar_line(kind):
    model = _model()
    pulses = _pulses(kind)
    batch = _batch(model, pulses, include_traces=True)
    solver = adc.ConditionalReadout(model)
    result = solver.responses(batch)
    assert result.traces is not None
    for read, grid in enumerate(batch.grids[0]):
        emitted = grid.times_ns - model.receiver.time_of_flight_ns
        selected = emitted >= 0
        fields = np.zeros((len(emitted), 3), dtype=complex)
        fields[selected] = _reference_fields(model, pulses, emitted[selected])
        incident = np.array(
            [
                sum(p.incident(t, model.reference_frequency_ghz) for p in pulses)
                for t in emitted
            ]
        )
        scattered = np.sqrt(0.06 / 2) * np.exp(-0.37j) * fields
        oscillator = np.exp(
            2j * np.pi * (grid.demodulation_frequency_ghz - 6.01) * grid.times_ns
            - 1j * grid.phase_rad
            + 2j * np.pi * 6.01 * 1.5
        )
        signal = np.conj((incident[:, None] - scattered) * oscillator[:, None])
        signal *= 17 * np.exp(0.23j)
        expected = np.stack((signal.real.T, signal.imag.T), axis=-1) + [0.2, -0.4]
        np.testing.assert_allclose(
            result.traces[0, read], expected, rtol=1e-8, atol=2e-8
        )
        np.testing.assert_allclose(
            solver.response(pulses, grid).traces, expected, atol=2e-8
        )
    integrated = solver.responses(replace(batch, include_traces=False))
    np.testing.assert_allclose(integrated.sums, result.traces.sum(axis=-2), atol=2e-8)
    statistics = adc.round_statistics(
        np.array([[0.0, 1.0, 0.0]]),
        (
            adc.ChannelReadout(
                4, integrated, np.array([0], dtype=np.int64), model.receiver
            ),
        ),
        reps=100,
        seed=12,
        acquire_sequence=0,
        round_index=0,
    )
    np.testing.assert_array_equal(statistics.state_counts, [[0, 100, 0]])
    channel = statistics.channels[0]
    assert channel.shot_covariance.shape == (2, 2)
    assert channel.sums.shape == (1, 2, 2)
    stderr = np.sqrt((channel.shot_covariance.diagonal() + 1 / 12) / 100)
    assert np.all(
        np.abs(channel.sums[0] / 100 - integrated.sums[0, :, 1]) <= 6 * stderr
    )


@pytest.mark.parametrize("kind", ["flat", "sampled", "opaque"])
@pytest.mark.parametrize("jumping", [False, True], ids=["zero-rates", "ctmc"])
def test_one_cavity_raw_and_trace_paths_share_physical_memory(kind, jumping):
    # Identical state frequencies make every jumping path have an exact oracle.
    model = _model(jumping=jumping, identical=True)
    pulses = _pulses(kind)
    batch = _batch(model, pulses, include_traces=True)
    expected = adc.ConditionalReadout(model).responses(batch)
    assert expected.traces is not None
    identity = SamplingIdentity(71, (2, 3, 1))
    probabilities = np.array([[0.0, 1.0, 0.0]])
    traces = adc.sample_traces(probabilities, {4: batch}, reps=5, identity=identity)
    values = traces.channels[0].iq
    assert values.shape == (5, 2, 20, 2)
    np.testing.assert_allclose(
        values, np.broadcast_to(expected.traces[0, :, 1], values.shape), atol=2e-8
    )
    source = adc.PreparedRawShots(probabilities, {4: batch}, reps=5).bind(identity)
    raw = source.materialize(5, 0)[4]
    assert raw.shape == (5, 1, 2, 2)
    np.testing.assert_array_equal(
        raw[:, 0], np.broadcast_to(np.rint(expected.sums[0, :, 1]), (5, 2, 2))
    )
    np.testing.assert_array_equal(source.materialize(5, 0)[4], raw)
    population = adc.PopulationInput(model.state_labels, probabilities[0])
    windows = adc.ADC(model).sample_windows(
        population, pulses, batch.grids[0], identity, shots=5
    )
    for read, window in enumerate(windows):
        np.testing.assert_allclose(window.traces, values[:, read], atol=2e-8)
        np.testing.assert_array_equal(window.sums, window.traces.sum(axis=1))
    single = adc.ADC(model).sample(
        population, pulses, batch.grids[0][0], identity, shots=5
    )
    np.testing.assert_allclose(single.traces, windows[0].traces, atol=2e-8)


def test_one_cavity_state_dependent_jumps_match_conditional_first_moment():
    base = _model(jumping=True)
    model = replace(base, receiver=replace(base.receiver, gain=1700))
    pulses = _pulses("flat")
    batch = _batch(model, pulses, include_traces=False)
    source = adc.PreparedRawShots(
        np.array([[0.0, 1.0, 0.0]]), {4: batch}, reps=1200
    ).bind(SamplingIdentity(52))
    raw = source.materialize(1200, 0)[4][:, 0]
    expected = adc.ConditionalReadout(model).responses(batch).sums[0, :, 1]
    assert np.unique(raw[:, 0], axis=0).shape[0] > 20
    assert np.all(
        np.abs(raw.mean(axis=0) - expected)
        <= 6 * raw.std(axis=0) / np.sqrt(1200) + 0.51
    )


@pytest.mark.parametrize("kind", ["flat", "sampled", "opaque"])
def test_one_cavity_fields_and_products_keep_nonzero_origin_and_phase(kind):
    model = _model()
    pulses = _pulses(kind)
    initial = np.array([0.2 + 0.3j])
    times = np.linspace(5.0, 45.0, 2001)
    expected = _reference_fields(model, pulses, times, initial[0], 5.0)
    actual = adc.cavity_fields(
        model, pulses, times, initial_field=initial, initial_time_ns=5.0
    )
    assert actual.shape == (2001, 3, 1)
    np.testing.assert_allclose(actual[..., 0], expected, atol=2e-9)
    average = adc.cavity_field_average(
        model, pulses, 5.0, 45.0, initial_field=initial, initial_time_ns=5.0
    )
    products = (
        trapezoid(expected[:, :, None] * expected[:, None, :].conj(), times, axis=0)
        / 40
    )
    assert average.field_products.shape == (3, 1, 3, 1)
    np.testing.assert_allclose(
        average.field_products[:, 0, :, 0], products, rtol=2e-5, atol=2e-7
    )
    np.testing.assert_allclose(
        average.photons[:, 0], products.diagonal().real, rtol=2e-5, atol=2e-7
    )


@pytest.mark.parametrize("kind", ["flat", "sampled", "opaque"])
def test_one_cavity_initial_field_continuation_and_empty_ringdown(kind):
    model = _model(identical=True)
    pulses = _pulses(kind)
    initial = np.array([0.2 + 0.3j])
    whole = adc.cavity_fields(
        model,
        pulses,
        np.array([5.0, 18.0, 45.0]),
        initial_field=initial,
        initial_time_ns=5.0,
    )
    resumed = adc.cavity_fields(
        model,
        pulses,
        np.array([18.0, 45.0]),
        initial_field=whole[1, 0],
        initial_time_ns=18.0,
    )
    np.testing.assert_allclose(resumed, whole[1:], atol=2e-9)
    empty = adc.cavity_field_average(
        model, (), 5.0, 45.0, initial_field=initial, initial_time_ns=5.0
    )
    expected = abs(initial[0]) ** 2 * -np.expm1(-0.08 * 40) / (0.08 * 40)
    np.testing.assert_allclose(empty.photons, np.full((3, 1), expected), atol=1e-12)
    with pytest.raises(ValueError, match="initial field"):
        adc.cavity_fields(
            model,
            (),
            np.array([5.0]),
            initial_field=np.zeros(2, dtype=complex),
            initial_time_ns=5.0,
        )


def test_one_cavity_explicit_zero_mismatch_is_owned_zero_array():
    model = _model(mismatch=False)
    np.testing.assert_array_equal(model.mismatch_phase_rad, [0.0])
    assert not model.mismatch_phase_rad.flags.writeable
    explicit = replace(model, mismatch_phase_rad=np.array([0.0]))
    grid = adc.SampleGrid(np.arange(0.0, 40.0, 2.0), 6.01)
    np.testing.assert_array_equal(
        adc.ADC(model).response(_pulses("flat"), grid).traces,
        adc.ADC(explicit).response(_pulses("flat"), grid).traces,
    )


@pytest.mark.parametrize("cavities", [0, 3])
def test_readout_rejects_unsupported_cavity_counts(cavities):
    with pytest.raises(ValueError, match="1 or 2"):
        adc.ReadoutModel(
            "unsupported",
            ("g",),
            np.zeros((1, cavities)),
            6.0,
            np.zeros(cavities),
            np.zeros(cavities),
            np.zeros((1, 1)),
        )
