"""Conditional response reuse and exact constant-pulse physics at the ADC seam."""

from dataclasses import replace

import numpy as np
import pytest
import scipy.linalg
from zcu_tools.simulate import adc


def model() -> adc.ReadoutModel:
    return adc.ReadoutModel(
        "dc",
        ("g", "e", "f"),
        np.array([[6.001, 6.013], [6.004, 6.017], [6.007, 6.021]]),
        6.0,
        np.array([0.03, 0.05]),
        np.array([0.08, 0.06]),
        np.array([[0.0, 0.03, 0.002], [0.01, 0.0, 0.04], [0.001, 0.005, 0.0]]),
        adc.Receiver(gain=2.3, phase_rad=0.31, offset=np.array([0.4, -0.2])),
    )


def batch(m: adc.ReadoutModel, *, pulse=None) -> adc.ReadoutBatch:
    pulse = (
        adc.ReadoutPulse(3.0, 23.0, 1.2 + 0.3j, 6.0, phase_rad=0.2)
        if pulse is None
        else pulse
    )
    grids = (
        adc.SampleGrid(np.array([-4.0, 3.0, 5.1, 12.0, 23.0, 28.0, 43.0]), 6.002, 0.1),
        adc.SampleGrid(np.array([-3.0, 4.0, 6.0, 13.0, 24.0, 29.0, 44.0]), 6.004, 0.3),
    )
    return adc.ReadoutBatch(
        m, ((pulse,),), (grids,), np.array([0, 0], dtype=np.int64), True
    )


@pytest.mark.parametrize("detuned", [False, True])
def test_batched_response_agrees_with_dop853_for_all_reads(detuned, monkeypatch):
    m = model()
    inputs = batch(m)
    if detuned:
        inputs = replace(
            inputs, pulses=((replace(inputs.pulses[0][0], frequency_ghz=6.008),),)
        )
    reference = adc.ConditionalReadout(m, rtol=1e-12, atol=1e-14)
    expected = np.stack(
        [reference.response(inputs.pulses[0], g).traces for g in inputs.grids[0]]
    )
    expm = scipy.linalg.expm
    calls = []

    def observe_expm(matrix):
        calls.append(matrix.shape)
        return expm(matrix)

    monkeypatch.setattr(scipy.linalg, "expm", observe_expm)
    result = adc.ConditionalReadout(m, rtol=1e-12, atol=1e-14).responses(inputs)
    assert result.traces is not None
    np.testing.assert_allclose(result.traces[0], expected, atol=1e-9, rtol=0)
    np.testing.assert_allclose(result.sums[0], expected.sum(axis=-2), atol=1e-9, rtol=0)
    np.testing.assert_array_equal(result.sample_counts, [7, 7])
    assert calls


@pytest.mark.parametrize("carrier_ghz", [5.982, 6.008])
@pytest.mark.parametrize("start_ns", [3.0, -7.3])
def test_detuned_constant_response_uses_no_ode_and_preserves_frame_phases(
    carrier_ghz, start_ns, monkeypatch
):
    m = model()
    pulse = adc.ReadoutPulse(
        start_ns,
        start_ns + 20.0,
        1.2 + 0.3j,
        carrier_ghz,
        phase_rad=0.73,
        breakpoints_ns=(start_ns + 4.2, start_ns + 11.6),
    )
    inputs = batch(m, pulse=pulse)
    inputs = replace(
        inputs,
        grids=(
            tuple(
                replace(g, times_ns=g.times_ns + start_ns - 3.0)
                for g in inputs.grids[0]
            ),
        ),
    )
    reference = adc.ConditionalReadout(m, rtol=1e-12, atol=1e-14)
    expected = np.stack(
        [reference.response((pulse,), g).traces for g in inputs.grids[0]]
    )
    ode_calls = []
    from zcu_tools.simulate.adc import response

    solve_ivp = response.solve_ivp

    def observe_ode(*args, **kwargs):
        ode_calls.append(1)
        return solve_ivp(*args, **kwargs)

    monkeypatch.setattr(response, "solve_ivp", observe_ode)
    result = adc.ConditionalReadout(m).responses(inputs)
    assert not ode_calls, "constant carriers must use expm, not DOP853"
    assert result.traces is not None
    np.testing.assert_allclose(result.traces[0], expected, atol=1e-9, rtol=0)
    np.testing.assert_allclose(result.sums[0], expected.sum(axis=-2), atol=1e-9, rtol=0)


@pytest.mark.parametrize("degenerate", [False, True])
def test_frequency_batch_matches_individual_responses_across_interval_corners(
    degenerate,
):
    m = model()
    if degenerate:
        # Equal resonators leave an undamped dark mode. Equal cascade rates
        # also give a defective population generator; neither needs a cutoff.
        m = replace(
            m,
            frequencies_ghz=np.full((3, 2), 6.0),
            kappa_internal_per_ns=np.zeros(2),
            kappa_external_per_ns=np.full(2, 0.08),
            rates_per_ns=np.array(
                [[0.0, 0.02, 0.0], [0.0, 0.0, 0.02], [0.0, 0.0, 0.0]]
            ),
        )
    frequencies = (5.982, 6.0, 6.0 + 1e-13, 6.008)
    pulses = tuple(
        (
            adc.ReadoutPulse(
                -2.3,
                7.7,
                (1.2 + 0.3j) * (i + 1),
                frequency,
                phase_rad=0.23 * i,
                breakpoints_ns=(-0.2, 4.1),
            ),
        )
        for i, frequency in enumerate(frequencies)
    )
    times = np.array([-4.0, -2.3, -1.0, -0.2, 0.0, 4.1, 5.0, 7.7, 9.0])
    grids = tuple(
        (
            adc.SampleGrid(times, frequency, 0.1),
            adc.SampleGrid(times + 0.1, frequency + 0.002, 0.3),
        )
        for frequency in frequencies
    )
    inputs = adc.ReadoutBatch(m, pulses, grids, np.arange(4, dtype=np.int64), True)
    result = adc.ConditionalReadout(m).responses(inputs)
    individual = [
        adc.ConditionalReadout(m).responses(
            adc.ReadoutBatch(m, (p,), (g,), np.array([0], dtype=np.int64), True)
        )
        for p, g in zip(pulses, grids, strict=True)
    ]
    np.testing.assert_allclose(
        result.sums, np.concatenate([r.sums for r in individual]), atol=1e-10, rtol=0
    )
    assert result.traces is not None
    individual_traces = []
    for r in individual:
        assert r.traces is not None
        individual_traces.append(r.traces)
    np.testing.assert_allclose(
        result.traces,
        np.concatenate(individual_traces),
        atol=1e-10,
        rtol=0,
    )
    # An independent ODE reference also covers the resonant, defective case.
    reference = adc.ConditionalReadout(m, rtol=1e-12, atol=1e-14)
    expected = np.stack(
        [
            np.stack([reference.response(p, g).traces for g in reads])
            for p, reads in zip(pulses, grids, strict=True)
        ]
    )
    np.testing.assert_allclose(result.traces, expected, atol=1e-10, rtol=0)


def test_reused_grid_frequency_batch_matches_independent_trajectories():
    m = model()
    m = replace(
        m,
        receiver=replace(m.receiver, time_of_flight_ns=1.7),
        mismatch_phase_rad=np.array([0.2, -0.1]),
    )
    times = np.array([-4.0, 3.0, 5.1, 12.0, 23.0, 28.0, 43.0])
    frequencies = np.linspace(5.988, 6.012, 16)
    pulses = tuple(
        (adc.ReadoutPulse(3.0, 23.0, 1.2 + 0.3j, f, phase_rad=0.2),)
        for f in frequencies
    )
    grids = tuple((adc.SampleGrid(times, f, 0.1),) for f in frequencies)
    inputs = adc.ReadoutBatch(
        m, pulses, grids, np.arange(len(frequencies), dtype=np.int64), True
    )
    result = adc.ConditionalReadout(m).responses(inputs)
    # Independent ODE trajectories retain delay, mismatch and ringdown at each
    # carrier, while the batch shares one physical sampling schedule.
    reference = adc.ConditionalReadout(m, rtol=1e-12, atol=1e-14)
    expected = np.stack(
        [reference.response(p, g[0]).traces for p, g in zip(pulses, grids, strict=True)]
    )
    assert result.traces is not None
    np.testing.assert_allclose(result.traces[:, 0], expected, atol=1e-9, rtol=0)
    np.testing.assert_allclose(
        result.sums[:, 0], expected.sum(axis=-2), atol=1e-9, rtol=0
    )


def test_response_cache_reuses_content_across_batches_and_trace_requests():
    evaluations = []

    def shaped(local):
        evaluations.append(local)
        return 1.0 + 0.2 * np.sin(local)

    m = model()
    inputs = batch(m, pulse=adc.ReadoutPulse(3.0, 23.0, 1.0, 6.0, envelope=shaped))
    reader = adc.ConditionalReadout(m)
    sums = reader.responses(replace(inputs, include_traces=False))
    assert sums.traces is None
    count = len(evaluations)
    assert count > 0
    # Equal numeric grids/models and duplicated contents do not re-evaluate physics.
    copied = replace(
        inputs,
        model=replace(m),
        pulses=inputs.pulses * 2,
        grids=tuple(tuple(replace(g) for g in inputs.grids[0]) for _ in range(2)),
    )
    traces = reader.responses(copied)
    assert len(evaluations) == count
    np.testing.assert_array_equal(traces.sums[0], sums.sums[0])
    np.testing.assert_array_equal(traces.sums[1], sums.sums[0])
    assert traces.traces is not None
    np.testing.assert_array_equal(traces.traces.sum(axis=-2), traces.sums)


@pytest.mark.parametrize("change", ["model", "pulse", "grid", "demodulation"])
def test_response_cache_distinguishes_changed_physical_content(change):
    m = model()
    inputs = batch(m)
    reader = adc.ConditionalReadout(m)
    before = reader.responses(inputs)
    if change == "model":
        inputs = replace(
            inputs, model=replace(m, frequencies_ghz=m.frequencies_ghz + 0.007)
        )
    elif change == "pulse":
        inputs = replace(
            inputs, pulses=((replace(inputs.pulses[0][0], amplitude=2.0),),)
        )
    else:
        grid = inputs.grids[0][0]
        grid = (
            replace(grid, times_ns=grid.times_ns + 1.0)
            if change == "grid"
            else replace(grid, phase_rad=0.9)
        )
        inputs = replace(inputs, grids=((grid, inputs.grids[0][1]),))
    after = reader.responses(inputs)
    fresh = adc.ConditionalReadout(inputs.model).responses(inputs)
    assert np.max(np.abs(before.sums - after.sums)) > 1e-3
    np.testing.assert_array_equal(after.sums, fresh.sums)


def test_shaped_response_preserves_envelope_and_solver_failures():
    m = model()
    inputs = batch(
        m, pulse=adc.ReadoutPulse(3.0, 23.0, 1.0, 6.0, envelope=lambda t: np.nan)
    )
    with pytest.raises(ValueError, match="envelope.*finite"):
        adc.ConditionalReadout(m).responses(inputs)


def test_response_rejects_content_dependent_sample_counts_and_ragged_traces():
    m = model()
    inputs = batch(m)
    shorter = replace(inputs.grids[0][0], times_ns=np.array([0.0, 1.0]))
    inconsistent = replace(
        inputs, pulses=inputs.pulses * 2, grids=(inputs.grids[0], (shorter, shorter))
    )
    reader = adc.ConditionalReadout(m)
    with pytest.raises(ValueError, match="sample|grid"):
        reader.responses(inconsistent)
    ragged = replace(inputs, grids=((shorter, inputs.grids[0][1]),))
    with pytest.raises(ValueError, match="sample|grid"):
        reader.responses(ragged)
    sums_only = reader.responses(replace(ragged, include_traces=False))
    np.testing.assert_array_equal(sums_only.sample_counts, [2, 7])


@pytest.mark.parametrize("grid_kind", ["uniform", "one-sample", "alias", "late-origin"])
def test_integrated_sums_preserve_calibrated_discrete_field_sum(grid_kind):
    """The trace route is an independent oracle for discrete, not time, sums."""
    m = model()
    m = replace(
        m,
        receiver=replace(m.receiver, time_of_flight_ns=1.7),
        mismatch_phase_rad=np.array([0.2, -0.1]),
    )
    pulse = adc.ReadoutPulse(3.0, 23.0, 1.2 + 0.3j, 6.008, phase_rad=0.73)
    grid = adc.SampleGrid(np.arange(-5.0, 60.0, 0.5), 5.996, 0.3)
    if grid_kind == "one-sample":
        grid = replace(grid, times_ns=np.array([12.0]))
    elif grid_kind == "alias":
        grid = replace(grid, demodulation_frequency_ghz=8.008)
    elif grid_kind == "late-origin":
        # Decimal steps at late origins have nonidentical float64 differences.
        pulse = replace(pulse, start_ns=100003.0, end_ns=100023.0)
        grid = replace(grid, times_ns=99995.0 + np.arange(650) * 0.1)
    inputs = adc.ReadoutBatch(m, ((pulse,),), ((grid,),), np.array([0], dtype=np.int64))
    reader = adc.ConditionalReadout(m)
    result = reader.responses(inputs)
    field = adc.ConditionalReadout(m).responses(replace(inputs, include_traces=True))
    assert result.traces is None
    assert field.traces is not None
    actual = result.sums[..., 0] + 1j * result.sums[..., 1]
    expected = field.sums[..., 0] + 1j * field.sums[..., 1]
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=0)
    # Equal content is stable across repeated contents, batches and trace requests.
    copied = replace(
        inputs, model=replace(m), pulses=inputs.pulses * 2, grids=inputs.grids * 2
    )
    repeated = reader.responses(copied)
    np.testing.assert_array_equal(repeated.sums[0], result.sums[0])
    np.testing.assert_array_equal(repeated.sums[1], result.sums[0])
    traced = reader.responses(replace(inputs, include_traces=True))
    np.testing.assert_array_equal(traced.traces, field.traces)


@pytest.mark.parametrize("unsupported", ["nonuniform", "shaped", "defective"])
def test_integrated_fallback_preserves_field_values(unsupported):
    m = model()
    pulse = adc.ReadoutPulse(3.0, 23.0, 1.2 + 0.3j, 6.008)
    grid = adc.SampleGrid(np.arange(-5.0, 60.0, 0.5), 5.996, 0.3)
    if unsupported == "nonuniform":
        grid = replace(grid, times_ns=np.array([-5.0, 2.0, 5.1, 23.0, 43.0]))
    elif unsupported == "shaped":
        pulse = replace(pulse, envelope=lambda t: 1.0 + 0.1 * np.sin(t))
    else:
        m = replace(
            m,
            rates_per_ns=np.array(
                [[0.0, 0.03, 0.0], [0.0, 0.0, 0.03], [0.0, 0.0, 0.0]]
            ),
        )
    inputs = adc.ReadoutBatch(m, ((pulse,),), ((grid,),), np.array([0], dtype=np.int64))
    result = adc.ConditionalReadout(m).responses(inputs)
    decimated = adc.ConditionalReadout(m).responses(
        replace(inputs, include_traces=True)
    )
    np.testing.assert_array_equal(result.sums, decimated.sums)
    assert decimated.traces is not None
    np.testing.assert_array_equal(decimated.sums, decimated.traces.sum(axis=-2))


def test_mixed_integrated_contents_preserve_each_physical_response():
    m = model()
    pulse = adc.ReadoutPulse(3.0, 23.0, 1.2 + 0.3j, 6.008)
    shaped = replace(pulse, envelope=lambda t: 1.0 + 0.1 * np.sin(t))
    uniform = adc.SampleGrid(np.arange(-5.0, 60.0, 0.5), 5.996, 0.3)
    nonuniform = replace(
        uniform, times_ns=uniform.times_ns + 0.01 * np.sin(uniform.times_ns)
    )
    inputs = adc.ReadoutBatch(
        m,
        ((pulse,), (shaped,), (pulse,)),
        ((uniform,), (uniform,), (nonuniform,)),
        np.array([2, 0, 1, 0], dtype=np.int64),
    )
    result = adc.ConditionalReadout(m).responses(inputs)
    field = adc.ConditionalReadout(m).responses(replace(inputs, include_traces=True))
    np.testing.assert_allclose(result.sums[0], field.sums[0], rtol=1e-10, atol=0)
    np.testing.assert_array_equal(result.sums[1:], field.sums[1:])


@pytest.mark.parametrize("tones", [2, 4])
@pytest.mark.parametrize("guard", ["direct", "defective", "singular", "nonuniform"])
def test_flat_mux_integrates_carriers_without_ode_and_keeps_guard_fallback(
    tones, guard, monkeypatch
):
    from zcu_tools.simulate.adc import response

    m = model()
    m = replace(
        m,
        receiver=replace(
            m.receiver,
            time_of_flight_ns=1.7,
            noise_covariance=np.array([[1.0, 0.2], [0.2, 1.0]]),
        ),
        mismatch_phase_rad=np.array([0.2, -0.1]),
    )
    if guard == "defective":
        m = replace(
            m,
            rates_per_ns=np.array(
                [[0.0, 0.03, 0.0], [0.0, 0.0, 0.03], [0.0, 0.0, 0.0]]
            ),
        )
    elif guard == "singular":
        # The first carrier coincides with an undamped dark mode. It must use
        # full block expm, while detuned carriers can still use direct sums.
        m = replace(
            m,
            frequencies_ghz=np.full((3, 2), 6.0),
            kappa_internal_per_ns=np.zeros(2),
            kappa_external_per_ns=np.full(2, 0.08),
        )
    pulses = tuple(
        adc.ReadoutPulse(
            -2.3,
            17.7 - i,
            (1.2 + 0.3j) / (i + 1),
            6.0 + 0.018 * i,
            phase_rad=0.2 - 0.3 * i,
        )
        for i in range(tones)
    )
    times = np.arange(-5.0, 40.0, 0.5)
    if guard == "nonuniform":
        times = times + 0.01 * np.sin(times)
    grids = (
        adc.SampleGrid(times, 6.005, 0.31),
        adc.SampleGrid(times[::2] + 0.13, 5.996, -0.23),
    )
    inputs = adc.ReadoutBatch(m, (pulses,), (grids,), np.array([0], dtype=np.int64))
    oracle = adc.ConditionalReadout(m, rtol=1e-12, atol=1e-14)
    expected = np.stack([oracle.response(pulses, g).traces.sum(axis=-2) for g in grids])
    ode_calls, exponential_sizes = [], []
    solve_ivp, expm = response.solve_ivp, scipy.linalg.expm

    def observe_ode(*args, **kwargs):
        ode_calls.append(1)
        return solve_ivp(*args, **kwargs)

    def observe_expm(matrix):
        exponential_sizes.append(len(matrix))
        return expm(matrix)

    monkeypatch.setattr(response, "solve_ivp", observe_ode)
    monkeypatch.setattr(scipy.linalg, "expm", observe_expm)
    reader = adc.ConditionalReadout(m)
    result = reader.responses(inputs)
    assert not ode_calls, (
        "flat common-start mux must integrate each carrier without ODE"
    )
    if guard == "direct":
        assert not exponential_sizes, (
            "uniform well-conditioned mux must use discrete sums"
        )
    else:
        assert exponential_sizes, "rejected sums must retain constant-field propagation"
    if guard == "singular":
        assert 3 * len(m.state_labels) in exponential_sizes, (
            "singular coupling needs full block expm"
        )
    counts = np.array([len(g.times_ns) for g in grids])
    noise_sd = np.sqrt(counts[:, None, None] * np.diag(m.receiver.noise_covariance))
    assert np.max(abs(result.sums[0] - expected) / noise_sd) < 1e-7
    np.testing.assert_allclose(result.sums[0], expected, atol=2e-8, rtol=0)
    np.testing.assert_array_equal(result.sample_counts, counts)
    assert result.traces is None
    repeated = reader.responses(
        replace(inputs, pulses=(pulses, pulses), grids=(grids, grids))
    )
    np.testing.assert_array_equal(repeated.sums, np.repeat(result.sums, 2, axis=0))
    # A prior integrated cache must not turn shaped/staggered contents into
    # flat common-start fields.
    for changed in (
        (replace(pulses[0], envelope=lambda t: 1 + 0.1 * np.sin(t)), *pulses[1:]),
        (replace(pulses[0], start_ns=-3.0), *pulses[1:]),
    ):
        one = replace(
            inputs, pulses=(changed,), grids=((grids[0],),), include_traces=True
        )
        before = len(ode_calls)
        traced = reader.responses(one)
        assert len(ode_calls) > before
        reference = oracle.response(changed, grids[0]).traces.sum(axis=-2)[None]
        np.testing.assert_allclose(traced.sums[0], reference, atol=2e-8, rtol=0)


@pytest.mark.parametrize("tones", [2, 4])
@pytest.mark.parametrize("guard", ["regular", "defective", "singular"])
def test_flat_mux_traces_superpose_carriers_without_ode(tones, guard, monkeypatch):
    from zcu_tools.simulate.adc import response

    m = model()
    m = replace(
        m,
        receiver=replace(m.receiver, time_of_flight_ns=1.7),
        mismatch_phase_rad=np.array([0.2, -0.1]),
    )
    if guard == "defective":
        m = replace(
            m,
            rates_per_ns=np.array(
                [[0.0, 0.03, 0.0], [0.0, 0.0, 0.03], [0.0, 0.0, 0.0]]
            ),
        )
    elif guard == "singular":
        m = replace(
            m,
            frequencies_ghz=np.full((3, 2), 6.0),
            kappa_internal_per_ns=np.zeros(2),
            kappa_external_per_ns=np.full(2, 0.08),
        )
    pulses = tuple(
        adc.ReadoutPulse(
            -2.3,
            17.7 - i,
            (1.2 + 0.3j) / (i + 1),
            6.0 + 0.018 * i,
            phase_rad=0.2 - 0.3 * i,
        )
        for i in range(tones)
    )
    # Nonuniform grids include pre-onset, pulse corners and unequal ringdowns.
    times = np.array([-5.0, -0.6, 0.0, 3.7, 11.2, 16.4, 17.4, 18.4, 19.4, 35.0])
    grids = (
        adc.SampleGrid(times, 6.005, 0.31),
        adc.SampleGrid(times + 0.13, 5.996, -0.23),
    )
    inputs = adc.ReadoutBatch(
        m, (pulses,), (grids,), np.array([0, 0], dtype=np.int64), True
    )
    oracle = adc.ConditionalReadout(m, rtol=1e-12, atol=1e-14)
    expected = np.stack([oracle.response(pulses, g).traces for g in grids])
    ode_calls = []
    solve_ivp = response.solve_ivp

    def observe_ode(*args, **kwargs):
        ode_calls.append(1)
        return solve_ivp(*args, **kwargs)

    monkeypatch.setattr(response, "solve_ivp", observe_ode)
    reader = adc.ConditionalReadout(m)
    # Integrated preparation must not suppress subsequent dense trace requests.
    reader.responses(replace(inputs, include_traces=False))
    result = reader.responses(inputs)
    assert not ode_calls, "flat common-start mux traces must not solve a field ODE"
    assert result.traces is not None
    np.testing.assert_allclose(result.traces[0], expected, rtol=1e-6, atol=1e-9)
    np.testing.assert_array_equal(result.sums, result.traces.sum(axis=-2))
    np.testing.assert_allclose(
        result.traces[0, :, :, 0],
        np.broadcast_to(m.receiver.offset, (len(grids), len(m.state_labels), 2)),
    )
    repeated = reader.responses(inputs)
    np.testing.assert_array_equal(repeated.traces, result.traces)
    np.testing.assert_array_equal(repeated.sums, result.sums)


@pytest.mark.parametrize("include_traces", [False, True])
def test_multi_carrier_batch_matches_joint_field_and_adds_offset_once(include_traces):
    m = model()
    pulses = (
        adc.ReadoutPulse(3.0, 23.0, 1.2 + 0.3j, 6.0, phase_rad=0.2),
        adc.ReadoutPulse(3.0, 19.0, -0.7 + 0.1j, 6.018, phase_rad=-0.4),
    )
    grid = adc.SampleGrid(np.arange(-2.0, 31.0, 2.0), 6.005, 0.31)
    try:
        inputs = adc.ReadoutBatch(
            m, (pulses,), ((grid,),), np.array([0], dtype=np.int64), include_traces
        )
    except ValueError as error:
        pytest.fail(f"batched readout must accept coherent incident tones: {error}")
    reference = adc.ConditionalReadout(m, rtol=1e-12, atol=1e-14)
    expected = reference.response(pulses, grid).traces
    # Independent linearity oracle, subtracting the offset counted twice.
    separate = sum(reference.response((pulse,), grid).traces for pulse in pulses)
    np.testing.assert_allclose(expected, separate - m.receiver.offset, atol=2e-8)
    actual = adc.ConditionalReadout(m, rtol=1e-12, atol=1e-14).responses(inputs)
    np.testing.assert_allclose(actual.sums[0, 0], expected.sum(axis=-2), atol=2e-8)
    if include_traces:
        assert actual.traces is not None
        np.testing.assert_allclose(actual.traces[0, 0], expected, atol=2e-8)


@pytest.mark.parametrize("tones", [1, 2])
@pytest.mark.parametrize("include_traces", [False, True])
def test_sampled_envelope_propagates_without_field_ode(
    tones, include_traces, monkeypatch
):
    from zcu_tools.simulate.adc import response

    assert hasattr(adc, "PiecewiseLinearEnvelope"), (
        "ADC needs content-defined sampled envelopes"
    )
    m = model()
    m = replace(
        m,
        receiver=replace(m.receiver, time_of_flight_ns=1.7),
        mismatch_phase_rad=np.array([0.2, -0.1]),
    )
    knots = np.array([0.0, 1.3, 4.7, 9.0, 16.0])
    values = np.array([0.1, 0.8 + 0.1j, 1.1 - 0.2j, 0.7, 0.05], dtype=complex)
    envelope = adc.PiecewiseLinearEnvelope(knots, values)
    pulses = tuple(
        adc.ReadoutPulse(
            -2.3,
            17.7 - i,
            (1.2 + 0.3j) / (i + 1),
            6.0 + 0.018 * i,
            phase_rad=0.23 - 0.4 * i,
            envelope=envelope,
            breakpoints_ns=tuple(-2.3 + knots[1:]),
        )
        for i in range(tones)
    )
    times = np.array([-5.0, -0.6, 0.0, 3.7, 11.2, 16.4, 17.4, 18.4, 19.4, 35.0])
    grids = (
        adc.SampleGrid(times, 6.005, 0.31),
        adc.SampleGrid(times + 0.13, 5.996, -0.23),
    )
    inputs = adc.ReadoutBatch(
        m, (pulses,), (grids,), np.array([0], dtype=np.int64), include_traces
    )
    oracle = adc.ConditionalReadout(m, rtol=1e-12, atol=1e-14)
    expected = np.stack([oracle.response(pulses, g).traces for g in grids])
    calls = []
    solve = response.solve_ivp

    def observe(*args, **kwargs):
        calls.append(1)
        return solve(*args, **kwargs)

    monkeypatch.setattr(response, "solve_ivp", observe)
    reader = adc.ConditionalReadout(m)
    result = reader.responses(inputs)
    assert not calls, "known sampled carriers must not solve a field ODE"
    np.testing.assert_allclose(
        result.sums[0], expected.sum(axis=-2), rtol=1e-6, atol=2e-9
    )
    if include_traces:
        assert result.traces is not None
        np.testing.assert_allclose(result.traces[0], expected, rtol=1e-6, atol=2e-9)
    # Numeric copies are the same content, unlike opaque callbacks.
    copied = adc.PiecewiseLinearEnvelope(knots.copy(), values.copy())
    repeated = reader.responses(
        replace(inputs, pulses=(tuple(replace(p, envelope=copied) for p in pulses),))
    )
    np.testing.assert_array_equal(result.sums, repeated.sums)
    opaque = tuple(replace(p, envelope=lambda t: envelope(t)) for p in pulses)
    fallback = reader.responses(replace(inputs, pulses=(opaque,)))
    assert calls, "opaque callbacks must retain the general ODE"
    np.testing.assert_allclose(fallback.sums, result.sums, rtol=1e-6, atol=2e-9)


def test_flat_integrated_contents_use_a_carrier_axis_and_preserve_fallback(monkeypatch):
    m = model()
    frequencies = np.linspace(5.98, 6.02, 70)
    pulses = tuple(
        (adc.ReadoutPulse(-2.3, 17.7, 1.2 + 0.3j, f, phase_rad=0.2),)
        for f in frequencies
    )
    times = np.arange(-5.0, 40.0, 0.5)
    grids = tuple((adc.SampleGrid(times, f + 0.002, 0.31),) for f in frequencies)
    # A nonuniform content cannot enter the spectral tile.
    grids = (
        *grids[:-1],
        (replace(grids[-1][0], times_ns=times + 0.01 * np.sin(times)),),
    )
    inputs = adc.ReadoutBatch(m, pulses, grids, np.arange(70, dtype=np.int64))
    individual = [
        adc.ConditionalReadout(m)
        .responses(adc.ReadoutBatch(m, (p,), (g,), np.array([0], dtype=np.int64)))
        .sums[0]
        for p, g in zip(pulses, grids, strict=True)
    ]
    contractions = []
    einsum = np.einsum

    def observe(subscripts, *operands, **kwargs):
        if any(
            np.ndim(a) == 3
            and np.shape(a)[0] > 1
            and np.shape(a)[1:] == (2 * len(m.state_labels), len(m.state_labels))
            for a in operands
        ):
            contractions.append(True)
        return einsum(subscripts, *operands, **kwargs)

    monkeypatch.setattr(np, "einsum", observe)
    result = adc.ConditionalReadout(m).responses(inputs)
    assert contractions, (
        "multiple contents must share a contraction, not loop over sums"
    )
    np.testing.assert_allclose(result.sums, individual, rtol=1e-10, atol=1e-9)
