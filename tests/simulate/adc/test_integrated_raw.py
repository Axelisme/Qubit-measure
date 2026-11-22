"""Deferred ADC shots: physical paths, independent means and QICK layouts."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from zcu_tools.simulate import adc
from zcu_tools.simulate.boundary import SamplingIdentity
from zcu_tools.simulate.evolution import Cancelled


def model(rates=None, *, receiver=None):
    return adc.ReadoutModel(
        "integrated-paths",
        ("g", "e", "f"),
        np.array([[6.0, 6.1], [6.04, 6.1], [5.96, 6.1]]),
        6.0,
        np.array([0.04, 0.02]),
        np.array([0.16, 0.0]),
        np.zeros((3, 3)) if rates is None else rates,
        receiver=adc.Receiver(gain=100) if receiver is None else receiver,
    )


def batch(readout_model, *, shaped=False):
    envelope = (
        adc.PiecewiseLinearEnvelope(
            np.array([0, 15, 40.0]), np.array([0, 1 + 0.2j, 0.4], complex)
        )
        if shaped
        else None
    )
    pulse = adc.ReadoutPulse(5, 65, 0.8 + 0.2j, 6.007, 0.2, envelope)
    grids = (
        adc.SampleGrid(np.arange(-5, 86, 2, dtype=float), 6.003, 0.1),
        adc.SampleGrid(np.array([-5, 5, 10, 25, 45, 65, 85.0]), 6.003, 0.1),
    )
    return adc.ReadoutBatch(
        readout_model,
        ((adc.ReadoutEvent((pulse,), grids),),),
        np.array([0], dtype=np.int64),
    )


def prepare(populations, batches, *, reps):
    # Assert the new public capability at the existing ADC package seam, so a
    # missing implementation is a behaviour failure rather than an import error.
    factory = getattr(adc, "PreparedRawShots", None)
    assert factory is not None, "ADC must prepare deferred physical raw shots"
    return factory(adc.ProjectionReadout.unprojected(populations), batches, reps=reps)


@pytest.mark.parametrize("shaped", [False, True])
@pytest.mark.parametrize("avg_level", [0, 1, 2])
def test_zero_jump_raw_preserves_receivers_windows_and_loop_layout(shaped, avg_level):
    receiver = adc.Receiver(
        gain=17, phase_rad=0.3, offset=np.array([0.2, -0.4]), time_of_flight_ns=2.5
    )
    readout_model = replace(
        model(receiver=receiver), mismatch_phase_rad=np.array([0.6, 0.0])
    )
    content = batch(readout_model, shaped=shaped)
    content = replace(content, content_index=np.zeros(6, dtype=np.int64))
    populations = np.zeros((2, 3, 3))
    populations[..., 1] = 1
    prepared = prepare(populations, {3: content}, reps=7)
    source = prepared.bind(SamplingIdentity(17, (2, 3, 1)))
    raw = source.materialize(7, avg_level)[3]
    expected = adc.ADC(readout_model).responses(content).sums[0, :, 1]
    flat = np.moveaxis(raw, avg_level, 2)
    assert raw.dtype == np.int64
    assert flat.shape == (2, 3, 7, 2, 2)
    assert_array_equal(flat, np.broadcast_to(np.rint(expected), flat.shape))
    raw[...] = -999
    assert_array_equal(
        source.materialize(7, avg_level)[3],
        np.moveaxis(np.broadcast_to(np.rint(expected), flat.shape), 2, avg_level),
    )


@pytest.mark.parametrize(
    "rates",
    [
        np.array([[0, 0.025, 0], [0, 0, 0], [0, 0, 0]]),
        np.array([[0, 0.05, 0], [0, 0, 0.05], [0, 0, 0]]),
        np.array([[0, 0.2, 0.1], [0.1, 0, 0.2], [0.2, 0.1, 0]]),
    ],
    ids=["single-jump", "cascade", "many-jumps"],
)
@pytest.mark.parametrize("shaped", [False, True])
def test_integrated_shots_form_bridge_and_match_conditional_expectation(rates, shaped):
    readout_model = model(rates)
    content = batch(readout_model, shaped=shaped)
    populations = (
        np.array([0.0, 1.0, 0.0])
        if not rates[:, 2].any()
        else np.array([0.0, 0.0, 1.0])
    )
    source = prepare(populations, {0: content}, reps=1600).bind(SamplingIdentity(52))
    raw = source.materialize(1600, 0)[0]
    expected = (
        adc.ADC(readout_model).responses(content).sums[0, :, populations.argmax()]
    )
    error = np.abs(raw.mean(axis=0) - expected)
    stderr = raw.std(axis=0) / np.sqrt(len(raw))
    assert np.all(error <= 6 * stderr + 0.51)
    # A pure initial state used to produce one conditional centre. Physical
    # paths have a bridge, including repeated transitions beyond two jumps.
    assert np.unique(raw[:, 0], axis=0).shape[0] > 150
    assert raw[:, 0].var(axis=0).sum() > 10000
    assert_array_equal(source.materialize(1600, 0)[0], raw)


@pytest.mark.parametrize("mux", [False, True])
def test_receivers_share_full_paths_and_windows_with_independent_noise(mux):
    rates = np.array([[0, 0.12, 0.04], [0.06, 0, 0.1], [0.09, 0.03, 0]])
    content = batch(model(rates), shaped=mux)
    pulse = content.events[0][0].pulses[0]
    if mux:
        content = replace(
            content,
            events=(
                (
                    replace(
                        content.events[0][0],
                        pulses=(
                            pulse,
                            replace(pulse, start_ns=15, frequency_ghz=6.021),
                        ),
                    ),
                ),
            ),
        )
    grid = content.events[0][0].grids[0]
    content = replace(
        content,
        events=((replace(content.events[0][0], grids=(grid, grid)),),),
    )
    shifted = replace(
        content,
        model=replace(
            content.model, receiver=adc.Receiver(gain=100, offset=np.array([3.0, -2.0]))
        ),
    )
    noisy = replace(
        content,
        model=replace(
            content.model,
            receiver=adc.Receiver(gain=100, noise_covariance=np.eye(2) * 4),
        ),
    )
    source = prepare(
        np.array([0.0, 0.0, 1.0]),
        {7: content, 2: shifted, 5: noisy, 9: noisy},
        reps=500,
    ).bind(SamplingIdentity(61))
    raw = source.materialize(500, 0)
    assert list(raw) == [7, 2, 5, 9]
    assert raw[7][:, 0].var(axis=0).sum() > 10000
    assert_array_equal(raw[7][:, 0], raw[7][:, 1])
    assert_array_equal(
        raw[2] - raw[7],
        np.broadcast_to(len(grid.times_ns) * np.array([3, -2]), raw[7].shape),
    )
    residual_a = (raw[5] - raw[7]).reshape(500, -1)
    residual_b = (raw[9] - raw[7]).reshape(500, -1)
    assert_allclose(residual_a.var(axis=0), len(grid.times_ns) * 4, rtol=0.22)
    assert np.max(np.abs(np.corrcoef(residual_a.T, residual_b.T)[:4, 4:])) < 0.16
    expected = adc.ADC(content.model).responses(content).sums[0, :, 2]
    assert np.all(
        np.abs(raw[7].mean(axis=0) - expected)
        < 6 * raw[7].std(axis=0) / np.sqrt(500) + 0.51
    )
    assert_array_equal(source.materialize(500, 0)[7], raw[7])


def test_mux_carriers_superpose_on_the_same_jumping_cavity():
    content = batch(model(np.array([[0, 0.1, 0.03], [0.05, 0, 0.08], [0.02, 0.04, 0]])))
    first = content.events[0][0].pulses[0]
    second = replace(first, start_ns=15, frequency_ghz=6.023, amplitude=0.3 - 0.4j)
    mux = replace(
        content,
        events=((replace(content.events[0][0], pulses=(first, second)),),),
    )
    # The silent first pulse preserves the population origin for the later tone.
    later = replace(
        content,
        events=(
            (
                replace(
                    content.events[0][0], pulses=(replace(first, amplitude=0), second)
                ),
            ),
        ),
    )
    raw = (
        prepare(np.array([0.0, 0.0, 1.0]), {0: mux, 1: content, 2: later}, reps=120)
        .bind(SamplingIdentity(4))
        .materialize(120, 0)
    )
    assert np.unique(raw[0][:, 0], axis=0).shape[0] > 100
    assert_allclose(raw[0], raw[1] + raw[2], atol=1)


def test_deferred_raw_does_not_change_statistics_stream_or_retain_noisy_results():
    content = batch(model(np.array([[0, 0.025, 0], [0, 0, 0.04], [0, 0, 0]])))
    content = replace(
        content,
        events=(
            (
                replace(
                    content.events[0][0],
                    grids=(
                        content.events[0][0].grids[0],
                        content.events[0][0].grids[0],
                    ),
                ),
            ),
        ),
    )
    conditional = adc.ADC(content.model).responses(content)
    channel = adc.ChannelReadout(
        0, conditional, np.array(0, dtype=np.int64), content.model.receiver
    )
    probabilities = np.array([0.1, 0.2, 0.7])
    sampler = adc.PreparedStatistics(
        adc.ProjectionReadout.unprojected(probabilities), (channel,), reps=13
    )
    prepared = prepare(probabilities, {0: content}, reps=13)
    arguments = dict(seed=12, acquire_sequence=2, round_index=3)
    before = sampler.draw(**arguments)
    first = prepared.bind(before.raw_identity).materialize(13, 0)[0]
    sampler.draw(seed=12, acquire_sequence=2, round_index=4)
    after = sampler.draw(**arguments)
    assert_array_equal(before.state_counts, after.state_counts)
    assert_array_equal(before.channels[0].sums, after.channels[0].sums)
    source = prepared.bind(after.raw_identity)
    assert (
        source.materialize_cancellable(13, 0, cancel=Event())[0].tobytes()
        == first.tobytes()
    )
    assert not np.array_equal(
        prepared.bind(SamplingIdentity(13)).materialize(13, 0)[0], first
    )


def test_raw_rounds_reuse_noiseless_setup_only_in_the_active_batch():
    calls = 0

    def envelope(_ns: float) -> complex:
        nonlocal calls
        calls += 1
        return 1.0 + 0j

    content = batch(model(receiver=adc.Receiver(noise_covariance=np.eye(2) * 4)))
    (event,) = content.events[0]
    (pulse,) = event.pulses
    event = replace(event, pulses=(replace(pulse, envelope=envelope),))
    content = replace(content, events=((event,),))
    chain = adc.ProjectionChain(np.array([0.1, 0.2, 0.7]), ())
    prepared = adc.PreparedRawShots(chain, {0: content}, reps=5)
    round_1 = SamplingIdentity(12, (3, 0, 1))
    round_2 = SamplingIdentity(12, (3, 1, 1))
    owner = adc.RawPlanOwner()
    assert calls == 0
    owner.bind(prepared, round_1)
    source = owner.bind(prepared, round_1)
    assert calls == 0, "binding without taking raw ran envelope setup"
    first = source.materialize(5, 0)[0]
    del source
    setup_calls = calls
    assert setup_calls > 0
    second = owner.bind(prepared, round_2).materialize(5, 0)[0]
    # With no CTMC jumps, the opaque envelope is only needed for noiseless setup.
    assert calls == setup_calls, "a later round repeated noiseless envelope setup"
    assert not np.array_equal(first, second)

    shifted = replace(event, pulses=(replace(event.pulses[0], amplitude=0.4j),))
    changed = replace(content, events=((shifted,),))
    new_prepared = adc.PreparedRawShots(chain, {0: changed}, reps=5)
    different = owner.bind(new_prepared, round_2).materialize(5, 0)[0]
    assert calls > setup_calls
    assert not np.array_equal(second, different)

    after_b = calls
    replay = owner.bind(prepared, round_1).materialize(5, 0)[0]
    assert calls > after_b, "returning to A revived the retired generation"
    assert_array_equal(replay, first)

    # The original bind is always cold, even while the same batch has a memo.
    after_replay = calls
    assert_array_equal(prepared.bind(round_2).materialize(5, 0)[0], second)
    assert calls > after_replay


def test_retiring_a_batch_during_raw_setup_does_not_publish_its_late_plan():
    watchdog = 30
    building_or_done = Event()
    proceed = Event()
    a_calls = 0
    b_calls = 0

    def first_envelope(_ns: float) -> complex:
        nonlocal a_calls
        a_calls += 1
        if a_calls == 1:
            building_or_done.set()
            # Once notified, the test retires A and releases this worker in finally.
            assert proceed.wait(timeout=watchdog), (
                "retirement blocked on the raw solver"
            )
        return 1 + 0j

    def next_envelope(_ns: float) -> complex:
        nonlocal b_calls
        b_calls += 1
        return 0.3j

    original = batch(model())
    (event,) = original.events[0]
    (pulse,) = event.pulses

    def prepared(envelope):
        content = replace(
            original,
            events=((replace(event, pulses=(replace(pulse, envelope=envelope),)),),),
        )
        return prepare(np.array([1.0, 0, 0]), {0: content}, reps=1)

    owner = adc.RawPlanOwner()
    identity = SamplingIdentity(1)
    old = owner.bind(prepared(first_envelope), identity)
    next_batch = prepared(next_envelope)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(old.materialize, 1, 0)
        # Materialization signals on envelope entry or, if it exits early, via the future.
        pending.add_done_callback(lambda _: building_or_done.set())
        try:
            assert building_or_done.wait(timeout=watchdog), (
                "raw worker made no progress"
            )
            if pending.done():
                new = None
            else:
                next_source = owner.bind(next_batch, identity)
                proceed.set()  # Let A finish while B materializes.
                new = next_source.materialize(1, 0)[0]
        finally:
            proceed.set()
            previous = pending.result(timeout=watchdog)[0]

    assert new is not None, "raw finished before entering envelope setup"

    assert not np.array_equal(new, previous)
    b_setup = b_calls
    assert b_setup > 0
    assert_array_equal(owner.bind(next_batch, identity).materialize(1, 0)[0], new)
    assert b_calls == b_setup, "retired setup disturbed the current generation"
    a_setup = a_calls
    assert_array_equal(old.materialize(1, 0)[0], previous)
    assert a_calls > a_setup, "late setup refilled the retired generation"


@pytest.mark.parametrize("stop_at", ["entry", "during-event"])
@pytest.mark.parametrize("with_owner", [False, True])
def test_raw_stop_does_not_observe_later_events(stop_at, with_owner):
    cancel = Event()

    def first_envelope(_ns):
        if stop_at == "entry":
            pytest.fail("pre-cancelled raw must not evaluate the first envelope")
        cancel.set()
        return 1.0 + 0j

    def later_envelope(_ns):
        pytest.fail("interrupted raw must not evaluate a later event's envelope")

    content = batch(model())
    (event,) = content.events[0]
    (pulse,) = event.pulses
    events = tuple(
        replace(event, pulses=(replace(pulse, envelope=envelope),))
        for envelope in (first_envelope, later_envelope)
    )
    content = replace(content, events=(events,))
    p = np.array([[1.0, 0.0, 0.0]])
    prepared = adc.PreparedRawShots(
        adc.ProjectionReadout(np.ones(1), (p, p)), {0: content}, reps=1
    )
    identity = SamplingIdentity(1)
    owner = adc.RawPlanOwner()
    source = owner.bind(prepared, identity) if with_owner else prepared.bind(identity)
    if stop_at == "entry":
        cancel.set()
    with pytest.raises(Cancelled):
        source.materialize_cancellable(1, 0, cancel=cancel)
    assert cancel.is_set()


def test_raw_stop_at_receiver_only_group_prevents_later_path_sampling():
    class StopDuringFirstGroup:
        def __init__(self):
            self.checkpoints = iter(("entry", "event"))

        def is_set(self):
            # Entry and the logical event precede point groups. Stop becomes
            # pending after entering the event, before leaving its first group.
            return next(self.checkpoints, None) is None

    # Only the second group has a physical path. Its rate makes the first
    # transition shorter than the floating-point resolution at pulse start.
    content = batch(model(np.array([[0, 0, 0], [1e24, 0, 0], [0, 0, 0]])))
    (event,) = content.events[0]
    content = replace(
        content,
        events=((replace(event, pulses=()),), (event,)),
        content_index=np.array([0, 1], dtype=np.int64),
    )
    source = prepare(np.array([[1.0, 0, 0], [1.0, 0, 0]]), {0: content}, reps=1).bind(
        SamplingIdentity(1)
    )

    with pytest.raises(ValueError, match="transition times.*resolution"):
        source.materialize(1, 0)
    with pytest.raises(Cancelled):
        source.materialize_cancellable(1, 0, cancel=StopDuringFirstGroup())


def test_raw_stop_before_empty_delivery_returns_no_arrays():
    class StopAfterEntry:
        def __init__(self):
            self.requested = False

        def is_set(self):
            requested = self.requested
            self.requested = True
            return requested

    source = prepare(np.array([1.0, 0, 0]), {0: batch(model())}, reps=0).bind(
        SamplingIdentity(3)
    )
    with pytest.raises(Cancelled):
        source.materialize_cancellable(0, 0, cancel=StopAfterEntry())


@pytest.mark.parametrize(
    "reps,avg_level", [(6, 0), (-1, 0), (7, -1), (7, 1), (True, 0)]
)
def test_raw_rejects_reps_or_layout_mismatch(reps, avg_level):
    source = prepare(np.array([1.0, 0, 0]), {0: batch(model())}, reps=7).bind(
        SamplingIdentity(3)
    )
    with pytest.raises(ValueError, match="reps|avg_level|layout"):
        source.materialize(reps, avg_level)


def test_zero_reps_is_empty_and_receiver_overflow_fails():
    content = batch(model())
    source = prepare(np.array([1.0, 0, 0]), {0: content}, reps=0).bind(
        SamplingIdentity(3)
    )
    assert source.materialize(0, 0)[0].shape == (0, 2, 2)
    overflow = replace(
        content,
        model=replace(
            content.model, receiver=adc.Receiver(offset=np.array([2.0**63, 0]))
        ),
    )
    source = prepare(np.array([1.0, 0, 0]), {0: overflow}, reps=1).bind(
        SamplingIdentity(3)
    )
    with pytest.raises(ValueError, match="int64"):
        source.materialize(1, 0)


def test_state_independent_coalescent_cavities_keep_exact_sum_despite_many_jumps():
    detuning = 0.04 / (2 * np.pi)
    readout_model = replace(
        model(np.array([[0, 0.2, 0.1], [0.1, 0, 0.2], [0.2, 0.1, 0]])),
        frequencies_ghz=np.tile([6 - detuning, 6 + detuning], (3, 1)),
        kappa_internal_per_ns=np.array([0.02, 0.02]),
        kappa_external_per_ns=np.array([0.08, 0.08]),
    )
    content = batch(readout_model)
    raw = (
        prepare(np.array([0.2, 0.3, 0.5]), {0: content}, reps=30)
        .bind(SamplingIdentity(2))
        .materialize(30, 0)[0]
    )
    expected = (
        adc.ADC(replace(readout_model, rates_per_ns=np.zeros((3, 3))))
        .responses(content)
        .sums[0, :, 0]
    )
    assert_allclose(raw, np.broadcast_to(np.rint(expected), raw.shape), atol=0)
