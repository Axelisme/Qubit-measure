"""Compiled R-pi programs produce additive summaries of independent rep cycles."""

from threading import Event

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.analysis.readout_chain import RPiSums, rpi_analysis
from zcu_tools.program import v2
from zcu_tools.program.acquisition import StoppedPartialAcquireError
from zcu_tools.program.v2.modules import ConstWaveformCfg

from tests.program._acquisition_soc import AcquisitionSoc
from tests.program._profiles import load_soccfg
from tests.program._shot_reduction import reduce_round


def rpi_builder(*, reps: int = 3, points: int = 2, reps_innermost: bool = True):
    builder = v2.TimelineBuilder(
        load_soccfg(n_gens=2),
        reps=reps,
        reps_innermost=reps_innermost,
        relax_delay=0.0,
    )
    builder.add_loop("setting", points)
    return builder


def emit_chain(builder: v2.TimelineBuilder, reads: int = 16) -> None:
    readout = v2.PulseReadout(
        "readout",
        v2.PulseReadout.Cfg(
            gen_ch=0,
            ro_ch=0,
            nqz=1,
            freq=100.0,
            gain=0.1,
            waveform=ConstWaveformCfg(length=0.2),
            ro_length=0.1,
            post_delay=0.1,
        ),
    )
    pi = v2.Pulse(
        "pi",
        v2.Pulse.Cfg(
            ch=1,
            nqz=1,
            freq=200.0,
            gain=0.2,
            waveform=ConstWaveformCfg(length=0.02),
        ),
    )
    with builder.repeat("rpi", reads):
        builder.emit(readout, tag="sample")
        builder.emit(pi)


def rpi_program(
    *, reps: int = 3, points: int = 2, reps_innermost: bool = True, reads: int = 16
):
    builder = rpi_builder(reps=reps, points=points, reps_innermost=reps_innermost)
    emit_chain(builder, reads=reads)
    return builder.build()


def pair_oracle(segments: list[NDArray[np.float64]]) -> NDArray[np.float64]:
    """Enumerate each read and legal ordered pair without vectorized lag slices."""
    record = np.zeros(35, dtype=np.float64)
    for segment in segments:
        record[0] += 1
        for t, (i, q) in enumerate(segment):
            record[1] += 1
            record[2] += i
            record[3] += q
            record[4] += i * i + q * q
            for later in range(t + 1, min(t + 16, len(segment))):
                lag = later - t
                record[4 + lag] += i * segment[later, 0] + q * segment[later, 1]
                record[19 + lag] += 1
    return record


@pytest.mark.parametrize("reads", [16, 64])
def test_rep_cycles_match_each_pair_without_mutation(reads: int):
    iq = np.random.default_rng(2407).normal(size=(2, 3, reads, 2))
    # Distinct rep offsets make illegal cross-rep products observable in every lag.
    iq += np.array([10.0, -30.0, 70.0])[None, :, None, None]
    original = iq.copy()
    process = v2.rpi_sums(rpi_program(reads=reads), "sample")
    actual = reduce_round(process, {0: iq})
    expected = np.stack([pair_oracle(list(point)) for point in iq])
    assert actual.shape == (2, 35)
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual[:, 5:20], expected[:, 5:20], rtol=1e-12)
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-12)
    np.testing.assert_array_equal(actual[:, 0], [3, 3])
    np.testing.assert_array_equal(actual[:, 1], [3 * reads, 3 * reads])
    for point in actual:
        np.testing.assert_array_equal(point[20:35], 3 * (reads - np.arange(1, 16)))
    np.testing.assert_array_equal(iq, original)


def test_producer_summary_can_analyze_seeded_alternating_iq():
    rng = np.random.default_rng(507)
    length = 4096 * 16
    signs = np.cumprod(np.where(rng.random(length) < 0.875, -1.0, 1.0))
    iq = rng.normal(size=(length, 2))
    iq[:, 0] += 3 * signs
    record = reduce_round(
        v2.rpi_sums(rpi_program(reps=4096, points=1), "sample"),
        {0: iq.reshape(1, 4096, 16, 2)},
    )
    result = rpi_analysis(RPiSums.from_processor(record[0]))
    assert result.valid, result.invalid_reasons
    assert result.gate_passed
    assert result.score is not None


def test_pooled_rounds_decode_to_saved_summary_and_public_analysis():
    rng = np.random.default_rng(938)
    rounds = [rng.normal(size=(2, 3, 16, 2)) for _ in range(2)]
    process = v2.rpi_sums(rpi_program(), "sample")
    pooled = reduce_round(process, {0: rounds[0]}) + reduce_round(
        process, {0: rounds[1]}
    )
    original = pooled.copy()
    for point in range(2):
        expected = pair_oracle([rep for iq in rounds for rep in iq[point]])
        decoded = RPiSums.from_processor(pooled[point])
        assert decoded.read_count == 96
        np.testing.assert_array_equal(decoded.segment_lengths, [16])
        np.testing.assert_array_equal(decoded.segment_counts, [6])
        np.testing.assert_allclose(decoded.first_moment_sums, expected[2:4])
        assert decoded.same_read_square_sum == pytest.approx(expected[4])
        np.testing.assert_allclose(decoded.lag_product_sums, expected[5:20])
        np.testing.assert_array_equal(decoded.pair_counts, expected[20:35])
        reference = RPiSums(
            read_count=96,
            first_moment_sums=expected[2:4],
            same_read_square_sum=float(expected[4]),
            lag_product_sums=expected[5:20],
            pair_counts=expected[20:35].astype(np.int64),
            segment_lengths=np.array([16], dtype=np.int64),
            segment_counts=np.array([6], dtype=np.int64),
        )
        actual_analysis = rpi_analysis(decoded)
        expected_analysis = rpi_analysis(reference)
        assert actual_analysis.contrast == pytest.approx(expected_analysis.contrast)
        assert actual_analysis.null_variance == pytest.approx(
            expected_analysis.null_variance
        )
        assert actual_analysis.invalid_reasons == expected_analysis.invalid_reasons
    np.testing.assert_array_equal(pooled, original)
    # The decoder owns detached arrays, so later edits to a saved record are safe.
    detached = RPiSums.from_processor(pooled[0])
    pooled[:] = 0
    np.testing.assert_array_equal(detached.pair_counts, np.arange(90, 5, -6))


@pytest.mark.parametrize("points,reps_innermost", [(2, False), (2, True), (1, False)])
def test_compiled_loop_order_requires_innermost_reps_only_for_multiple_settings(
    points: int, reps_innermost: bool
):
    prog = rpi_program(points=points, reps_innermost=reps_innermost)
    if points > 1 and not reps_innermost:
        with pytest.raises(ValueError, match="reps_innermost=True"):
            v2.rpi_sums(prog, "sample")
    else:
        result = reduce_round(
            v2.rpi_sums(prog, "sample"), {0: np.ones((points, 3, 16, 2))}
        )
        np.testing.assert_array_equal(result[:, 1], np.full(points, 48))
        np.testing.assert_array_equal(result[:, 20], np.full(points, 45))


@pytest.mark.parametrize("mismatch", ["short", "extra_trigger", "nested"])
def test_compiled_readout_provenance_requires_one_fixed_complete_chain(mismatch: str):
    builder = rpi_builder()
    if mismatch == "nested":
        with builder.repeat("outer", 2):
            emit_chain(builder)
    else:
        emit_chain(builder, reads=15 if mismatch == "short" else 16)
    if mismatch == "extra_trigger":
        builder.trigger([0])
    prog = builder.build()
    with pytest.raises(ValueError, match="expected.*actual"):
        v2.rpi_sums(prog, "sample")


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((1, 3, 16, 2), np.float64),
        ((2, 2, 16, 2), np.float64),
        ((2, 3, 15, 2), np.float64),
        ((2, 3, 16, 3), np.float64),
        ((2, 48, 2), np.float64),
        ((2, 3, 16, 2), np.float32),
    ],
)
def test_processor_rejects_incoming_round_shape_or_dtype_mismatch(shape, dtype):
    process = v2.rpi_sums(rpi_program(), "sample")
    with pytest.raises(ValueError, match="ADC 0.*expected.*actual"):
        reduce_round(process, {0: np.ones(shape, dtype=dtype)})


def test_read_label_and_selected_adc_are_required_without_mutating_declarations():
    prog = rpi_program()
    with pytest.raises(ValueError, match="label 'absent'.*actual 0.*sample"):
        v2.rpi_sums(prog, "absent")
    process = v2.rpi_sums(prog, "sample")
    with pytest.raises(KeyError, match="0"):
        reduce_round(process, {1: np.zeros((2, 3, 16, 2))})
    selected = np.ones((2, 3, 16, 2))
    np.testing.assert_array_equal(
        reduce_round(process, {0: selected, 1: selected * 100}),
        reduce_round(process, {0: selected}),
    )
    assert prog.ro_chs[0]["trigs"] == 16


def expect_sums(result: object) -> NDArray[np.float64]:
    """Require the public summed acquisition result, not native ADC lists."""
    if not isinstance(result, np.ndarray) or result.dtype != np.float64:
        pytest.fail(f"Expected float64 summed ndarray, got {type(result).__name__}")
    return result


@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("stepped", [False, True])
def test_acquire_summed_transport_chunks_and_round_resets_preserve_completed_cycles(
    monkeypatch: pytest.MonkeyPatch, cancelled: bool, stepped: bool
):
    prog = rpi_program()
    soc = AcquisitionSoc(load_soccfg(n_gens=2).get_cfg())
    stop = Event()
    rng = np.random.default_rng(214)
    raw = [rng.integers(-500, 500, size=(2, 3, 16, 2)) for _ in range(2)]
    rows = [values.reshape(-1, 2) for values in raw]
    # Transport chunk cuts a setting's reps, but the complete round owns all cycles.
    batches = iter(
        [
            [(2, ([rows[0][:32]], {}))],
            [(4, ([rows[0][32:]], {}))],
            [(2 if cancelled else 6, ([rows[1][:32] if cancelled else rows[1]], {}))],
        ]
    )

    def poll():
        batch = next(batches, None)
        if batch is None:
            stop.set()
            return []
        return batch

    monkeypatch.setattr(soc, "poll_data", poll)
    hooks = []
    result = prog.acquire_summed(
        soc,
        reducer=v2.rpi_sums(prog, "sample"),
        rounds=2,
        step_rounds=stepped,
        progress=False,
        cancel_flag=stop,
        round_hook=lambda count, data, flag: hooks.append(data.copy()),
    )
    if stepped:
        assert result is None
        assert prog.finish_round()
        prog.prepare_round()
        assert prog.finish_round() is False
        result = prog.finish_acquire()
    result = expect_sums(result)
    completed = 1 if cancelled else 2
    length = prog.ro_chs[0]["length"]
    expected = np.stack(
        [
            pair_oracle(
                [rep / length for values in raw[:completed] for rep in values[point]]
            )
            for point in range(2)
        ]
    )
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-10)
    np.testing.assert_array_equal(result[:, 0], 3 * completed)
    np.testing.assert_allclose(hooks[-1], result)
    assert len(hooks) == completed
    decoded = RPiSums.from_processor(result[0])
    np.testing.assert_array_equal(decoded.segment_lengths, [16])
    np.testing.assert_array_equal(decoded.segment_counts, [3 * completed])
    rpi_analysis(decoded)


@pytest.mark.parametrize("stepped", [False, True])
def test_rpi_cancel_before_first_completed_round_has_no_saved_summary(
    monkeypatch: pytest.MonkeyPatch, stepped: bool
):
    prog = rpi_program()
    soc = AcquisitionSoc(load_soccfg(n_gens=2).get_cfg())
    stop = Event()
    batches = iter([[(1, ([np.zeros((16, 2), dtype=np.int64)], {}))]])

    def poll():
        batch = next(batches, None)
        if batch is None:
            stop.set()
            return []
        return batch

    monkeypatch.setattr(soc, "poll_data", poll)
    hooks = []

    def acquire():
        prog.acquire_summed(
            soc,
            reducer=v2.rpi_sums(prog, "sample"),
            rounds=1,
            step_rounds=stepped,
            progress=False,
            cancel_flag=stop,
            round_hook=lambda count, data, flag: hooks.append(data.copy()),
        )
        if stepped:
            assert prog.finish_round() is False
            prog.finish_acquire()

    with pytest.raises(StoppedPartialAcquireError):
        acquire()
    assert hooks == []
