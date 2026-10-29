"""Public native IQ acquisition preserves chunks, projection and cancellation."""

import logging
from threading import Event

import numpy as np
import pytest
from zcu_tools.cfg_model import LinearClassifier
from zcu_tools.program.acquisition import (
    AcquisitionContractError,
    StoppedPartialAcquireError,
)
from zcu_tools.program.herald import HeraldSpec

from tests.program._acquisition_soc import make_chunk_acquisition


def test_decimated_rejects_herald_before_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    soc, program = make_chunk_acquisition()
    spec = HeraldSpec(0, 0, LinearClassifier(angle=0.0, threshold=0.0), 2)

    def unexpected_poll():
        pytest.fail("invalid herald acquisition reached transport")

    monkeypatch.setattr(soc, "poll_data", unexpected_poll)
    with pytest.raises(ValueError, match="decimated.*herald"):
        program.acquire_decimated(soc, herald=[spec], progress=False)


@pytest.mark.parametrize("stepped", [False, True])
def test_acquire_preserves_chunk_offsets_and_read_order(
    monkeypatch: pytest.MonkeyPatch, stepped: bool
) -> None:
    soc, program = make_chunk_acquisition()
    raw = np.array(
        [
            [10, -10],
            [20, -20],
            [30, -30],
            [40, -40],
            [50, -50],
            [60, -60],
            [70, -70],
            [80, -80],
        ],
        dtype=np.int64,
    )
    # Multiple chunks in one poll, followed by another poll at a nonzero offset.
    batches = iter(
        [
            [(1, ([raw[:2]], {})), (2, ([raw[2:6]], {}))],
            [(1, ([raw[6:]], {}))],
        ]
    )
    monkeypatch.setattr(soc, "poll_data", lambda: next(batches))
    if stepped:
        program.acquire(soc, progress=False, step_rounds=True)
        assert program.finish_round() is False
        result = program.finish_acquire()
    else:
        result = program.acquire(soc, progress=False)

    captured = program.get_raw()
    assert captured is not None
    np.testing.assert_array_equal(captured[0], raw.reshape(4, 2, 2))
    assert result is not None
    np.testing.assert_allclose(
        result[0], np.array([[40, -40], [50, -50]]) / program.ro_chs[0]["length"]
    )


@pytest.mark.parametrize("option_list", [False, True])
def test_ordinary_acquire_preserves_native_projection_sign_offset_and_equality(
    monkeypatch: pytest.MonkeyPatch, option_list: bool
) -> None:
    soc, program = make_chunk_acquisition(reps=3, reads=1)
    program.soccfg["readouts"][0]["iq_offset"] = 1.0
    raw = (
        np.array([[0, 0], [0, 1], [0, 2]], dtype=np.int64) * program.ro_chs[0]["length"]
    )
    monkeypatch.setattr(soc, "poll_data", lambda: [(3, ([raw], {}))])
    result = program.acquire(
        soc,
        threshold=[0.0] if option_list else 0.0,
        angle=[np.pi / 2] if option_list else np.pi / 2,
        remove_offset=True,
        progress=False,
    )
    assert result is not None
    np.testing.assert_allclose(result[0], [[1 / 3, 0]])


def test_acquire_logs_size_mismatch_but_keeps_numpy_broadcasting(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    soc, program = make_chunk_acquisition(reps=3)
    batches = iter(
        [
            [(2, ([np.array([[11, -11]], dtype=np.int64)], {}))],
            [(1, ([np.array([[21, -21], [22, -22]], dtype=np.int64)], {}))],
        ]
    )
    monkeypatch.setattr(soc, "poll_data", lambda: next(batches))
    with caplog.at_level(logging.ERROR, logger="qick.qick_asm"):
        result = program.acquire(soc, progress=False)
    assert caplog.messages == [
        "data size mismatch: new_points=2, nreads=2, data shape (1, 2)"
    ]
    captured = program.get_raw()
    assert captured is not None
    np.testing.assert_array_equal(
        captured[0],
        [[[11, -11], [11, -11]], [[11, -11], [11, -11]], [[21, -21], [22, -22]]],
    )
    assert result is not None
    np.testing.assert_allclose(
        result[0], np.array([[43, -43], [44, -44]]) / (3 * program.ro_chs[0]["length"])
    )


@pytest.mark.parametrize("stepped", [False, True])
@pytest.mark.parametrize("rows", [5, 6])
def test_acquire_rejects_shot_overrun_before_writing_chunk(
    monkeypatch: pytest.MonkeyPatch, stepped: bool, rows: int
) -> None:
    soc, program = make_chunk_acquisition(reps=3)
    first = np.array([[11, -11], [12, -12]], dtype=np.int64)
    batches = iter(
        [
            [(1, ([first], {}))],
            [(3, ([np.full((rows, 2), 17, dtype=np.int64)], {}))],
        ]
    )
    monkeypatch.setattr(soc, "poll_data", lambda: next(batches))
    completed_rounds: list[int] = []

    def record_round(count: int, _data: object, _stop: object) -> None:
        completed_rounds.append(count)

    message = (
        r"expected_shots=3, received_shots=1, chunk_shots=3, "
        rf"chunk_rows=\[{rows}\], round_index=0"
    )
    if stepped:
        program.acquire(soc, step_rounds=True, progress=False, round_hook=record_round)
        with pytest.raises(AcquisitionContractError, match=message):
            program.finish_round()
    else:
        with pytest.raises(AcquisitionContractError, match=message):
            program.acquire(soc, progress=False, round_hook=record_round)
    captured = program.get_raw()
    assert captured is not None
    np.testing.assert_array_equal(
        captured[0], [[[11, -11], [12, -12]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]]
    )
    assert program.get_rounds() == []
    assert completed_rounds == []


@pytest.mark.parametrize("completed_round", [False, True])
def test_cancel_discards_partial_chunks_and_retains_only_completed_rounds(
    monkeypatch: pytest.MonkeyPatch, completed_round: bool
) -> None:
    soc, program = make_chunk_acquisition(reps=3)
    stop = Event()
    full = np.array([[10, -10], [20, -20]] * 3, dtype=np.int64)
    batches = iter(
        [
            [(3, ([full], {}))],
            [(1, ([np.array([[91, -91], [92, -92]], dtype=np.int64)], {}))],
        ]
        if completed_round
        else [
            [(1, ([np.array([[91, -91], [92, -92]], dtype=np.int64)], {}))],
        ]
    )

    def poll_data():
        batch = next(batches, None)
        if batch is not None:
            return batch
        # The first partial chunk has been stored before this poll requests stop.
        stop.set()
        return [(2, ([np.full((4, 2), 99, dtype=np.int64)], {}))]

    monkeypatch.setattr(soc, "poll_data", poll_data)
    program.acquire(soc, rounds=2, progress=False, step_rounds=True, cancel_flag=stop)
    if completed_round:
        assert program.finish_round() is True
        program.prepare_round()
    assert program.finish_round() is False
    captured = program.get_raw()
    assert captured is not None
    np.testing.assert_array_equal(captured[0], np.zeros((3, 2, 2), dtype=np.int64))
    if completed_round:
        result = program.finish_acquire()
        assert result is not None
        np.testing.assert_allclose(
            result[0], np.array([[10, -10], [20, -20]]) / program.ro_chs[0]["length"]
        )
    else:
        with pytest.raises(
            StoppedPartialAcquireError, match="before the first round completed"
        ):
            program.finish_acquire()
