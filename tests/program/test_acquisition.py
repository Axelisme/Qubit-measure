"""Public acquisition classifies normalized IQ without depending on lab cores."""

import inspect
import logging
from threading import Event

import numpy as np
import pytest
from zcu_tools.analysis.singleshot import classify_result
from zcu_tools.program.acquisition import StoppedPartialAcquireError
from zcu_tools.program.v2.modular import ModularProgramV2
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg

from tests.program._acquisition_soc import make_acquisition_soc


@pytest.mark.parametrize("stepped", [False, True])
def test_acquire_threshold_warning_identifies_the_public_caller(stepped: bool) -> None:
    soc, soccfg = make_acquisition_soc()
    program = ModularProgramV2(
        soccfg,
        modules=[
            PulseReadoutCfg(
                gen_ch=0,
                nqz=1,
                gain=0.1,
                waveform=ConstWaveformCfg(length=1.0),
                ro_ch=0,
                ro_length=1.0,
                freq=100.0,
            ).build("readout")
        ],
        reps=1,
    )
    options = dict(threshold=1.0, ge_radius=0.5, progress=False)
    call_lines: list[int] = []
    if stepped:
        program.acquire(
            soc,
            step_rounds=True,
            **options,
            rounds=2,
        )
        with pytest.warns(
            UserWarning,
            match="^Both threshold and ge_radius are set, threshold will be applied and ge_radius will be ignored$",
        ) as recorded:
            for _ in range(2):
                call_lines.append(inspect.stack()[0].lineno + 1)
                program.finish_round()
        result = program.finish_acquire()
    else:
        with pytest.warns(
            UserWarning,
            match="^Both threshold and ge_radius are set, threshold will be applied and ge_radius will be ignored$",
        ) as recorded:
            call_lines.extend([inspect.stack()[0].lineno + 1] * 2)
            result = program.acquire(
                soc,
                **options,
                rounds=2,
            )
    assert len(recorded) == 2
    assert all(warning.filename == __file__ for warning in recorded)
    assert [warning.lineno for warning in recorded] == call_lines
    assert result is not None
    np.testing.assert_array_equal(result[0], np.zeros((1, 2)))


@pytest.mark.parametrize("radius", [0.5, 1.0, 1.5, 2.0, 3.0])
@pytest.mark.parametrize("remove_offset", [False, True])
def test_acquire_population_matches_offline_classification(
    monkeypatch: pytest.MonkeyPatch, radius: float, remove_offset: bool
) -> None:
    signals = np.array([-1, 1, 0, 0.2, -0.2, 4, -1 + radius], dtype=np.complex128)
    expected = classify_result(signals, -1, 1, radius)
    assert np.all(sum(mask.astype(int) for mask in expected) == 1)
    assert expected[2][2]  # An exact tie belongs to neither population.
    soc, soccfg = make_acquisition_soc()
    offset = 0.25 if remove_offset else 0.0
    soccfg["readouts"][0]["iq_offset"] = offset
    program = ModularProgramV2(
        soccfg,
        modules=[
            PulseReadoutCfg(
                gen_ch=0,
                nqz=1,
                gain=0.1,
                waveform=ConstWaveformCfg(length=1.0),
                ro_ch=0,
                ro_length=soccfg.cycles2us(100, ro_ch=0),
                freq=100.0,
            ).build("readout")
        ],
        sweep=[("sample", len(signals))],
        reps=1,
    )
    raw = np.rint(
        (np.stack((signals.real, signals.imag), axis=-1) + offset)
        * program.ro_chs[0]["length"]
    ).astype(np.int64)
    delivered = False

    def poll_data(totaltime: float = 0.1, timeout: float | None = None):
        nonlocal delivered
        if delivered:
            return []
        delivered = True
        return [(len(signals), ([raw], {}))]

    monkeypatch.setattr(soc, "poll_data", poll_data)
    actual = program.acquire(
        soc,
        g_center=-1,
        e_center=1,
        ge_radius=radius,
        remove_offset=remove_offset,
        progress=False,
    )
    assert actual is not None
    np.testing.assert_array_equal(actual[0][0], np.stack(expected[:2], axis=-1))
    assert actual[0].dtype == np.float64


def _make_chunk_acquisition(*, reps: int = 4):
    soc, soccfg = make_acquisition_soc()
    readouts = [
        PulseReadoutCfg(
            gen_ch=0,
            nqz=1,
            gain=0.1 * (index + 1),
            waveform=ConstWaveformCfg(length=1.0),
            ro_ch=0,
            ro_length=1.0,
            freq=100.0,
        ).build(f"readout_{index}")
        for index in range(2)
    ]
    program = ModularProgramV2(
        soccfg,
        modules=readouts,
        reps=reps,
    )
    return soc, program


@pytest.mark.parametrize("stepped", [False, True])
def test_acquire_preserves_chunk_offsets_and_read_order(
    monkeypatch: pytest.MonkeyPatch, stepped: bool
) -> None:
    soc, program = _make_chunk_acquisition()
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


def test_acquire_logs_size_mismatch_but_keeps_numpy_broadcasting(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    soc, program = _make_chunk_acquisition(reps=3)
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


@pytest.mark.parametrize("rows", [7, 8])
def test_acquire_logs_overflow_before_propagating_slice_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, rows: int
) -> None:
    soc, program = _make_chunk_acquisition(reps=3)
    monkeypatch.setattr(
        soc,
        "poll_data",
        lambda: [(4, ([np.full((rows, 2), 17, dtype=np.int64)], {}))],
    )
    with (
        caplog.at_level(logging.ERROR, logger="qick.qick_asm"),
        pytest.raises(
            ValueError,
            match=rf"could not broadcast input array from shape \({rows},2\) into shape \(6,2\)",
        ),
    ):
        program.acquire(soc, progress=False)
    expected = []
    if rows == 7:
        expected.append("data size mismatch: new_points=4, nreads=2, data shape (7, 2)")
    expected.append("got too much data: count=0, new_points=4, total_count=3")
    assert caplog.messages == expected
    captured = program.get_raw()
    assert captured is not None
    np.testing.assert_array_equal(captured[0], np.zeros((3, 2, 2), dtype=np.int64))


@pytest.mark.parametrize("completed_round", [False, True])
def test_cancel_discards_partial_chunks_and_retains_only_completed_rounds(
    monkeypatch: pytest.MonkeyPatch, completed_round: bool
) -> None:
    soc, program = _make_chunk_acquisition(reps=3)
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
