"""Public per-shot processing contracts, herald composition, and round lifecycle."""

from threading import Event

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.cfg_model import LinearClassifier
from zcu_tools.program import herald as herald_processing
from zcu_tools.program.acquisition import StoppedPartialAcquireError
from zcu_tools.program.herald import HeraldSpec
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.program.v2.modular import ModularProgramV2
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg

from tests.program._acquisition_soc import AcquisitionSoc, make_chunk_acquisition


@pytest.mark.parametrize("remove_offset", [False, True])
@pytest.mark.parametrize("stepped", [False, True])
def test_processed_receives_all_normalized_reads_and_averages_payload(
    monkeypatch: pytest.MonkeyPatch, remove_offset: bool, stepped: bool
) -> None:
    soc, program = make_chunk_acquisition(reps=2, sweep=[("outer", 2), ("inner", 2)])
    program.soccfg["readouts"][0]["iq_offset"] = 7.0
    iq = np.arange(1, 33, dtype=np.float64).reshape(8, 2, 2)
    raw = (iq * program.ro_chs[0]["length"]).astype(np.int64)
    monkeypatch.setattr(soc, "poll_data", lambda: [(8, ([raw.reshape(-1, 2)], {}))])
    inputs = []

    def processor(values: NDArray[np.float64]) -> NDArray[np.float64]:
        inputs.append(values.copy())
        return np.concatenate(
            [values[:, :1], values[:, :, :1].sum(axis=1, keepdims=True)], axis=-1
        )

    result = program.acquire_processed(
        soc,
        processors={0: processor},
        progress=False,
        remove_offset=remove_offset,
        step_rounds=stepped,
    )
    if stepped:
        assert result is None
        assert program.finish_round() is False
        result = program.finish_acquire()
    assert len(inputs) == 1
    assert inputs[0].dtype == np.float64
    np.testing.assert_array_equal(inputs[0], iq)
    assert result is not None
    np.testing.assert_array_equal(
        result[0], [[[[9, 10, 20], [13, 14, 28]], [[17, 18, 36], [21, 22, 44]]]]
    )
    assert result[0].dtype == np.float64
    captured = program.get_raw()
    assert captured is not None
    np.testing.assert_array_equal(captured[0], raw.reshape(2, 2, 2, 2, 2))
    ordinary = program.acquire(soc, progress=False, remove_offset=False)
    assert ordinary is not None
    np.testing.assert_array_equal(ordinary[0][0, ..., 0], [[9, 13], [17, 21]])
    assert ordinary[0].shape == (2, 2, 2, 2)


@pytest.mark.parametrize(
    ("processors", "options", "message"),
    [
        ({}, {}, "processors.*ADC"),
        ({0: np.copy, 1: np.copy}, {}, "processors.*ADC"),
        ({"0": np.copy}, {}, "processors.*integer"),
        ({0.0: np.copy}, {}, "processors.*integer"),
        ({False: np.copy}, {}, "processors.*integer"),
        ({0: None}, {}, "processors.*callable"),
        ({0: np.copy}, {"threshold": None}, "threshold"),
        ({0: np.copy}, {"angle": 0.0}, "angle"),
        ({0: np.copy}, {"herald": []}, "herald"),
    ],
)
def test_processed_rejects_invalid_admission_before_transport(
    monkeypatch: pytest.MonkeyPatch, processors, options, message: str
) -> None:
    soc, program = make_chunk_acquisition()

    def unexpected_start(*_args, **_kwargs):
        pytest.fail("invalid processed acquisition reached transport")

    monkeypatch.setattr(soc, "start_readout", unexpected_start)
    with pytest.raises(ValueError, match=message):
        program.acquire_processed(soc, processors=processors, progress=False, **options)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([[[1.0]]], "processor.*ndarray"),
        (np.ones((4, 1, 1), dtype=np.float32), "processor.*float64"),
        (np.ones((4, 1, 1), dtype=np.complex128), "processor.*float64"),
        (np.ones((4, 1)), "processor.*rank"),
        (np.ones((3, 1, 1)), "processor.*shots"),
        (np.ones((4, 0, 1)), "processor.*nonempty"),
        (np.ones((4, 1, 0)), "processor.*nonempty"),
    ],
)
def test_processed_rejects_invalid_payload_without_publishing_round(
    payload, message: str
) -> None:
    soc, program = make_chunk_acquisition()
    hooks = []
    with pytest.raises(ValueError, match=message):
        program.acquire_processed(
            soc,
            processors={0: lambda _iq: payload},
            progress=False,
            round_hook=lambda count, _data, _stop: hooks.append(count),
        )
    assert hooks == []
    assert program.get_rounds() == []


@pytest.mark.parametrize("later_shape", [(4, 2, 2), (4, 1, 3)])
def test_processed_rejects_dimensions_changing_between_rounds(later_shape) -> None:
    soc, program = make_chunk_acquisition()
    payloads = iter([np.ones((4, 1, 2)), np.ones(later_shape)])
    hooks = []
    with pytest.raises(ValueError, match="ADC 0.*processor.*shape.*changed"):
        program.acquire_processed(
            soc,
            processors={0: lambda _iq: next(payloads)},
            rounds=2,
            progress=False,
            round_hook=lambda count, _data, _stop: hooks.append(count),
        )
    assert hooks == [1]
    completed = program.get_rounds()
    assert completed is not None
    assert len(completed) == 1
    restarted = program.acquire_processed(
        soc, processors={0: lambda _iq: np.ones(later_shape)}, progress=False
    )
    assert restarted is not None
    np.testing.assert_array_equal(restarted[0], np.ones(later_shape[1:]))


def test_heralded_removes_selected_read_and_zeroes_rejected_nan_payloads() -> None:
    iq = np.array(
        [
            [[2, -2], [-1, 0], [20, -20]],
            [[np.nan, np.inf], [1, 0], [np.nan, np.inf]],
            [[6, -6], [0, 0], [60, -60]],
        ],
        dtype=np.float64,
    )
    original = iq.copy()
    received = []

    def inner(values: NDArray[np.float64]) -> NDArray[np.float64]:
        received.append(values.copy())
        return values

    spec = HeraldSpec(0, 1, LinearClassifier(angle=0.0, threshold=0.0), 3)
    processor = herald_processing.heralded(spec, inner)
    with np.errstate(all="raise"):
        payload = processor(iq)
    np.testing.assert_array_equal(received[0], iq[:, [0, 2]])
    np.testing.assert_array_equal(
        payload,
        [
            [[2, -2, 1], [20, -20, 1]],
            [[0, 0, 0], [0, 0, 0]],
            [[6, -6, 1], [60, -60, 1]],
        ],
    )
    assert payload.dtype == np.float64
    np.testing.assert_array_equal(iq, original)


@pytest.mark.parametrize("stepped", [False, True])
def test_processed_herald_ratio_weights_completed_rounds_and_hooks(
    monkeypatch: pytest.MonkeyPatch, stepped: bool
) -> None:
    soc, program = make_chunk_acquisition(reps=3, sweep=[("point", 2)])
    length = program.ro_chs[0]["length"]
    batches = iter(
        [
            [
                [[-1, 0], [2, -2]],
                [[1, 0], [90, -90]],
                [[1, 0], [90, -90]],
                [[1, 0], [90, -90]],
                [[1, 0], [90, -90]],
                [[1, 0], [90, -90]],
            ],
            [
                [[-1, 0], [6, -6]],
                [[1, 0], [90, -90]],
                [[-1, 0], [10, -10]],
                [[1, 0], [90, -90]],
                [[1, 0], [90, -90]],
                [[1, 0], [90, -90]],
            ],
        ]
    )
    monkeypatch.setattr(
        soc,
        "poll_data",
        lambda: [
            (6, ([np.array(next(batches), dtype=np.int64).reshape(-1, 2) * length], {}))
        ],
    )
    hooks = []

    def hook(count, data, _stop):
        values, fraction = herald_processing.herald_ratio(data[0])
        hooks.append((count, values.copy(), fraction.copy()))

    spec = HeraldSpec(0, 0, LinearClassifier(angle=0.0, threshold=0.0), 2)
    with np.errstate(all="raise"):
        result = program.acquire_processed(
            soc,
            processors={0: herald_processing.heralded(spec, np.copy)},
            rounds=2,
            step_rounds=stepped,
            progress=False,
            round_hook=hook,
        )
        if stepped:
            assert program.finish_round() is True
            program.prepare_round()
            assert program.finish_round() is False
            result = program.finish_acquire()
        assert result is not None
        values, fraction = herald_processing.herald_ratio(result[0])
    np.testing.assert_allclose(values, [[[6, -6], [np.nan, np.nan]]], equal_nan=True)
    np.testing.assert_array_equal(fraction, [0.5, 0])
    np.testing.assert_array_equal(result[0], [[[3, -3, 0.5], [0, 0, 0]]])
    assert [count for count, _, _ in hooks] == [1, 2]
    np.testing.assert_allclose(
        hooks[0][1], [[[2, -2], [np.nan, np.nan]]], equal_nan=True
    )
    np.testing.assert_allclose(hooks[0][2], [1 / 3, 0])
    np.testing.assert_array_equal(hooks[1][1], values)
    np.testing.assert_array_equal(hooks[1][2], fraction)


def test_processed_keeps_independent_adc_shapes_and_equal_round_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    soccfg = make_mock_soccfg(n_gens=2, n_readouts=2)
    soc = AcquisitionSoc(soccfg.get_cfg())
    program = ModularProgramV2(
        soccfg,
        reps=2,
        modules=[
            PulseReadoutCfg(
                gen_ch=ch,
                ro_ch=ch,
                nqz=1,
                freq=100.0,
                gain=0.1,
                waveform=ConstWaveformCfg(length=1.0),
                ro_length=soccfg.cycles2us(100 + ch, ro_ch=ch),
            ).build(f"readout_{ch}")
            for ch in (1, 0)
        ],
    )
    batches = iter(
        [
            [[[2, -2], [4, -4]], [[10, -10], [20, -20]]],
            [[[6, -6], [8, -8]], [[30, -30], [40, -40]]],
        ]
    )

    def poll_data():
        return [
            (
                2,
                (
                    [
                        np.array(iq, dtype=np.int64) * program.ro_chs[ch]["length"]
                        for ch, iq in zip((1, 0), next(batches), strict=True)
                    ],
                    {},
                ),
            )
        ]

    monkeypatch.setattr(soc, "poll_data", poll_data)
    hooks = []
    result = program.acquire_processed(
        soc,
        processors={
            0: lambda iq: np.repeat(iq[:, :, :1], 2, axis=1),
            1: lambda iq: np.concatenate([iq, iq[:, :, :1] ** 2], axis=-1),
        },
        rounds=2,
        progress=False,
        round_hook=lambda _count, data, _stop: hooks.append([x.copy() for x in data]),
    )
    assert result is not None
    np.testing.assert_array_equal(result[0], [[5, -5, 30]])
    np.testing.assert_array_equal(result[1], [[25], [25]])
    np.testing.assert_array_equal(hooks[0][0], [[3, -3, 10]])
    np.testing.assert_array_equal(hooks[0][1], [[15], [15]])
    for actual, expected in zip(hooks[-1], result, strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("completed_round", [False, True])
@pytest.mark.parametrize("stepped", [False, True])
def test_processed_cancellation_keeps_only_completed_payloads(
    monkeypatch: pytest.MonkeyPatch, completed_round: bool, stepped: bool
) -> None:
    soc, program = make_chunk_acquisition(reps=3)
    stop = Event()
    length = program.ro_chs[0]["length"]
    full = (
        np.array(
            [[-1, 0], [2, -2], [1, 0], [90, -90], [1, 0], [90, -90]], dtype=np.int64
        )
        * length
    )
    partial = np.array([[-1, 0], [99, -99]], dtype=np.int64) * length
    batches = iter(
        [[(3, ([full], {}))], [(1, ([partial], {}))]]
        if completed_round
        else [[(1, ([partial], {}))]]
    )

    def poll_data():
        batch = next(batches, None)
        if batch is not None:
            return batch
        stop.set()
        return []

    monkeypatch.setattr(soc, "poll_data", poll_data)
    inputs = []
    hooks = []

    def inner(iq: NDArray[np.float64]) -> NDArray[np.float64]:
        inputs.append(iq.copy())
        return iq

    spec = HeraldSpec(0, 0, LinearClassifier(angle=0.0, threshold=0.0), 2)

    def acquire():
        result = program.acquire_processed(
            soc,
            processors={0: herald_processing.heralded(spec, inner)},
            rounds=2,
            progress=False,
            step_rounds=stepped,
            cancel_flag=stop,
            round_hook=lambda count, data, _stop: hooks.append((count, data[0].copy())),
        )
        if stepped:
            if completed_round:
                assert program.finish_round() is True
                program.prepare_round()
            assert program.finish_round() is False
            result = program.finish_acquire()
        return result

    if completed_round:
        result = acquire()
        assert result is not None
        with np.errstate(all="raise"):
            values, fraction = herald_processing.herald_ratio(result[0])
        np.testing.assert_array_equal(values, [[2, -2]])
        np.testing.assert_allclose(fraction, 1 / 3)
        assert len(inputs) == 1
        assert [count for count, _ in hooks] == [1]
        np.testing.assert_array_equal(hooks[0][1], result[0])
    else:
        with pytest.raises(StoppedPartialAcquireError):
            acquire()
        assert inputs == []
        assert hooks == []


def test_processed_round_hook_can_stop_before_next_round() -> None:
    soc, program = make_chunk_acquisition()
    hooks = []

    def hook(count, data, stop):
        hooks.append((count, data[0].copy()))
        stop.set()

    result = program.acquire_processed(
        soc,
        processors={0: lambda iq: iq + 3},
        rounds=3,
        progress=False,
        round_hook=hook,
    )
    assert result is not None
    assert [count for count, _ in hooks] == [1]
    np.testing.assert_array_equal(result[0], [[3, 3], [3, 3]])
    np.testing.assert_array_equal(hooks[0][1], result[0])


@pytest.mark.parametrize(
    "payload",
    [
        [[[1.0]]],
        np.ones((2, 1, 1), dtype=np.float32),
        np.ones((2, 1)),
        np.ones((1, 1, 1)),
        np.ones((2, 0, 1)),
        np.ones((2, 1, 0)),
    ],
)
def test_heralded_rejects_invalid_inner_payload(payload) -> None:
    spec = HeraldSpec(0, 0, LinearClassifier(angle=0.0, threshold=0.0), 2)
    processor = herald_processing.heralded(spec, lambda _iq: payload)
    with pytest.raises(ValueError, match="processor"):
        processor(np.zeros((2, 2, 2), dtype=np.float64))


@pytest.mark.parametrize(
    ("read_index", "declared_reads", "input_reads"),
    [
        (0, 3, 2),
        (0, 1, 1),
        (-1, 2, 2),
        (2, 2, 2),
    ],
)
def test_heralded_rejects_incompatible_read_layout(
    read_index: int, declared_reads: int, input_reads: int
) -> None:
    spec = HeraldSpec(
        0, read_index, LinearClassifier(angle=0.0, threshold=0.0), declared_reads
    )
    processor = herald_processing.heralded(spec, np.copy)
    with pytest.raises(ValueError, match="heralded.*read"):
        processor(np.zeros((2, input_reads, 2), dtype=np.float64))


@pytest.mark.parametrize(
    "result",
    [
        [[1.0, 0.5]],
        np.ones((1, 2), dtype=np.float32),
        np.ones(2),
        np.ones((0, 2)),
        np.ones((1, 1)),
        np.ones((1, 0)),
    ],
)
def test_herald_ratio_rejects_invalid_averaged_payload(result) -> None:
    with pytest.raises(ValueError, match="herald_ratio.*float64"):
        herald_processing.herald_ratio(result)
