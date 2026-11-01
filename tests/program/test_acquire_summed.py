from tests.program._timeline import compile_modules
"""Joint-ADC per-point sums and completed-round lifecycle at the public seam."""

from collections.abc import Mapping
from threading import Event
from typing import TypedDict

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.program.acquisition import StoppedPartialAcquireError

from tests.program._acquisition_soc import make_chunk_acquisition


def test_summed_snapshots_reused_processor_output_buffers() -> None:
    soc, prog = make_chunk_acquisition()
    payload = np.zeros((1, 2), dtype=np.float64)

    def processor(iq):
        payload[:] += 1
        return payload

    np.testing.assert_array_equal(
        prog.acquire_summed(soc, processor=processor, rounds=2, progress=False), [3, 3]
    )


def test_unswept_scalar_sums_and_later_decimated_acquisition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    soc, prog = make_chunk_acquisition(reps=2)
    result = prog.acquire_summed(
        soc, processor=lambda iq: np.array([2.0]), rounds=2, progress=False
    )
    np.testing.assert_array_equal(result, np.array(4.0))
    monkeypatch.setattr(
        soc, "get_decimated", lambda ch, address, length: np.full((length, 2), 5.0)
    )
    hooks = []
    decimated = prog.acquire_decimated(
        soc,
        rounds=2,
        progress=False,
        remove_offset=False,
        round_hook=lambda count, data, stop: hooks.append(data[0].copy()),
    )
    assert decimated is not None
    np.testing.assert_array_equal(decimated[0], 5.0)
    np.testing.assert_array_equal(hooks[-1], decimated[0])


@pytest.mark.parametrize("stepped", [False, True])
@pytest.mark.parametrize("completed", [False, True])
def test_summed_discards_incomplete_round_and_keeps_completed_rounds(
    monkeypatch: pytest.MonkeyPatch, stepped: bool, completed: bool
) -> None:
    soc, prog = make_chunk_acquisition(reps=2)
    stop = Event()
    full = np.full((4, 2), prog.ro_chs[0]["length"], dtype=np.int64)
    batches = iter(
        ([[(2, ([full], {}))]] if completed else []) + [[(1, ([full[:2]], {}))]]
    )
    hooks = []

    def poll():
        batch = next(batches, None)
        if batch is None:
            stop.set()
            return []
        return batch

    monkeypatch.setattr(soc, "poll_data", poll)

    def acquire():
        result = prog.acquire_summed(
            soc,
            processor=lambda iq: iq[0][:, :, 0].sum(axis=1),
            rounds=2,
            progress=False,
            step_rounds=stepped,
            cancel_flag=stop,
            round_hook=lambda count, data, flag: hooks.append(data.copy()),
        )
        if stepped:
            assert result is None
            if completed:
                assert prog.finish_round()
                prog.prepare_round()
            assert prog.finish_round() is False
            result = prog.finish_acquire()
        return result

    if completed:
        np.testing.assert_array_equal(acquire(), [2, 2])
        np.testing.assert_array_equal(hooks, [[2, 2]])
    else:
        with pytest.raises(StoppedPartialAcquireError):
            acquire()
        assert hooks == []


@pytest.mark.parametrize(
    "payload",
    [
        np.array(1.0),
        np.ones((2, 3)),
        np.ones((1, 0)),
        np.ones((1, 2), dtype=np.float32),
        [[1.0]],
    ],
)
def test_summed_rejects_invalid_processor_outputs_without_hooks(payload) -> None:
    soc, prog = make_chunk_acquisition()
    hooks = []
    with pytest.raises(ValueError, match="expected.*actual"):
        prog.acquire_summed(
            soc,
            processor=lambda iq: payload,
            progress=False,
            round_hook=lambda count, data, stop: hooks.append(count),
        )
    assert hooks == []


class _RejectedOptions(TypedDict, total=False):
    threshold: None
    angle: None
    herald: None


@pytest.mark.parametrize(
    "options", [{"threshold": None}, {"angle": None}, {"herald": None}]
)
def test_summed_rejects_qick_processing_options(options: _RejectedOptions) -> None:
    soc, prog = make_chunk_acquisition()
    option = next(iter(options))
    with pytest.raises(ValueError, match=rf"expected.*{option}.*actual"):
        prog.acquire_summed(soc, processor=lambda iq: np.ones((1, 2)), **options)


def test_summed_rejects_shape_changes_and_resets_on_next_acquire() -> None:
    soc, prog = make_chunk_acquisition()
    outputs = iter([np.ones((1, 2)), np.ones((1, 3))])
    hooks = []
    with pytest.raises(
        ValueError, match=r"expected shape \(1, 2\).*actual shape \(1, 3\)"
    ):
        prog.acquire_summed(
            soc,
            processor=lambda iq: next(outputs),
            rounds=2,
            progress=False,
            round_hook=lambda count, data, stop: hooks.append(count),
        )
    assert hooks == [1]
    np.testing.assert_array_equal(
        prog.acquire_summed(soc, processor=lambda iq: np.ones((1, 3)), progress=False),
        [1, 1, 1],
    )


@pytest.mark.parametrize("reps_innermost", [False, True])
def test_summed_normalizes_moves_reps_and_adds_rounds(
    monkeypatch: pytest.MonkeyPatch, reps_innermost: bool
) -> None:
    soc, source = make_chunk_acquisition(reps=2, sweep=[("outer", 2), ("inner", 2)])
    prog = compile_modules(
        source.soccfg,
        modules=source.modules,
        reps=2,
        sweep=[("outer", 2), ("inner", 2)],
        reps_innermost=reps_innermost,
    )
    prog.soccfg["readouts"][0]["iq_offset"] = 17.0
    iq = np.arange(1, 33, dtype=np.float64).reshape(2, 2, 2, 2, 2)
    raw = (iq * prog.ro_chs[0]["length"]).astype(np.int64)
    monkeypatch.setattr(soc, "poll_data", lambda: [(8, ([raw.reshape(-1, 2)], {}))])
    inputs = []
    hooks = []

    def process(values: Mapping[int, NDArray[np.float64]]) -> NDArray[np.float64]:
        inputs.append(values[0].copy())
        return values[0][:, :, 0].sum(axis=1)

    result = prog.acquire_summed(
        soc,
        processor=process,
        rounds=2,
        progress=False,
        round_hook=lambda count, data, stop: hooks.append((count, data.copy())),
    )
    expected = np.array(
        [[[6, 8], [22, 24]], [[38, 40], [54, 56]]]
        if reps_innermost
        else [[[18, 20], [26, 28]], [[34, 36], [42, 44]]],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(
        inputs[0], np.moveaxis(iq, 2 if reps_innermost else 0, 2).reshape(4, 2, 2, 2)
    )
    np.testing.assert_array_equal(result, expected * 2)
    np.testing.assert_array_equal(hooks[0][1], expected)
    np.testing.assert_array_equal(hooks[1][1], result)
    assert [count for count, _ in hooks] == [1, 2]
    assert inputs[0].dtype == np.float64
