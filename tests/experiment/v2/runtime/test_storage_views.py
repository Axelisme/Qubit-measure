import numpy as np
import pytest
from zcu_tools.experiment.v2.runtime.result_tree import ResultTree
from zcu_tools.experiment.v2.runtime.schedule import SignalBuffer


def test_scalar_and_slice_result_storage_views_write_through() -> None:
    storage = np.arange(6.0).reshape(2, 3)
    tree = ResultTree([{"signal": storage}])
    node = tree.at(0).child("signal")
    assert node.data is storage

    scalar = node.child(-1).child(1)
    assert scalar.data.shape == ()
    assert np.shares_memory(scalar.data, storage)
    scalar.set(np.array(20.0))
    assert storage[1, 1] == 20.0

    sliced = node.child(slice(None)).child(slice(0, 3, 2))
    assert sliced.data.shape == (2, 2)
    assert np.shares_memory(sliced.data, storage)
    sliced.set(np.array([[10.0, 12.0], [13.0, 15.0]]))
    expected = np.array([[10.0, 1.0, 12.0], [13.0, 20.0, 15.0]])
    np.testing.assert_array_equal(storage, expected)

    buffer = SignalBuffer((2, 3), dtype=np.float64, update_interval=None)
    buffer.set(np.arange(6.0).reshape(2, 3))
    assert buffer.at().value is buffer.data

    slot = buffer[-1, 1]
    assert slot.value.shape == ()
    assert np.shares_memory(slot.value, buffer.data)
    slot.set(np.array(20.0))
    assert buffer.data[1, 1] == 20.0

    slice_slot = buffer.at(slice(None), slice(0, 3, 2))
    assert slice_slot.value.shape == (2, 2)
    assert np.shares_memory(slice_slot.value, buffer.data)
    slice_slot.set(np.array([[10.0, 12.0], [13.0, 15.0]]))
    np.testing.assert_array_equal(buffer.data, expected)


def test_advanced_index_copy_rejection_keeps_result_storage() -> None:
    initial = np.arange(6.0).reshape(2, 3)
    storage = initial.copy()
    tree = ResultTree([{"signal": storage}])
    node = tree.at(0).child("signal")
    buffer = SignalBuffer((2, 3), dtype=np.float64, update_interval=None)
    buffer.set(initial)

    with pytest.raises(
        ValueError, match="^NDArray path indexing must select a writable view$"
    ):
        _ = node.child((0, 1)).data
    with pytest.raises(
        ValueError, match="^NDArray path indexing must select a writable view$"
    ):
        buffer.at((0, 1))

    with pytest.raises(
        ValueError, match="^Scalar NDArray path indexing only supports integer axes$"
    ):
        _ = node.child(np.int64(0)).child(1).data
    with pytest.raises(
        ValueError, match="^Scalar NDArray path indexing only supports integer axes$"
    ):
        buffer.at(np.int64(0), 1)

    with pytest.raises(IndexError, match="index 2 is out of bounds"):
        _ = node.child(2).child(0).data
    with pytest.raises(IndexError, match="index 2 is out of bounds"):
        buffer.at(2, 0)

    np.testing.assert_array_equal(storage, initial)
    np.testing.assert_array_equal(buffer.data, initial)
