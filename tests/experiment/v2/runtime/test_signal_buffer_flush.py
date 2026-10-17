"""SignalBuffer: an explicit flush reaches the update callback despite the throttle."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from zcu_tools.experiment.v2.runtime.schedule import SignalBuffer
from zcu_tools.utils import func_tools


def test_signal_buffer_flush_forces_throttled_final_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    updates: list[np.ndarray[Any, Any]] = []
    times = iter([10.0, 12.0, 12.1, 13.0, 14.0])
    monkeypatch.setattr(func_tools.time, "time", lambda: next(times))
    buffer = SignalBuffer(
        (1,),
        dtype=np.float64,
        on_update=lambda data: updates.append(data.copy()),
        update_interval=0.5,
    )

    buffer.trigger_update()
    buffer.trigger_update()
    buffer.trigger_update(flush=True)

    assert len(updates) == 2


def test_signal_buffer_write_triggers_update_and_manual_trigger():
    updates: list[np.ndarray] = []
    buffer = SignalBuffer(
        (2,),
        dtype=np.float64,
        on_update=lambda data: updates.append(data.copy()),
        update_interval=None,
    )

    buffer.set(np.array([1.0, 2.0]))
    slot = buffer[1]
    slot.set(np.array(3.0))
    buffer.trigger_update()

    assert np.asarray(slot.value).item() == 3.0
    assert np.allclose(updates[0], [1.0, 2.0])
    assert np.allclose(updates[1], [1.0, 3.0])
    assert np.allclose(updates[2], [1.0, 3.0])
