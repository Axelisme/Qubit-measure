"""SignalBuffer: an explicit flush reaches the update callback despite the throttle."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from zcu_tools.experiment.v2.runner import SignalBuffer
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
