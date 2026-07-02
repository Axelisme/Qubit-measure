from __future__ import annotations

import pytest
from zcu_tools.utils import func_tools


def test_min_interval_throttles_by_duty_cycle_ratio(monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []
    times = iter([10.0, 12.0, 13.0, 15.0, 16.0])
    monkeypatch.setattr(func_tools.time, "time", lambda: next(times))

    throttled = func_tools.MinIntervalFunc(lambda value: calls.append(value), 0.5)

    throttled("first")
    throttled("skipped")
    throttled("second")

    assert calls == ["first", "second"]


def test_min_interval_rejects_invalid_duty_cycle_ratio():
    with pytest.raises(ValueError, match="duty-cycle ratio"):
        func_tools.MinIntervalFunc(lambda: None, 0.0)

    with pytest.raises(ValueError, match="duty-cycle ratio"):
        func_tools.MinIntervalFunc(lambda: None, 1.1)


def test_min_interval_func_keeps_historical_keyword_name():
    wrapped = func_tools.MinIntervalFunc(lambda: None, min_interval=0.5)

    assert wrapped.duty_cycle_ratio == 0.5


def test_min_interval_factory_passthrough_cases():
    def callback() -> None:
        return None

    assert func_tools.min_interval(None, 0.5) is None
    assert func_tools.min_interval(callback, None) is callback
    assert func_tools.min_interval(callback, 1.0) is callback


def test_min_interval_factory_wraps_callback():
    def callback() -> None:
        return None

    wrapped = func_tools.min_interval(callback, 0.5)

    assert isinstance(wrapped, func_tools.MinIntervalFunc)
