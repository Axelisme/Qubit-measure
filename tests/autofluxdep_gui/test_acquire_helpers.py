from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from zcu_tools.experiment.v2.runner import StopSignal, schedule_stop_scope
from zcu_tools.gui.app.autofluxdep.nodes import acquire as acquire_mod
from zcu_tools.gui.app.autofluxdep.nodes.builder import RunEnv
from zcu_tools.gui.app.autofluxdep.nodes.qubit_freq import QubitFreqBuilder
from zcu_tools.program.v2 import Module, ProgramV2Cfg
from zcu_tools.progress_bar import BaseProgressBar, use_pbar_factory
from zcu_tools.progress_bar.base import ProgressTotal, ProgressValue


class FakeModule(Module):
    def __init__(self, name: str) -> None:
        self.name = name

    def init(self, prog: Any) -> None:
        pass

    def run(self, prog: Any, t: Any = 0.0) -> Any:
        return t


class RecordingProgressBar(BaseProgressBar):
    def __init__(
        self,
        *,
        total: ProgressTotal = None,
        desc: str = "",
        leave: bool = True,
        disabled: bool = False,
    ) -> None:
        self._total = total
        self._desc = desc
        self.leave = leave
        self.disabled = disabled
        self.closed = False
        self._n: ProgressValue = 0

    def set_description(self, description: str) -> None:
        self._desc = description

    def update(self, value: ProgressValue = 1) -> None:
        self._n += value

    def set_progress(self, value: ProgressValue) -> None:
        self._n = value

    def reset(self) -> None:
        self._n = 0

    def refresh(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    @property
    def total(self) -> ProgressTotal:
        return self._total

    @total.setter
    def total(self, value: ProgressTotal) -> None:
        self._total = value

    @property
    def n(self) -> ProgressValue:
        return self._n

    @property
    def desc(self) -> str:
        return self._desc


def _recording_pbar_factory(
    records: list[RecordingProgressBar],
) -> Callable[..., RecordingProgressBar]:
    def factory(*args: Any, **kwargs: Any) -> RecordingProgressBar:
        desc = kwargs.get("desc", args[1] if len(args) > 1 else "")
        total = kwargs.get("total", args[2] if len(args) > 2 else None)
        bar = RecordingProgressBar(
            total=total,
            desc=str(desc) if desc else "",
            leave=bool(kwargs.get("leave", True)),
            disabled=bool(kwargs.get("disable", False)),
        )
        records.append(bar)
        return bar

    return factory


def test_snr_stop_checker_waits_until_probe_has_value(monkeypatch: Any):
    calls = {"count": 0}

    def fake_snr_checker(ctx: Any, snr_threshold: float | None, signal2real_fn: Any):
        assert snr_threshold == 50.0

        def check() -> bool:
            calls["count"] += 1
            assert ctx.value is not None
            signal2real_fn(ctx.value)
            return True

        return check

    monkeypatch.setattr(acquire_mod, "snr_checker", fake_snr_checker)
    probe = acquire_mod.SnrProbe()
    env = RunEnv(
        flux=0.0,
        flux_idx=0,
        schema=QubitFreqBuilder().make_default_schema(),
    )

    checkers = acquire_mod.build_stop_checkers(
        env,
        probe,
        lambda signals: np.asarray(signals, dtype=np.complex128).real,
    )

    assert len(checkers) == 1
    assert checkers[0]() is False
    assert calls["count"] == 0

    probe.value = np.asarray([1.0 + 0.0j], dtype=np.complex128)
    assert checkers[0]() is True
    assert calls["count"] == 1


def test_acquire_retry_reads_default_and_validates_non_negative():
    schema = QubitFreqBuilder().make_default_schema()
    env = RunEnv(flux=0.0, flux_idx=0, schema=schema, knobs_snapshot={})
    assert acquire_mod.acquire_retry(env) == acquire_mod.DEFAULT_ACQUIRE_RETRY

    env = RunEnv(
        flux=0.0,
        flux_idx=0,
        schema=schema,
        knobs_snapshot={"acquire_retry": 0},
    )
    assert acquire_mod.acquire_retry(env) == 0

    env = RunEnv(
        flux=0.0,
        flux_idx=0,
        schema=schema,
        knobs_snapshot={"acquire_retry": 2},
    )
    assert acquire_mod.acquire_retry(env) == 2

    env = RunEnv(
        flux=0.0,
        flux_idx=0,
        schema=schema,
        node_name="qubit_freq",
        knobs_snapshot={"acquire_retry": -1},
    )
    with pytest.raises(RuntimeError, match="acquire_retry must be non-negative"):
        acquire_mod.acquire_retry(env)


def test_run_schedule_acquire_completed_updates_signal():
    updates: list[np.ndarray] = []
    progress_bars: list[RecordingProgressBar] = []

    class SuccessfulProgram:
        def __init__(
            self,
            soccfg: Any,
            cfg: ProgramV2Cfg,
            *,
            modules: list[Module],
            sweep: list[tuple[str, Any]] | None,
        ) -> None:
            self.cfg_model = cfg
            self.modules = modules
            self.sweep = sweep

        def acquire(
            self,
            soc: Any,
            *,
            progress: bool,
            round_hook: Any,
            stop_checkers: list[Any],
            **_kwargs: Any,
        ) -> np.ndarray:
            assert soc == "soc"
            assert progress is False
            assert all(not checker() for checker in stop_checkers)
            round_hook(1, np.array([1.0]))
            return np.array([2.0])

        def acquire_decimated(self, *_args: Any, **_kwargs: Any) -> list[np.ndarray]:
            raise NotImplementedError

    env = RunEnv(
        flux=0.0,
        flux_idx=0,
        schema=QubitFreqBuilder().make_default_schema(),
        soc="soc",
        soccfg="soccfg",
        knobs_snapshot={"acquire_retry": 0},
    )

    with use_pbar_factory(_recording_pbar_factory(progress_bars)):
        acquired = acquire_mod.run_schedule_acquire(
            env=env,
            cfg=ProgramV2Cfg(rounds=2),
            signal_shape=(1,),
            dtype=np.float64,
            configure_builder=lambda builder: builder.add(FakeModule("readout")),
            raw2signal_fn=lambda raw: np.asarray(raw, dtype=np.float64),
            on_update=lambda data: updates.append(data.copy()),
            program_cls=SuccessfulProgram,
        )

    assert acquired.stopped is False
    assert acquired.signal is not None
    np.testing.assert_allclose(acquired.signal, np.array([2.0]))
    np.testing.assert_allclose(updates[0], np.array([1.0]))
    np.testing.assert_allclose(updates[-1], np.array([2.0]))
    assert [bar.leave for bar in progress_bars] == [False]
    assert progress_bars[0].closed is True


def test_run_schedule_acquire_failed_raises_after_retry_exhaustion():
    attempts: list[int] = []

    class FailingProgram:
        def __init__(
            self,
            soccfg: Any,
            cfg: ProgramV2Cfg,
            *,
            modules: list[Module],
            sweep: list[tuple[str, Any]] | None,
        ) -> None:
            self.cfg_model = cfg
            self.modules = modules
            self.sweep = sweep

        def acquire(self, *_args: Any, round_hook: Any, **_kwargs: Any) -> np.ndarray:
            attempts.append(1)
            round_hook(1, np.array([float(len(attempts))]))
            raise RuntimeError("acquire failed")

        def acquire_decimated(self, *_args: Any, **_kwargs: Any) -> list[np.ndarray]:
            raise NotImplementedError

    env = RunEnv(
        flux=0.0,
        flux_idx=0,
        schema=QubitFreqBuilder().make_default_schema(),
        soc="soc",
        soccfg="soccfg",
        knobs_snapshot={"acquire_retry": 1},
    )

    with pytest.raises(RuntimeError, match="RuntimeError: acquire failed"):
        acquire_mod.run_schedule_acquire(
            env=env,
            cfg=ProgramV2Cfg(rounds=1),
            signal_shape=(1,),
            dtype=np.float64,
            configure_builder=lambda builder: builder.add(FakeModule("readout")),
            raw2signal_fn=lambda raw: np.asarray(raw, dtype=np.float64),
            program_cls=FailingProgram,
        )

    assert attempts == [1, 1]


def test_run_schedule_acquire_stopped_returns_sentinel_without_failure():
    stop = StopSignal()

    class StoppingProgram:
        def __init__(
            self,
            soccfg: Any,
            cfg: ProgramV2Cfg,
            *,
            modules: list[Module],
            sweep: list[tuple[str, Any]] | None,
        ) -> None:
            self.cfg_model = cfg
            self.modules = modules
            self.sweep = sweep

        def acquire(self, *_args: Any, round_hook: Any, **_kwargs: Any) -> np.ndarray:
            round_hook(1, np.array([1.0]))
            stop.set_stop()
            return np.array([2.0])

        def acquire_decimated(self, *_args: Any, **_kwargs: Any) -> list[np.ndarray]:
            raise NotImplementedError

    env = RunEnv(
        flux=0.0,
        flux_idx=0,
        schema=QubitFreqBuilder().make_default_schema(),
        soc="soc",
        soccfg="soccfg",
        knobs_snapshot={"acquire_retry": 3},
    )

    with schedule_stop_scope(stop):
        acquired = acquire_mod.run_schedule_acquire(
            env=env,
            cfg=ProgramV2Cfg(rounds=1),
            signal_shape=(1,),
            dtype=np.float64,
            configure_builder=lambda builder: builder.add(FakeModule("readout")),
            raw2signal_fn=lambda raw: np.asarray(raw, dtype=np.float64),
            program_cls=StoppingProgram,
        )

    assert acquired.stopped is True
    assert acquired.signal is None
