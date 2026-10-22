"""RunService admission builders and controllable operation/background fakes."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any
from unittest.mock import MagicMock

import numpy as np
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.device import BaseDevice
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    ContextReadiness,
    ExpAdapterProtocol,
    RunRequest,
)
from zcu_tools.gui.app.measure.events.run import RunFinishedPayload, RunStartedPayload
from zcu_tools.gui.app.measure.events.tab import TabInteractionChangedPayload
from zcu_tools.gui.app.measure.services.guard import RunInputs
from zcu_tools.gui.app.measure.services.operation_gate import (
    OperationGate,
)
from zcu_tools.gui.app.measure.services.run import RunService
from zcu_tools.gui.app.measure.state import Session, SessionEnv, State
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.event_bus import BasePayload
from zcu_tools.gui.form_adapter import Form
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.operation_runner import (
    OperationRunner,
)
from zcu_tools.gui.session.services.progress import ProgressService
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.program.v2 import Module, ProgramV2Cfg

from tests.gui._progress_fakes import DirectProgressTransport
from tests.gui.app.measure._cfg_fakes import parameter_capture
from tests.gui.app.measure._form_fakes import make_form


def make_state(
    *,
    readiness: ContextReadiness = ContextReadiness.EMPTY,
) -> tuple[State, str, MagicMock]:
    state = State(
        SessionEnv(
            parameters=parameter_capture(),
            soc=MagicMock(),
            soccfg=MagicMock(),
            readiness=readiness,
        )
    )
    tab_id = "tab-1"
    adapter = MagicMock()
    adapter.run.return_value = RunRecord(None, object())
    adapter.capabilities = AdapterCapabilities(requires_soc=True)
    state.add_tab(
        tab_id,
        Session(
            adapter_name="any",
            adapter=adapter,
            cfg=make_form(Form(ConfigBase, {}), state=state),
        ),
    )
    return state, tab_id, adapter


def make_plots() -> Plots:
    return Plots(NonPresentingHost())


def make_inputs(state: State, tab_id: str, adapter: ExpAdapterProtocol) -> RunInputs:
    ctx = state.session_env
    parameters = ctx.parameters
    return RunInputs(
        tab_id=tab_id,
        adapter_name=state.get_tab(tab_id).adapter_name,
        request=RunRequest(
            soc=ctx.soc,
            soccfg=ctx.soccfg,
            cfg_context=None
            if parameters is None
            else parameters.library.validation_context(),
        ),
        accepted_cfg=state.get_tab(tab_id).cfg.accept(),
        adapter=adapter,
        parameters=parameters,
    )


class FakeBg:
    """Synchronous background executor stub: calls work() and on_done/on_error inline."""

    def __init__(self, *, fail_submit: bool = False, fail_work: bool = False) -> None:
        self.bus = EventBus()
        self.events: list[BasePayload] = []
        self.bus.subscribe(RunStartedPayload, self.events.append)
        self.bus.subscribe(RunFinishedPayload, self.events.append)
        self.bus.subscribe(TabInteractionChangedPayload, self.events.append)
        self._fail_submit = fail_submit
        self._fail_work = fail_work
        self.last_work: Callable[[], Any] | None = None
        self.last_on_done: Callable[[Any], None] | None = None
        self.last_on_error: Callable[[Exception], None] | None = None

    def submit(
        self,
        work: Callable[[], Any],
        *,
        run_in_pool: bool,
        on_done: Callable[[Any], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        if self._fail_submit:
            raise RuntimeError("worker boom")
        self.last_work = work
        self.last_on_done = on_done
        self.last_on_error = on_error

    def run_work(self) -> None:
        """Drive the captured work synchronously (for tests that need to inspect outcome)."""
        assert self.last_work is not None
        if self._fail_work:
            assert self.last_on_error is not None
            self.last_on_error(RuntimeError("work boom"))
        else:
            try:
                result = self.last_work()
                assert self.last_on_done is not None
                self.last_on_done(result)
            except (RuntimeError, ValueError, TypeError, OSError) as exc:
                assert self.last_on_error is not None
                self.last_on_error(exc)


class ConstructorFailingProgram:
    @property
    def cfg_model(self) -> ProgramV2Cfg:
        raise NotImplementedError

    def __init__(
        self,
        soccfg: Any,
        cfg: ProgramV2Cfg,
        *,
        modules: list[Any],
        sweep: list[tuple[str, Any]] | None,
    ) -> None:
        raise RuntimeError("builder boom")

    def acquire(self, *_args: Any, **_kwargs: Any) -> np.ndarray:
        raise NotImplementedError

    def acquire_decimated(self, *_args: Any, **_kwargs: Any) -> list[np.ndarray]:
        raise NotImplementedError


class NoopModule(Module):
    def __init__(self, name: str) -> None:
        self.name = name

    def init(self, prog: Any) -> None:
        pass

    def run(self, prog: Any, t: Any = 0.0) -> Any:
        return t


def make_run_service(
    state: State,
    *,
    fail_submit: bool = False,
    devices: Mapping[str, BaseDevice[Any]] | None = None,
) -> tuple[RunService, OperationGate, FakeBg, OperationHandles]:
    bg = FakeBg(fail_submit=fail_submit)
    bus = bg.bus
    gate = OperationGate(bus)
    handles = OperationHandles()
    writeback = MagicMock()
    progress = ProgressService(DirectProgressTransport())
    runner = OperationRunner(gate, handles, progress, bg, bus)
    svc = RunService(
        state,
        runner,
        bus,
        handles,
        writeback,
        gate=gate,
        devices=lambda: {} if devices is None else devices,
        refresh_devices=lambda: None,
    )
    return svc, gate, bg, handles


def last_run_finished_payload(events: Sequence[BasePayload]) -> RunFinishedPayload:
    """Return the last finished outcome observed through the supplied event bus."""
    for payload in reversed(events):
        if isinstance(payload, RunFinishedPayload):
            return payload
    raise AssertionError("no RUN_FINISHED emitted")
