"""RunSession: one in-memory autofluxdep sweep lifecycle."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from zcu_tools.experiment.v2.runner import StopSignal, schedule_stop_scope
from zcu_tools.gui.app.autofluxdep.cfg import RunCfgSnapshot
from zcu_tools.gui.app.autofluxdep.derivation import SmoothingService
from zcu_tools.gui.app.autofluxdep.nodes.builder import PlacedNode
from zcu_tools.gui.app.autofluxdep.orchestrator import (
    InfoStore,
    ModuleSource,
    Notify,
    Orchestrator,
    RunError,
    RunObserver,
    SkipReason,
)
from zcu_tools.gui.app.autofluxdep.services.run_store import RunStore
from zcu_tools.gui.app.autofluxdep.tools import Tools
from zcu_tools.gui.session.scopes import progress_ambient
from zcu_tools.progress_bar import make_pbar

logger = logging.getLogger(__name__)


class RunSessionStatus(str, Enum):
    READY = "ready"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    FINISHED = "finished"
    STOPPED = "stopped"
    FAILED = "failed"


class RunEventSink(Protocol):
    """Main-thread event bridge owned by Controller."""

    def emit_point_done(self, idx: int) -> None: ...
    def emit_node_entered(self, name: str, idx: int) -> None: ...
    def emit_predictor_changed(self) -> None: ...


@dataclass(frozen=True)
class RunSegmentOutcome:
    """Worker return value for one RunSession segment."""

    info: InfoStore
    run_error: RunError | None
    stopped: bool
    paused: bool
    next_flux_idx: int


class RunSession(RunObserver):
    """Owns run-lived state across one or more OperationRunner segments."""

    def __init__(
        self,
        *,
        providers: list[PlacedNode],
        user_nodes: list[PlacedNode],
        flux_values: list[float],
        flux_device: str | None,
        results: dict[str, Any],
        cfg_snapshots: dict[str, RunCfgSnapshot],
        store: RunStore,
        tools: Tools,
        ml: ModuleSource,
        soc: Any,
        soccfg: Any,
        md: Any,
        notify: Notify | None,
        event_sink: RunEventSink,
        has_loaded_predictor: bool,
        progress_label: str,
    ) -> None:
        self.providers = list(providers)
        self.user_nodes = list(user_nodes)
        self.flux_values = [float(value) for value in flux_values]
        self.flux_device = flux_device
        self.results = results
        self.cfg_snapshots = dict(cfg_snapshots)
        self.store = store
        self.tools = tools
        self.ml = ml
        self.soc = soc
        self.soccfg = soccfg
        self.md = md
        self.notify = notify
        self._event_sink = event_sink
        self._has_loaded_predictor = has_loaded_predictor
        self._progress_label = progress_label
        self._user_node_names = {node.name for node in user_nodes}
        self._user_node_types = {node.name: node.type_name for node in user_nodes}
        self._user_stop_event = threading.Event()
        self._schedule_stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._info = InfoStore()
        specs = [
            spec for provider in self.providers for spec in provider.smooth_specs()
        ]
        self._smoothing = SmoothingService.from_specs(specs) if specs else None
        self._next_flux_idx = 0
        self._status = RunSessionStatus.READY
        self._progress_bar: Any | None = None

    @property
    def status(self) -> RunSessionStatus:
        return self._status

    @property
    def next_flux_idx(self) -> int:
        return self._next_flux_idx

    @property
    def info(self) -> InfoStore:
        return self._info

    @property
    def stop_requested(self) -> bool:
        return self._user_stop_event.is_set()

    @property
    def pause_requested(self) -> bool:
        return self._pause_event.is_set()

    def request_stop(self) -> None:
        self._user_stop_event.set()
        self._schedule_stop_event.set()

    def request_pause(self) -> bool:
        if self._status not in {RunSessionStatus.READY, RunSessionStatus.RUNNING}:
            return False
        self._status = RunSessionStatus.PAUSING
        self._pause_event.set()
        return True

    def prepare_segment(self, *, continuing: bool) -> None:
        """Reset stale segment flags before the operation becomes cancellable."""
        expected = RunSessionStatus.PAUSED if continuing else RunSessionStatus.READY
        if self._status is not expected:
            raise RuntimeError(
                f"run session segment cannot start from {self._status.value}"
            )
        self._user_stop_event.clear()
        self._schedule_stop_event.clear()
        self._pause_event.clear()

    def start_or_continue(self, progress_factory: Any) -> RunSegmentOutcome:
        if self._status not in {
            RunSessionStatus.READY,
            RunSessionStatus.PAUSING,
            RunSessionStatus.PAUSED,
        }:
            raise RuntimeError(
                f"run session is not startable from {self._status.value}"
            )
        if self._next_flux_idx > len(self.flux_values):
            raise RuntimeError("run session cursor is past the flux sweep")

        if self._status is RunSessionStatus.PAUSED:
            self.store.mark_running(self._next_flux_idx)

        was_pausing = self._status is RunSessionStatus.PAUSING
        self._status = RunSessionStatus.RUNNING
        if not was_pausing:
            self._pause_event.clear()
        start_idx = self._next_flux_idx

        with (
            progress_ambient(progress_factory),
            schedule_stop_scope(StopSignal(self._schedule_stop_event)),
        ):
            pbar = make_pbar(
                total=len(self.flux_values),
                desc=self._progress_label,
                leave=True,
            )
            if start_idx:
                pbar.set_progress(start_idx)
            self._progress_bar = pbar

            try:
                orch = Orchestrator(
                    providers=self.providers,
                    tools=self.tools,
                    ml=self.ml,
                    soc=self.soc,
                    soccfg=self.soccfg,
                    md=self.md,
                    flux_device=self.flux_device,
                    results=self.results,
                    cfg_snapshots=self.cfg_snapshots,
                    notify=self.notify,
                    smoothing=self._smoothing,
                )
                self._info = orch.run(
                    self.flux_values,
                    start_idx=start_idx,
                    info=self._info,
                    observer=self,
                    should_stop=self._schedule_stop_event.is_set,
                    pause_requested=self._pause_event.is_set,
                )
                pbar.refresh()
                paused = (
                    self._pause_event.is_set()
                    and not self._user_stop_event.is_set()
                    and orch.run_error is None
                    and self._next_flux_idx < len(self.flux_values)
                )
                return RunSegmentOutcome(
                    info=self._info,
                    run_error=orch.run_error,
                    stopped=self._user_stop_event.is_set(),
                    paused=paused,
                    next_flux_idx=self._next_flux_idx,
                )
            finally:
                self._progress_bar = None
                pbar.close()

    def mark_paused(self) -> None:
        self.store.mark_paused(self._next_flux_idx)
        self._status = RunSessionStatus.PAUSED

    def finalize(self, status: str, *, error: Exception | None = None) -> None:
        self.store.finalize(status, error=error, next_flux_idx=self._next_flux_idx)
        if status == "finished":
            self._status = RunSessionStatus.FINISHED
        elif status == "stopped":
            self._status = RunSessionStatus.STOPPED
        else:
            self._status = RunSessionStatus.FAILED

    def on_point(self, idx: int, flux: float, info: InfoStore) -> None:
        del flux, info
        if self._progress_bar is not None:
            self._progress_bar.set_progress(idx + 1)
        self._next_flux_idx = idx + 1
        self._event_sink.emit_point_done(idx)

    def on_node(self, name: str, idx: int) -> None:
        if name in self._user_node_names:
            self._event_sink.emit_node_entered(name, idx)

    def on_skip(self, name: str, idx: int, reason: SkipReason) -> None:
        if name in self._user_node_names:
            self.store.record_node_skipped(name, idx, reason)

    def on_node_row(self, name: str, idx: int, patch: Any, info: InfoStore) -> None:
        if name not in self._user_node_names:
            return
        self.store.write_node_row(name, idx, patch, info)
        if (
            self._user_node_types.get(name) == "qubit_freq"
            and "qubit_freq" in patch.values()
            and self._has_loaded_predictor
        ):
            self._event_sink.emit_predictor_changed()

    def on_node_failed(self, name: str, idx: int, exc: Exception, stage: str) -> None:
        if name in self._user_node_names:
            self.store.record_node_failed(name, idx, exc, stage)

    def on_flux_committed(self, idx: int, flux: float, info: InfoStore) -> None:
        self.store.commit_flux(idx, flux, info)
