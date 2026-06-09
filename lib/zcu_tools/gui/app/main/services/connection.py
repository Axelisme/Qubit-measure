from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Tuple, Union

logger = logging.getLogger(__name__)

from qtpy.QtCore import QObject, QThread, QTimer, Signal  # type: ignore[attr-defined]

from zcu_tools.gui.session.events import (
    PredictorChangedPayload,
    SocChangedPayload,
)
from zcu_tools.gui.session.operation_handles import OperationHandles, OperationOutcome
from zcu_tools.gui.session.types import SocCfgHandle, SocHandle
from zcu_tools.simulate.fluxonium.predict import FluxoniumPredictor

from .operation_gate import OperationGate, OperationKind

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.event_bus import EventBus
    from zcu_tools.gui.app.main.state import State


# ---------------------------------------------------------------------------
# Typed requests
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConnectMockRequest:
    """Connect to a local in-process MockSoc."""


@dataclass(frozen=True)
class ConnectRemoteRequest:
    """Connect to a remote ZCU board via SocProxy."""

    ip: str
    port: int


ConnectRequest = Union[ConnectMockRequest, ConnectRemoteRequest]


@dataclass(frozen=True)
class LoadPredictorRequest:
    path: str
    flux_bias: float


@dataclass(frozen=True)
class PredictFreqRequest:
    value: float
    transition: Tuple[int, int]


# ---------------------------------------------------------------------------
# Typed expected failures
# ---------------------------------------------------------------------------


class SoCConnectionError(RuntimeError):
    """Expected failure: SoC connection attempt rejected by user environment."""


class PredictorLoadError(RuntimeError):
    """Expected failure: predictor file could not be loaded / parsed."""


class PredictorNotLoaded(RuntimeError):
    """Expected failure: predict_freq called before any predictor was loaded."""


# ---------------------------------------------------------------------------
# Background connect worker (remote only)
# ---------------------------------------------------------------------------


class _ConnectWorker(QThread):
    """Run remote SoC connect on a background thread; mock connect bypasses this."""

    connected: Signal = Signal(object, object)  # soc, soccfg
    failed: Signal = Signal(str)  # error message

    def __init__(
        self,
        connect_callable: Callable[[], Tuple[SocHandle, SocCfgHandle]],
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._connect_callable = connect_callable
        self._result: Optional[Tuple[SocHandle, SocCfgHandle]] = None
        self._error: Optional[str] = None
        self.finished.connect(self._emit_outcome)

    def run(self) -> None:
        try:
            self._result = self._connect_callable()
        except (ConnectionRefusedError, TimeoutError, OSError) as exc:
            self._error = f"Connection failed: {exc}"
        except Exception as exc:
            self._error = f"Connection error: {exc}"

    def _emit_outcome(self) -> None:
        if self._error is not None:
            self.failed.emit(self._error)
        elif self._result is not None:
            soc, soccfg = self._result
            self.connected.emit(soc, soccfg)
        else:
            raise RuntimeError("Connect worker stopped without outcome")


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class ConnectionService(QObject):
    """Owns SoC connect lifecycle, predictor loading, and frequency prediction.

    Connect work is always reported via Qt signals (connection_finished /
    connection_failed) so the View has a single non-blocking contract. Mock
    connects complete synchronously and dispatch the signal on the next event
    loop tick; remote connects run on a background QThread.

    Predictor load and predict_freq remain synchronous; they raise
    PredictorLoadError / PredictorNotLoaded for user-facing problems.
    """

    connection_finished: Signal = Signal()
    connection_failed: Signal = Signal(str)

    def __init__(
        self,
        state: "State",
        bus: "EventBus",
        gate: OperationGate,
        handles: OperationHandles,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._bus = bus
        # Connect composes both leaves (ADR-0019): an Exclusion lease
        # (SOC_CONNECT vs run / another connect) + a Handle. A connect has no
        # cancellation point, so the handle carries no stop_event.
        self._gate = gate
        self._handles = handles
        self._predictor_path: Optional[str] = None
        self._active_worker: Optional[_ConnectWorker] = None
        self._active_token: Optional[int] = None
        self._pending_is_mock: bool = False

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def has_soc(self) -> bool:
        return self._state.exp_context.soc is not None

    def get_soccfg(self) -> Optional[SocCfgHandle]:
        return self._state.exp_context.soccfg

    def is_mock_soc(self) -> bool:
        """Whether the current connection is the offline mock board.

        Reflects the last connect request's kind; only meaningful while a SoC is
        connected (a successful connect keeps soc set, so pair with has_soc).
        """
        return self._pending_is_mock

    def is_connect_active(self) -> bool:
        return self._active_token is not None

    def get_predictor(self) -> Optional[FluxoniumPredictor]:
        return self._state.exp_context.predictor

    def get_predictor_info(self) -> Optional[dict]:
        predictor = self._state.exp_context.predictor
        if predictor is None:
            return None
        return {"path": self._predictor_path, "flux_bias": predictor.flux_bias}

    # ------------------------------------------------------------------
    # Connect (async-shaped API)
    # ------------------------------------------------------------------

    def start_connect(self, req: ConnectRequest) -> int:
        """Start a SoC connect; outcome always arrives via signals.

        Mock connects complete synchronously and dispatch their signal via
        QTimer.singleShot(0, ...) so the View sees a consistent async flow.
        Remote connects run on a background _ConnectWorker. Returns the
        operation token (handle).
        """
        if not isinstance(req, (ConnectMockRequest, ConnectRemoteRequest)):
            raise TypeError(f"Unsupported connect request: {type(req).__name__}")

        # Symmetric release: settled/freed exactly-once on the terminal path via
        # _release_lease, called from _finish_success / _finish_failure (mock
        # dispatches them synchronously, remote from the worker slot). Fail-fast
        # on conflict before opening the handle (ADR-0019); no stop_event (connect
        # has no cancellation point).
        self._gate.ensure_can_start(OperationKind.SOC_CONNECT)
        token = self._handles.create()
        self._gate.register(token, OperationKind.SOC_CONNECT, owner_id="soc")
        self._active_token = token
        self._pending_is_mock = isinstance(req, ConnectMockRequest)

        if isinstance(req, ConnectMockRequest):
            logger.info("start_connect: mock")
            try:
                from zcu_tools.program.v2.mocksoc import (
                    make_mock_soc,
                    make_mock_soccfg,
                )

                soc = make_mock_soc()
                soccfg = make_mock_soccfg()
            except Exception as exc:
                error = f"Mock SoC initialisation failed: {exc}"
                QTimer.singleShot(0, lambda err=error: self._finish_failure(err))
                return token
            QTimer.singleShot(0, lambda: self._finish_success(soc, soccfg))
            return token

        # ConnectRemoteRequest
        ip = req.ip
        port = req.port
        logger.info("start_connect: remote %s:%d", ip, port)

        def connect_callable() -> Tuple[SocHandle, SocCfgHandle]:
            try:
                from zcu_tools.remote import make_soc_proxy
            except ImportError as exc:
                raise SoCConnectionError(
                    f"Cannot import ZCU client libraries: {exc}. "
                    "Use MockSoc for offline testing."
                ) from exc
            return make_soc_proxy(ip, port)

        try:
            worker = _ConnectWorker(connect_callable, parent=self)
            self._active_worker = worker
            worker.connected.connect(self._on_remote_connected)
            worker.failed.connect(self._on_remote_failed)
            worker.finished.connect(worker.deleteLater)
            worker.start()
        except Exception:
            self._active_worker = None
            self._release_lease(OperationOutcome("failed", "connect failed to start"))
            raise
        return token

    def _on_remote_connected(self, soc: object, soccfg: object) -> None:
        self._active_worker = None
        self._finish_success(soc, soccfg)  # type: ignore[arg-type]

    def _on_remote_failed(self, error: str) -> None:
        self._active_worker = None
        self._finish_failure(error)

    def _finish_success(self, soc: SocHandle, soccfg: SocCfgHandle) -> None:
        try:
            self._apply_connection(soc, soccfg)
        finally:
            self._release_lease(OperationOutcome("finished"))
        self.connection_finished.emit()

    def _finish_failure(self, error: str) -> None:
        self._release_lease(OperationOutcome("failed", error))
        self.connection_failed.emit(error)

    def _release_lease(self, outcome: OperationOutcome) -> None:
        token = self._active_token
        if token is None:
            raise RuntimeError("Connection completed without an active operation token")
        self._active_token = None
        self._handles.settle(token, outcome)
        self._gate.release(token)

    def _apply_connection(self, soc: SocHandle, soccfg: SocCfgHandle) -> None:
        new_ctx = dataclasses.replace(self._state.exp_context, soc=soc, soccfg=soccfg)
        self._state.set_context(new_ctx)
        # soc is its own resource (a run depends on it independently of context);
        # bump it here, on the main thread, where the soc is written. Deliberately
        # NOT bumping ``context``: set_context no longer does so, and a soc connect
        # does not change md/ml — bumping context would spuriously mark md/ml-
        # dependent ops (run / editor.commit / writeback) stale.
        self._state.version.bump("soc")
        self._bus.emit(
            SocChangedPayload(soc=soc, soccfg=soccfg, is_mock=self._pending_is_mock),
        )

    # ------------------------------------------------------------------
    # Predictor (sync)
    # ------------------------------------------------------------------

    def load_predictor(self, req: LoadPredictorRequest) -> None:
        logger.info("load_predictor: path=%r flux_bias=%r", req.path, req.flux_bias)
        try:
            predictor = FluxoniumPredictor.from_file(req.path, flux_bias=req.flux_bias)
        except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
            raise PredictorLoadError(f"Failed to load predictor: {exc}") from exc
        self._predictor_path = req.path
        new_ctx = dataclasses.replace(self._state.exp_context, predictor=predictor)
        self._state.set_context(new_ctx)
        self._bus.emit(PredictorChangedPayload())

    def clear_predictor(self) -> None:
        logger.info("clear_predictor")
        self._predictor_path = None
        new_ctx = dataclasses.replace(self._state.exp_context, predictor=None)
        self._state.set_context(new_ctx)
        self._bus.emit(PredictorChangedPayload())

    def predict_freq(self, req: PredictFreqRequest) -> float:
        predictor = self._state.exp_context.predictor
        if predictor is None:
            raise PredictorNotLoaded("No predictor loaded — load one first")
        return float(predictor.predict_freq(req.value, transition=req.transition))
