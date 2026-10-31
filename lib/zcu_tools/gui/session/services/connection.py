from __future__ import annotations

from zcu_tools.gui.session.ports import BackgroundError

import dataclasses
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from zcu_tools.gui.session.events import ConnectionFinishedPayload, SocChangedPayload
from zcu_tools.gui.session.operation_handles import OperationOutcome
from zcu_tools.gui.session.operation_runner import (
    BgResult,
    ExclusionRequest,
    OperationSpec,
    SettleFn,
)
from zcu_tools.gui.session.ports import OperationKind
from zcu_tools.gui.session.types import SocCfgHandle, SocHandle

if TYPE_CHECKING:
    from zcu_tools.gui.event_bus import BaseEventBus
    from zcu_tools.gui.session.operation_runner import OperationRunner
    from zcu_tools.gui.session.state import SessionState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed requests
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConnectRemoteRequest:
    """Remote endpoint: ip is the board host/address, port its TCP service port."""

    ip: str
    port: int


ConnectRequest = ConnectRemoteRequest


# ---------------------------------------------------------------------------
# Typed expected failures
# ---------------------------------------------------------------------------


class SoCConnectionError(RuntimeError):
    """Expected failure: SoC connection attempt rejected by user environment."""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class SoCConnectionService:
    """Owns SoC connect lifecycle, delegating async work to OperationRunner.

    Async terminal outcomes are published as typed EventBus facts on the owner
    thread. Remote connects run via the injected executor. A board this service
    connected through make_soc_proxy is released when it is replaced or
    uninstalled; boards installed by callers stay with their creators.
    """

    def __init__(
        self,
        state: SessionState,
        bus: BaseEventBus,
        runner: OperationRunner,
    ) -> None:
        self._state = state
        self._bus = bus
        # The runner owns exclusion and handles for both inline and async work.
        # Connect has no cancellation point, so every spec uses cancel_hook=None.
        self._runner = runner
        self._connected_request: ConnectRequest | None = None

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def has_soc(self) -> bool:
        return self._state.session_env.soc is not None

    def get_soccfg(self) -> SocCfgHandle | None:
        return self._state.session_env.soccfg

    def is_mock_soc(self) -> bool:
        """Forward the installed board's optional ``is_mock`` flag, else False."""
        return bool(getattr(self._state.session_env.soc, "is_mock", False))

    def install_soc(self, soc: SocHandle, soccfg: SocCfgHandle) -> None:
        """Publish an already constructed board and its QICK configuration.

        Call on the session owner thread. Replace any installed pair, clear the
        remote endpoint, bump the SoC version and publish SOC_CHANGED. Reject
        Run or connect conflicts through the SOC_CONNECT gate. The new board is
        never closed by this service; its creator owns cleanup. An old board this
        service connected is released after the new pair is published.
        """
        self._replace_soc(soc, soccfg)

    def uninstall_soc(self) -> None:
        """Clear the board/configuration and endpoint on the session owner thread.

        Publish SOC_CHANGED with None handles and bump the SoC version, even
        when already empty. Reject Run/connect conflicts. A removed board this
        service connected is released after publication, with INFO log lines
        before and after; a caller-installed board stays with its creator.
        """
        self._replace_soc(None, None)

    def _replace_soc(self, soc: SocHandle | None, soccfg: SocCfgHandle | None) -> None:
        spec = OperationSpec(
            exclusion=ExclusionRequest(
                kind=OperationKind.SOC_CONNECT,
                owner_id="soc",
                note="replace SoC",
            ),
            owner_id="soc",
            wants_progress=False,
            cancel_hook=None,
        )
        with self._runner.inline(spec):
            self._apply_connection(soc, soccfg, None)

    def current_request(self) -> ConnectRemoteRequest | None:
        """Read the last successful remote endpoint on-owner, without network I/O.

        None means no board or a caller-installed board without a remote
        endpoint. This detached value is not a connection preference or a file
        format; failed connects preserve the previous successful endpoint.
        """
        return self._connected_request

    def connected_endpoint(self) -> dict[str, str | int | None]:
        """The last successful remote address, including GUI-initiated connects."""
        if isinstance(self._connected_request, ConnectRemoteRequest):
            return {
                "address": self._connected_request.ip,
                "port": self._connected_request.port,
            }
        return {"address": None, "port": None}

    # ------------------------------------------------------------------
    # Connect (async-shaped API, OperationRunner client)
    # ------------------------------------------------------------------

    @staticmethod
    def _run_connect_work(req: ConnectRequest) -> tuple[SocHandle, SocCfgHandle]:
        """Do the actual SoC connect, returning the (soc, soccfg) handles.

        The single source of truth for "how a connect is performed", shared by the
        async ``start_connect`` (off-main via the runner) and the synchronous
        ``connect_sync`` (on-main via the wire). It does only the connect itself —
        no State writes, no version bump, no event emit (those are the caller's job
        via ``_apply_connection``), so neither caller's threading model leaks in.
        """
        # remote
        try:
            from zcu_tools.qick_remote import make_soc_proxy
        except ImportError as exc:
            raise SoCConnectionError(
                f"Cannot import ZCU client libraries: {exc}. "
                "Use a startup Python environment for offline acquisition."
            ) from exc
        return make_soc_proxy(req.ip, req.port)

    def connect_sync(self, req: ConnectRequest) -> tuple[SocHandle, SocCfgHandle]:
        """Connect synchronously on the calling (Qt main) thread.

        The wire path (``soc.connect``) wants a single blocking call that returns
        the connected SoC directly, with all post-connect side effects applied
        before it returns — so it runs on-main and the IO worker blocks on it. It
        still holds the same SOC_CONNECT exclusion lease as the async path (so the
        wire connect and the GUI's connect button cannot race), minting a token
        purely as the lease handle (no async await: the work runs inline here, not
        on a bg thread). On success it calls the shared ``_apply_connection`` — the
        identical State write and soc version bump as the async path. This low-level
        connection does not provision devices or a simulated environment.
        """
        if type(req) is not ConnectRemoteRequest:
            raise TypeError(f"Unsupported connect request: {type(req).__name__}")

        spec = OperationSpec(
            exclusion=ExclusionRequest(
                kind=OperationKind.SOC_CONNECT,
                owner_id="soc",
                note="connect SoC (remote)",
            ),
            owner_id="soc",
            wants_progress=False,
            cancel_hook=None,
        )
        with self._runner.inline(spec) as lease:
            try:
                soc, soccfg = self._run_connect_work(req)
                self._apply_connection(soc, soccfg, req)
            except Exception as exc:
                # Keep connection error classification as domain policy; the
                # runner still owns settlement and release failure isolation.
                lease.settle(OperationOutcome("failed", _format_error(exc)))
                raise
            return soc, soccfg

    def start_connect(self, req: ConnectRequest) -> int:
        """Start a SoC connect; outcome arrives as ``ConnectionFinishedPayload``.

        Delegates to OperationRunner with cancel_hook=None (connect has no
        cancellation point — §B.3 equivalence with old _ConnectWorker). Returns
        the operation token minted by runner.begin.
        """
        if type(req) is not ConnectRemoteRequest:
            raise TypeError(f"Unsupported connect request: {type(req).__name__}")

        def work(_factory: object) -> tuple[SocHandle, SocCfgHandle]:
            # factory is None (wants_progress=False). Both branches run off-main via
            # the executor and publish one typed completion fact on the owner thread.
            return self._run_connect_work(req)

        def on_terminal(bg: BgResult, settle: SettleFn) -> None:
            # Runs on main thread (bg marshal). _apply_connection writes State and
            # calls version.bump("soc") — main-thread invariant preserved (§B.2).
            if bg.ok:
                soc, soccfg = bg.result
                self._apply_connection(soc, soccfg, req)
                settle(OperationOutcome("finished"))
                self._bus.emit(
                    ConnectionFinishedPayload(success=True, is_mock=self.is_mock_soc())
                )
            else:
                assert bg.error is not None
                msg = _format_error(bg.error)
                settle(OperationOutcome("failed", msg))
                self._bus.emit(
                    ConnectionFinishedPayload(
                        success=False,
                        is_mock=self.is_mock_soc(),
                        error_message=msg,
                    )
                )

        spec = OperationSpec(
            exclusion=ExclusionRequest(
                kind=OperationKind.SOC_CONNECT,
                owner_id="soc",
                note="connect SoC (remote)",
            ),
            owner_id="soc",
            wants_progress=False,
            cancel_hook=None,
            work=work,
            run_in_pool=False,  # dedicated thread, matching old _ConnectWorker
            on_terminal=on_terminal,
        )

        return self._runner.begin(spec)

    def _apply_connection(
        self,
        soc: SocHandle | None,
        soccfg: SocCfgHandle | None,
        req: ConnectRequest | None,
    ) -> None:
        is_mock = bool(getattr(soc, "is_mock", False))
        logger.info("SoC changed: connected=%s mock=%s", soc is not None, is_mock)
        previous = self._state.session_env.soc
        previous_request = self._connected_request
        new_ctx = dataclasses.replace(self._state.session_env, soc=soc, soccfg=soccfg)
        self._state.set_context(new_ctx)
        self._connected_request = req
        # soc is its own resource (a run depends on it independently of context);
        # bump it here, on the main thread, where the soc is written. Deliberately
        # NOT bumping ``context``: set_context no longer does so, and a soc connect
        # does not change md/ml — bumping context would spuriously mark md/ml-
        # dependent ops (run / editor.commit / writeback) stale.
        self._state.version.bump("soc")
        self._bus.emit(
            SocChangedPayload(soc=soc, soccfg=soccfg, is_mock=is_mock),
        )
        # Release after publication so no subscriber still holds the old board.
        if (
            previous_request is not None
            and previous is not None
            and previous is not soc
        ):
            _release_connected(previous, previous_request)


def _release_connected(soc: SocHandle, req: ConnectRemoteRequest) -> None:
    """Close a board connection this service created, logging start and duration.

    A failed close is logged with its traceback and does not stop the caller's
    replacement or GUI teardown; the board is already unpublished.
    """
    from zcu_tools.qick_remote import release_soc_proxy

    logger.info("Releasing SoC connection %s:%s", req.ip, req.port)
    started = time.monotonic()
    try:
        release_soc_proxy(soc)
    except Exception:
        logger.exception("Releasing SoC connection %s:%s failed", req.ip, req.port)
        return
    elapsed_ms = (time.monotonic() - started) * 1000
    logger.info(
        "Released SoC connection %s:%s in %.0f ms", req.ip, req.port, elapsed_ms
    )


def _format_error(exc: BackgroundError) -> str:
    """Classify connection exceptions to user-facing prefixes (§B.5).

    Mirrors the old _ConnectWorker.run error classification so MCP / View error
    messages stay stable after the runner migration.
    """
    if isinstance(exc, (ConnectionRefusedError, TimeoutError, OSError)):
        return f"Connection failed: {exc}"
    return f"Connection error: {exc}"
