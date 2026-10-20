from __future__ import annotations

import dataclasses
import logging
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
from zcu_tools.gui.session.ports import ExclusionGate, OperationKind
from zcu_tools.gui.session.types import SocCfgHandle, SocHandle

if TYPE_CHECKING:
    from zcu_tools.gui.event_bus import BaseEventBus
    from zcu_tools.gui.session.operation_handles import OperationHandles
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
    thread. Remote connects run via the injected executor.
    """

    def __init__(
        self,
        state: SessionState,
        bus: BaseEventBus,
        gate: ExclusionGate,
        handles: OperationHandles,
        runner: OperationRunner,
    ) -> None:
        self._state = state
        self._bus = bus
        # Connect composes both leaves (ADR-0066): an Exclusion lease
        # (SOC_CONNECT vs run / another connect) + a Handle. A connect has no
        # cancellation point, so cancel_hook=None (§B.3 equivalence).
        self._gate = gate
        self._handles = handles
        self._runner = runner
        # _active_token: POST-BEGIN set, on_terminal cleared (run.py pattern).
        self._active_token: int | None = None
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
        Run or connect conflicts through the SOC_CONNECT gate. Neither the old
        nor new board is closed; their creator owns external resource cleanup.
        """
        self._replace_soc(soc, soccfg)

    def uninstall_soc(self) -> None:
        """Clear the board/configuration and endpoint on the session owner thread.

        Publish SOC_CHANGED with None handles and bump the SoC version, even
        when already empty. Reject Run/connect conflicts. Do not close the
        removed board; resource cleanup remains with its creator.
        """
        self._replace_soc(None, None)

    def _replace_soc(self, soc: SocHandle | None, soccfg: SocCfgHandle | None) -> None:
        self._gate.ensure_can_start(OperationKind.SOC_CONNECT, resource_id=None)
        origin = self._bus.current_origin
        token = self._handles.create(cancel_hook=None, origin=origin)
        self._gate.register(
            token,
            OperationKind.SOC_CONNECT,
            owner_id="soc",
            origin_kind=origin.kind,
            note="replace SoC",
        )
        with self._bus.origin(self._handles.event_origin(token)):
            try:
                self._apply_connection(soc, soccfg, None)
                self._handles.settle(token, OperationOutcome("finished"))
            except Exception as exc:
                self._handles.settle(token, OperationOutcome("failed", str(exc)))
                raise
            finally:
                self._gate.release(token)

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

    def is_connect_active(self) -> bool:
        # _active_token is POST-BEGIN set and cleared in on_terminal (§B.4 option A,
        # mirrors run.py _active_token pattern).
        return self._active_token is not None

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

        # Hold the SOC_CONNECT lease for the whole synchronous connect so it cannot
        # race the GUI's connect button (or an async start_connect). A connect has
        # no cancellation point, so the token is a lease handle only (cancel_hook
        # None); it is settled + released in the finally regardless of outcome.
        self._gate.ensure_can_start(OperationKind.SOC_CONNECT, resource_id=None)
        origin = self._bus.current_origin
        token = self._handles.create(cancel_hook=None, origin=origin)
        self._gate.register(
            token,
            OperationKind.SOC_CONNECT,
            owner_id="soc",
            origin_kind=origin.kind,
            note="connect SoC (remote)",
        )
        self._active_token = token
        with self._bus.origin(self._handles.event_origin(token)):
            try:
                soc, soccfg = self._run_connect_work(req)
                self._apply_connection(soc, soccfg, req)
                self._handles.settle(token, OperationOutcome("finished"))
                return soc, soccfg
            except Exception as exc:
                self._handles.settle(
                    token, OperationOutcome("failed", _format_error(exc))
                )
                raise
            finally:
                self._active_token = None
                self._gate.release(token)

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
            self._active_token = None
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

        token = self._runner.begin(spec)
        # POST-BEGIN: set bookkeeping only after begin() succeeds (never written
        # on conflict, mirrors run.py _active_token pattern).
        self._active_token = token
        return token

    def _apply_connection(
        self,
        soc: SocHandle | None,
        soccfg: SocCfgHandle | None,
        req: ConnectRequest | None,
    ) -> None:
        is_mock = bool(getattr(soc, "is_mock", False))
        logger.info("SoC changed: connected=%s mock=%s", soc is not None, is_mock)
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


def _format_error(exc: Exception) -> str:
    """Classify connection exceptions to user-facing prefixes (§B.5).

    Mirrors the old _ConnectWorker.run error classification so MCP / View error
    messages stay stable after the runner migration.
    """
    if isinstance(exc, (ConnectionRefusedError, TimeoutError, OSError)):
        return f"Connection failed: {exc}"
    return f"Connection error: {exc}"
