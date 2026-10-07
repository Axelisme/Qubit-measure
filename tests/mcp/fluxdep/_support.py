"""Recording native transport for public Fluxdep MCP contracts."""

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field

from zcu_tools.mcp.core.bridge import DeliverFn, OnClosedFn
from zcu_tools.mcp.fluxdep.assembly import FluxdepServer


@dataclass(frozen=True)
class GuiError:
    """Invocation failure's native code, diagnostic and optional reason."""

    code: str
    message: str
    reason: str | None = None


@dataclass(frozen=True)
class GuiResponse:
    """One queued native result, or one invocation error (never both)."""

    result: Mapping[str, object] | None = None
    error: GuiError | None = None


@dataclass
class RecordingTransport:
    """Synchronous raw-wire adapter; requests are detached complete envelopes.

    replies holds caller-supplied GUI outcomes. failure models transport failure
    after a request was admitted. close_count records lifecycle cleanup, not GUI
    process shutdown. The real bridge owns request IDs, correlation and waits.
    """

    replies: deque[GuiResponse] = field(default_factory=deque)
    requests: list[dict[str, object]] = field(default_factory=list)
    deliver_reply: DeliverFn | None = None
    is_open: bool = True
    close_count: int = 0
    failure: Exception | None = None

    def attach(
        self, deliver_reply: DeliverFn, deliver_event: DeliverFn, on_closed: OnClosedFn
    ) -> None:
        """Store the bridge's reply callback; this fake emits no events."""
        self.deliver_reply = deliver_reply

    def send_line(self, payload: dict[str, object]) -> None:
        """Record one request, then deliver the queued result or fail transport."""
        self.requests.append(dict(payload))
        if self.failure is not None:
            raise self.failure
        if self.deliver_reply is None or not self.replies:
            raise AssertionError("No attached callback or queued GUI response")
        reply = self.replies.popleft()
        if reply.error is not None:
            error = reply.error
            self.deliver_reply(
                {
                    "id": payload["id"],
                    "ok": False,
                    "error": {
                        "code": error.code,
                        "message": error.message,
                        "reason": error.reason,
                    },
                }
            )
        else:
            if reply.result is None:
                raise AssertionError("A successful GUI response needs a result")
            self.deliver_reply(
                {"id": payload["id"], "ok": True, "result": reply.result}
            )

    def close(self) -> None:
        """Close the transport without invoking GUI lifecycle commands."""
        self.is_open = False
        self.close_count += 1


@dataclass(frozen=True)
class Client:
    """One fresh assembled server and its recording native transport."""

    server: FluxdepServer
    transport: RecordingTransport
