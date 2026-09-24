"""Per-editor change-stream routing in RemoteControlAdapter.

Connection lifecycle is covered through NDJSON in
 tests/gui/app/main/services/remote/test_editor_lifecycle.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.main.services.remote import ControlOptions, RemoteControlAdapter
from zcu_tools.gui.app.main.services.remote.service import _ClientCtx
from zcu_tools.gui.remote.rpc_endpoint import ClientLink
from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler


@pytest.fixture(autouse=True)
def _qt(qapp):  # noqa: ARG001 — QtOwnerScheduler is a QObject; needs an app
    yield


def _service():
    ctrl = MagicMock()
    ctrl.get_bus.return_value = None  # disables event-bus subscription wiring
    svc = RemoteControlAdapter(
        controller=ctrl,
        opts=ControlOptions(port=0),
        owner_scheduler=QtOwnerScheduler(),
    )
    return svc, ctrl


def _link() -> ClientLink:
    """A ClientLink with a fresh _ClientCtx attached, as on_client_open would do."""
    link = ClientLink(peer="127.0.0.1:1", token_required=False)
    link.app_ctx = _ClientCtx()
    return link


# ---------------------------------------------------------------------------
# Per-editor change stream routing (_on_editor_event)
# ---------------------------------------------------------------------------


def _enqueued_events(link: ClientLink):
    """Decode every line currently queued on a client's outbound queue."""
    import json

    out = []
    while not link.outbound.empty():
        line = link.outbound.get_nowait()
        out.append(json.loads(line.decode("utf-8")))
    return out


def _register(svc: RemoteControlAdapter, link: ClientLink) -> None:
    """Insert a link into the endpoint's client registry (as accept would)."""
    with svc._endpoint._clients_lock:
        svc._endpoint._clients[object()] = link  # type: ignore[index]


def test_editor_event_only_to_subscribers():
    svc, _ = _service()
    sub = _link()
    sub.app_ctx.subscribed_editors.add("editor-1")  # type: ignore[attr-defined]
    other = _link()
    _register(svc, sub)
    _register(svc, other)

    svc._on_editor_event("editor-1", "editor_changed", lambda: ())

    sub_events = _enqueued_events(sub)
    assert len(sub_events) == 1
    assert sub_events[0]["event"] == "editor_changed"
    assert sub_events[0]["payload"]["editor_id"] == "editor-1"
    # non-subscriber got nothing.
    assert _enqueued_events(other) == []


def test_editor_closed_clears_subscription():
    svc, _ = _service()
    link = _link()
    link.app_ctx.subscribed_editors.add("editor-1")  # type: ignore[attr-defined]
    _register(svc, link)

    svc._on_editor_event("editor-1", "editor_closed", lambda: {"reason": "tab_closed"})

    events = _enqueued_events(link)
    assert events[0]["event"] == "editor_closed"
    assert events[0]["payload"]["reason"] == "tab_closed"
    # subscription auto-dropped after close push.
    assert "editor-1" not in link.app_ctx.subscribed_editors  # type: ignore[attr-defined]


def test_editor_subscribe_handler_updates_state():
    svc, _ = _service()
    link = _link()
    svc._handle_editor_subscribe(link, "1", {"editor_id": "editor-9"}, subscribe=True)
    assert link.app_ctx.subscribed_editors == {"editor-9"}  # type: ignore[attr-defined]
    svc._handle_editor_subscribe(link, "2", {"editor_id": "editor-9"}, subscribe=False)
    assert link.app_ctx.subscribed_editors == set()  # type: ignore[attr-defined]


def test_editor_subscribe_rejects_bad_id():
    from zcu_tools.gui.remote.errors import ErrorCode, RemoteError

    svc, _ = _service()
    link = _link()
    with pytest.raises(RemoteError) as ei:
        svc._handle_editor_subscribe(link, "1", {"editor_id": ""}, subscribe=True)
    assert ei.value.code == ErrorCode.INVALID_PARAMS


def test_editor_event_without_subscriber_does_not_build_payload():
    svc, _ = _service()
    _register(svc, _link())
    payload_factory = MagicMock(return_value={"paths": []})

    svc._on_editor_event("editor-1", "editor_changed", payload_factory)

    payload_factory.assert_not_called()


def test_editor_event_builds_and_encodes_once_for_multiple_subscribers(monkeypatch):
    from zcu_tools.gui.app.main.services.remote import service as service_module

    svc, _ = _service()
    links = [_link(), _link()]
    for link in links:
        link.app_ctx.subscribed_editors.add("editor-1")  # type: ignore[attr-defined]
        _register(svc, link)
    payload_factory = MagicMock(return_value=())
    encode = MagicMock(wraps=service_module.encode_line)
    monkeypatch.setattr(service_module, "encode_line", encode)

    svc._on_editor_event("editor-1", "editor_changed", payload_factory)

    payload_factory.assert_called_once_with()
    encode.assert_called_once()
    assert [_enqueued_events(link)[0]["event"] for link in links] == [
        "editor_changed",
        "editor_changed",
    ]


def test_editor_payload_failure_is_logged_and_does_not_enqueue(caplog):
    svc, _ = _service()
    link = _link()
    link.app_ctx.subscribed_editors.add("editor-1")  # type: ignore[attr-defined]
    _register(svc, link)

    with caplog.at_level("ERROR"):
        svc._on_editor_event(
            "editor-1",
            "editor_changed",
            MagicMock(side_effect=RuntimeError("paths failed")),
        )

    assert _enqueued_events(link) == []
    assert "failed to build editor push editor-1/editor_changed" in caplog.text


def test_failed_editor_closed_payload_keeps_subscription():
    svc, _ = _service()
    link = _link()
    link.app_ctx.subscribed_editors.add("editor-1")  # type: ignore[attr-defined]
    _register(svc, link)

    svc._on_editor_event(
        "editor-1",
        "editor_closed",
        MagicMock(side_effect=RuntimeError("encode input failed")),
    )

    assert "editor-1" in link.app_ctx.subscribed_editors  # type: ignore[attr-defined]
