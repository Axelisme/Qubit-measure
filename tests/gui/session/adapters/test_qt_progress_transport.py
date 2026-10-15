"""QtProgressTransport marshals worker-thread emits to the main thread."""

from __future__ import annotations

import pytest

from tests.gui._lifecycle_waits import running
from tests.gui._qt_waits import qt_until

pytest.importorskip("qtpy")


def test_emit_from_worker_thread_delivers_on_main_thread(qapp):
    import threading

    from zcu_tools.gui.session.adapters.qt_progress_transport import (
        QtProgressTransport,
    )
    from zcu_tools.gui.session.ports import ProgressEvent, ProgressEventKind

    transport = QtProgressTransport()  # built on the main (test) thread
    main_thread = threading.current_thread()
    received: list[tuple[ProgressEvent, threading.Thread]] = []

    transport.set_receiver(lambda ev: received.append((ev, threading.current_thread())))

    def emit_updates() -> None:
        for i in range(3):
            transport.emit(ProgressEvent(1, 0, ProgressEventKind.UPDATE, n=i))

    with running(emit_updates):
        qt_until(lambda: len(received) == 3)

    assert [ev.n for ev, _ in received] == [0, 1, 2]  # order preserved
    assert all(thr is main_thread for _, thr in received)  # delivered on main
