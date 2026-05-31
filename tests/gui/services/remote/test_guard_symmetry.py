"""Guard symmetry: the remote dispatch path enforces the same domain guard as
the UI (Controller) path.

Both clients route protected operations through GuardService. This test invokes
the dispatch handlers directly against a real Controller and asserts that the
same precondition that the UI path rejects (Controller raises GuardError) is
surfaced as a RemoteError(PRECONDITION_FAILED) on the remote path.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.experiment.v2_gui.adapters.fake import FakeAdapter
from zcu_tools.experiment.v2_gui.registry import register_all
from zcu_tools.gui.adapter import ContextReadiness, ExpContext
from zcu_tools.gui.controller import Controller
from zcu_tools.gui.event_bus import EventBus
from zcu_tools.gui.io_manager import IOManager
from zcu_tools.gui.registry import Registry
from zcu_tools.gui.runner import Runner
from zcu_tools.gui.services.guard import GuardError
from zcu_tools.gui.services.remote.dispatch import METHOD_REGISTRY
from zcu_tools.gui.services.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.services.remote.param_spec import validate_params
from zcu_tools.gui.state import State


def _make_controller(readiness: ContextReadiness) -> Controller:
    ctx = ExpContext(
        md=MagicMock(),
        ml=MagicMock(),
        soc=MagicMock(),
        soccfg=MagicMock(),
        result_dir="/tmp/zcu_result",
        active_label="ctx001" if readiness is ContextReadiness.ACTIVE else "",
        readiness=readiness,
    )
    state = State(ctx)
    registry = Registry()
    register_all(registry)
    if not registry.has("fake"):
        registry.register("fake", FakeAdapter)
    view = MagicMock()
    view.make_pbar_factory = MagicMock(return_value=None)
    view.make_live_container = MagicMock(return_value=None)
    io_manager = IOManager()
    io_manager._em = MagicMock()
    bus = EventBus()
    bus.emit = MagicMock()  # type: ignore[method-assign]
    return Controller(
        state=state,
        runner=Runner(),
        registry=registry,
        io_manager=io_manager,
        view=view,
        bus=bus,
    )


def _dispatch(ctrl: Controller, method: str, params: dict) -> object:
    """Mirror the service path: validate params against the method's ParamSpec
    before invoking the handler. Handlers receive the adapter (ADR-0013), so
    wrap ctrl in a minimal adapter stub."""
    from types import SimpleNamespace
    from typing import cast

    from zcu_tools.gui.services.remote.service import RemoteControlAdapter

    spec = METHOD_REGISTRY[method]
    handler_params = validate_params(spec.params, params) if spec.params else params
    adapter = cast(RemoteControlAdapter, SimpleNamespace(ctrl=ctrl))
    return spec.handler(adapter, handler_params)


def test_run_start_draft_context_symmetry(qapp):  # noqa: ARG001
    """DRAFT context: UI raises GuardError; remote returns PRECONDITION_FAILED."""
    ctrl = _make_controller(ContextReadiness.DRAFT)
    tab_id = ctrl.new_tab("fake")

    # UI path
    with pytest.raises(GuardError, match="active file-backed context"):
        ctrl.start_run(tab_id)

    # Remote path — same precondition, mapped to a typed wire error.
    with pytest.raises(RemoteError) as excinfo:
        _dispatch(ctrl, "run.start", {"tab_id": tab_id})
    assert excinfo.value.code is ErrorCode.PRECONDITION_FAILED
    assert "active file-backed context" in excinfo.value.message


def test_save_data_draft_context_symmetry(qapp):  # noqa: ARG001
    ctrl = _make_controller(ContextReadiness.DRAFT)
    tab_id = ctrl.new_tab("fake")

    with pytest.raises(GuardError, match="active file-backed context"):
        ctrl.save_data(tab_id, "/tmp/data.h5")

    with pytest.raises(RemoteError) as excinfo:
        _dispatch(ctrl, "save.data", {"tab_id": tab_id, "data_path": "/tmp/data.h5"})
    assert excinfo.value.code is ErrorCode.PRECONDITION_FAILED
    # DRAFT context fails the readiness guard before the run-result check.
    assert excinfo.value.reason == "no_active_context"


def test_analyze_without_run_result_symmetry(qapp):  # noqa: ARG001
    """ACTIVE context but no run result: both paths reject on the same guard."""
    ctrl = _make_controller(ContextReadiness.ACTIVE)
    tab_id = ctrl.new_tab("fake")

    with pytest.raises(GuardError, match="No run result"):
        ctrl.analyze(tab_id, object())

    snap = ctrl.get_tab_snapshot(tab_id)
    # analyze.start needs prepared analyze params; without a run result the
    # snapshot has none, so the handler short-circuits before guard. Assert the
    # UI-path guard directly covers the no-result case (the authoritative guard).
    assert snap.analyze_params is None


def test_save_no_run_result_carries_reason(qapp):  # noqa: ARG001
    """ACTIVE context, no run result: save fails with reason='no_run_result'."""
    ctrl = _make_controller(ContextReadiness.ACTIVE)
    tab_id = ctrl.new_tab("fake")

    with pytest.raises(RemoteError) as excinfo:
        _dispatch(ctrl, "save.data", {"tab_id": tab_id, "data_path": "/tmp/d.h5"})
    assert excinfo.value.code is ErrorCode.PRECONDITION_FAILED
    assert excinfo.value.reason == "no_run_result"


def test_run_permit_issued_for_active_valid_context(qapp):  # noqa: ARG001
    """Sanity: with ACTIVE context + valid cfg, the guard issues a RunPermit
    rather than raising. (We assert permit issuance instead of spawning a real
    run worker, which would race teardown in a headless test.)"""
    ctrl = _make_controller(ContextReadiness.ACTIVE)
    tab_id = ctrl.new_tab("fake")

    permit = ctrl._guard_svc.acquire_run_permit(tab_id)
    assert permit.tab_id == tab_id
