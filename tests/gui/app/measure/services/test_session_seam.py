"""Embedded borrowers use the same session owners as experiment controls."""

import pytest
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.services.app_services import build_app_services
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.adapters.manual_owner_scheduler import ManualOwnerScheduler
from zcu_tools.gui.session.adapters.qt_progress_transport import QtProgressTransport
from zcu_tools.gui.session.types import SessionEnv


def test_app_bundle_retains_shared_session(qapp, tmp_path):
    services = build_app_services(
        state=State(SessionEnv(parameters=None, soc=None, soccfg=None)),
        bus=BaseEventBus(),
        registry=Registry(),
        progress_transport=QtProgressTransport(),
        resource_versions=lambda: {},
        render_host=lambda: None,
        owner_scheduler=ManualOwnerScheduler(),
        project_root=str(tmp_path),
    )
    try:
        session = services.session
        assert session.context is services.context
        assert session.device is services.device
        assert session.progress is services.progress
        assert session.soc_control is services.soc_control
        assert session.value_sources.snapshot().values == {}
    finally:
        services.background.quiesce()


def test_controller_exposes_read_only_live_session(qapp, tmp_path):
    controller = Controller(
        State(SessionEnv(parameters=None, soc=None, soccfg=None)),
        Registry(),
        None,
        BaseEventBus(),
        project_root=str(tmp_path),
    )
    try:
        session = controller.session_services
        assert controller.session_services is session
        assert session.context_control is controller.context_control
        assert session.device_control is controller.device_control
        assert session.progress_control is controller.progress_control
        with pytest.raises(AttributeError):
            setattr(controller, "session_services", session)
    finally:
        controller.close()
