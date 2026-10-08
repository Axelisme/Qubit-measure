"""Explicit target capture through the shipped remote tab/app/context seam."""

from collections.abc import Generator, Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest
from zcu_tools.gui.app.measure.adapter import SessionEnv
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.remote import ControlOptions, RemoteControlAdapter
from zcu_tools.gui.app.measure.remote.handlers.tab import h_tab_new
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    EvalValue,
    ScalarSpec,
)
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.session.context_types import EntryRequest
from zcu_tools.resources.entry import RoleSpec

from tests.resources.entry.fakes import register_fakes, registry_state, restore_registry
from zcu_lab.v2.twotone.time_domain.t1.gui import T1Adapter


class TargetAdapter(T1Adapter):
    def parameter_roles(self) -> Mapping[str, RoleSpec]:
        return {"driver": RoleSpec("fake/drive/*")}

    def make_default_cfg(self, ctx: SessionEnv) -> CfgSchema:
        return CfgSchema(
            CfgSectionSpec(fields={"rate": ScalarSpec("Rate", float)}),
            CfgSectionValue(fields={"rate": EvalValue("driver.rate")}),
        )


class RolelessAdapter(TargetAdapter):
    def parameter_roles(self) -> Mapping[str, RoleSpec]:
        return {}


@dataclass
class TargetApplication:
    state: State
    controller: Controller
    remote: RemoteControlAdapter


@pytest.fixture
def application(qapp, tmp_path: Path) -> Generator[TargetApplication]:
    before = registry_state()
    try:
        register_fakes()
        installed = registry_state()
        registry = Registry()
        registry.register("targeted", TargetAdapter)
        registry.register("roleless", RolelessAdapter)
        state = State(SessionEnv(parameters=None, soc=None, soccfg=None))
        controller = Controller(
            state, registry, None, BaseEventBus(), project_root=str(tmp_path)
        )
        controller.setup_control.new_entry(EntryRequest("entry"))
        context = controller.context_control
        context.add_component("Q1", "fake/drive/a", {"rate": 10.0})
        context.add_component("Q2", "fake/drive/a", {"rate": 20.0})
        context.add_component("S1", "fake/sensor", {"rate": 30.0})
        context.set_component_focus("Q1")
        remote = RemoteControlAdapter(
            controller, ControlOptions(port=0), controller.owner_scheduler
        )
        yield TargetApplication(state, controller, remote)
        for tab_id in tuple(state.list_tab_ids()):
            controller.tab_control.close_tab(tab_id)
        assert registry_state() == installed
    finally:
        restore_registry(before)


def rate(application: TargetApplication, tab_id: str) -> object:
    cfg = application.state.get_tab(tab_id).cfg
    return cfg.accept(cfg.observe().ref.revision).values["rate"]


def test_explicit_target_uses_other_component_without_changing_focus(
    application: TargetApplication,
) -> None:
    receipt = h_tab_new(
        application.remote, {"adapter_name": "targeted", "target": "Q2"}
    )
    tab_id = receipt["tab_id"]
    assert isinstance(tab_id, str)
    assert rate(application, tab_id) == 20.0
    parameters = application.state.get_tab(tab_id).parameters
    assert parameters is not None and parameters.roles is not None
    assert dict(parameters.roles.names) == {"driver": "Q2"}
    assert application.controller.context_control.get_component_focus().name == "Q1"


def test_omitted_target_captures_focus_and_later_focus_only_affects_new_tabs(
    application: TargetApplication,
) -> None:
    one = h_tab_new(application.remote, {"adapter_name": "targeted"})["tab_id"]
    assert isinstance(one, str)
    application.controller.context_control.set_component_focus("Q2")
    two = h_tab_new(application.remote, {"adapter_name": "targeted"})["tab_id"]
    assert isinstance(two, str)
    context = application.controller.context_control
    context.set_parameter(context.read_context().target, "Q1.rate", 11.0)
    assert rate(application, one) == 10.0
    assert rate(application, two) == 20.0


@pytest.mark.parametrize("target", ["missing", "S1"])
def test_unresolvable_target_preserves_typed_details_and_registers_no_tab(
    application: TargetApplication, target: str
) -> None:
    with pytest.raises(RemoteError) as caught:
        h_tab_new(application.remote, {"adapter_name": "targeted", "target": target})
    error = caught.value
    assert error.code is ErrorCode.PRECONDITION_FAILED
    assert error.reason == "role_resolution_failed"
    assert error.data is not None
    assert error.data["role"] == "driver"
    assert error.data["focus"] == target
    assert error.data["required_kind"] == "fake/drive/*"
    assert error.data["detail"]
    assert application.state.list_tab_ids() == []
    assert application.state.active_tab_id is None
    assert application.controller.context_control.get_component_focus().name == "Q1"


@pytest.mark.parametrize("target", ["", " ", 7])
def test_invalid_target_is_rejected_before_creation(
    application: TargetApplication, target: object
) -> None:
    with pytest.raises(RemoteError) as caught:
        h_tab_new(application.remote, {"adapter_name": "targeted", "target": target})
    assert caught.value.code is ErrorCode.INVALID_PARAMS
    assert caught.value.reason == "invalid_target"
    assert application.state.list_tab_ids() == []


@pytest.mark.parametrize("target", ["", " "])
def test_direct_tab_target_rejects_empty_names(
    application: TargetApplication, target: str
) -> None:
    from zcu_tools.gui.expected_error import InvalidInputError

    with pytest.raises(InvalidInputError):
        application.controller.tab_control.new_tab("targeted", target=target)
    assert application.state.list_tab_ids() == []


def test_roleless_adapter_does_not_silently_ignore_explicit_target(
    application: TargetApplication,
) -> None:
    from zcu_tools.gui.expected_error import InvalidInputError

    with pytest.raises(InvalidInputError):
        application.controller.tab_control.new_tab("roleless", target="Q1")
    assert application.state.list_tab_ids() == []
