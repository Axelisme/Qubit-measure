"""Explicit target capture through the shipped remote tab/app/context seam."""

from collections.abc import Generator, Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest
from zcu_tools.cfg_model import ConstWaveformCfg, PulseSpec
from zcu_tools.gui.app.measure.adapter import SessionEnv
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.remote import ControlOptions, RemoteControlAdapter
from zcu_tools.gui.app.measure.remote.handlers.tab import (
    h_tab_new,
    h_tab_parameter_read,
)
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    ScalarSpec,
)
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.session.context_types import EntryRequest
from zcu_tools.resources.entry import RoleSpec, component_registry

from tests.resources.entry.fakes import (
    Driver,
    register_fakes,
    registry_state,
    restore_registry,
)
from zcu_lab.v2.twotone.time_domain.t1.gui import T1Adapter


class LibraryDriver(Driver):
    module: dict[str, str]


class TargetAdapter(T1Adapter):
    def parameter_roles(self) -> Mapping[str, RoleSpec]:
        return {"driver": RoleSpec("fake/drive/*")}

    def make_default_cfg(self, ctx: SessionEnv) -> CfgSchema:
        return CfgSchema(
            CfgSectionSpec(fields={"rate": ScalarSpec("Rate", float)}),
            CfgSectionValue(fields={"rate": EvalValue("driver.rate")}),
        )


class DependentTargetAdapter(TargetAdapter):
    def parameter_roles(self) -> Mapping[str, RoleSpec]:
        return {
            "driver": RoleSpec("fake/drive/*"),
            "target": RoleSpec("fake/drive/*", via="driver.source"),
        }

    def make_default_cfg(self, ctx: SessionEnv) -> CfgSchema:
        return CfgSchema(
            CfgSectionSpec(
                fields={
                    "rate": ScalarSpec("Rate", float),
                    "peer_rate": ScalarSpec("Peer rate", float),
                }
            ),
            CfgSectionValue(
                fields={
                    "rate": EvalValue("driver.rate"),
                    "peer_rate": EvalValue("target.rate"),
                }
            ),
        )


class RolelessAdapter(TargetAdapter):
    def parameter_roles(self) -> Mapping[str, RoleSpec]:
        return {}

    def make_default_cfg(self, ctx: SessionEnv) -> CfgSchema:
        return CfgSchema(
            CfgSectionSpec(fields={"rate": ScalarSpec("Rate", float)}),
            CfgSectionValue(fields={"rate": DirectValue(1.0)}),
        )


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
        component_registry.register("fake/drive/library", LibraryDriver)
        installed = registry_state()
        registry = Registry()
        registry.register("targeted", TargetAdapter)
        registry.register("dependent", DependentTargetAdapter)
        registry.register("roleless", RolelessAdapter)
        state = State(SessionEnv(parameters=None, soc=None, soccfg=None))
        controller = Controller(
            state, registry, None, BaseEventBus(), project_root=str(tmp_path)
        )
        controller.setup_control.new_entry(EntryRequest("entry"))
        context = controller.context_control
        context.add_component("Q1", "fake/drive/a", {"rate": 10.0})
        context.add_component(
            "Q2", "fake/drive/a", {"rate": 20.0, "source": "Q1", "energy_a": None}
        )
        context.add_component("S1", "fake/sensor", {"rate": 30.0})
        context.add_component(
            "QL",
            "fake/drive/library",
            {"rate": 40.0, "module": {"pi": "driver.pulses.pi"}},
        )
        context.set_module(
            context.read_context().target,
            "QL.pulses.pi",
            PulseSpec(
                ch=0,
                nqz=1,
                freq=100.0,
                gain=0.25,
                waveform=ConstWaveformCfg(length=0.05),
            ),
        )
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


def test_explicit_target_consumes_focus_and_resolves_same_kind_peer_via(
    application: TargetApplication,
) -> None:
    tab_id = h_tab_new(
        application.remote, {"adapter_name": "dependent", "target": "Q2"}
    )["tab_id"]
    assert isinstance(tab_id, str)
    tab = application.state.get_tab(tab_id)
    assert tab.parameters is not None and tab.parameters.roles is not None
    assert dict(tab.parameters.roles.names) == {"driver": "Q2", "target": "Q1"}
    cfg = tab.cfg
    accepted = cfg.accept(cfg.observe().ref.revision)
    assert accepted.values == {"rate": 20.0, "peer_rate": 10.0}
    assert application.controller.context_control.get_component_focus().name == "Q1"


def test_explicit_target_does_not_hide_missing_same_kind_peer_reference(
    application: TargetApplication,
) -> None:
    with pytest.raises(RemoteError) as caught:
        h_tab_new(application.remote, {"adapter_name": "dependent", "target": "Q1"})
    error = caught.value
    assert error.code is ErrorCode.PRECONDITION_FAILED
    assert error.reason == "role_resolution_failed"
    assert error.data is not None
    assert error.data["role"] == "target"
    assert error.data["focus"] == "Q1"
    assert error.data["required_kind"] == "fake/drive/*"
    assert error.data["detail"]
    assert application.state.list_tab_ids() == []
    assert application.state.active_tab_id is None
    assert application.controller.context_control.get_component_focus().name == "Q1"


def test_parameter_read_uses_fixed_tab_roles_and_distinguishes_missing_null(
    application: TargetApplication,
) -> None:
    tab_id = h_tab_new(
        application.remote, {"adapter_name": "targeted", "target": "Q2"}
    )["tab_id"]
    assert isinstance(tab_id, str)
    context = application.controller.context_control
    context.set_parameter(context.read_context().target, "Q2.rate", 25.0)
    context.set_component_focus("Q1")
    sources = h_tab_parameter_read(
        application.remote,
        {
            "tab_id": tab_id,
            "paths": ["driver.rate", "driver.energy_a", "driver.not_supplied"],
            "module_refs": [],
        },
    )
    assert sources == {
        "values": {"driver.rate": 20.0, "driver.energy_a": None},
        "modules": {},
    }
    assert context.get_component_focus().name == "Q1"


def test_parameter_read_resolves_library_role_slots_in_original_capture(
    application: TargetApplication,
) -> None:
    tab_id = h_tab_new(
        application.remote, {"adapter_name": "targeted", "target": "QL"}
    )["tab_id"]
    context = application.controller.context_control
    context.set_parameter(
        context.read_context().target, "QL.module.pi", "QL.pulses.missing"
    )
    context.set_module(
        context.read_context().target,
        "QL.pulses.later",
        PulseSpec(
            ch=0,
            nqz=1,
            freq=200.0,
            gain=0.25,
            waveform=ConstWaveformCfg(length=0.05),
        ),
    )
    sources = h_tab_parameter_read(
        application.remote,
        {
            "tab_id": tab_id,
            "paths": [],
            "module_refs": [
                "driver.module.pi",
                "driver.pulses.pi",
                "QL.pulses.pi",
                "driver.pulses.later",
                "<Custom:pulse>",
            ],
        },
    )
    assert sources == {
        "values": {},
        "modules": {
            "driver.module.pi": "QL.pulses.pi",
            "driver.pulses.pi": "QL.pulses.pi",
            "QL.pulses.pi": "QL.pulses.pi",
            "driver.pulses.later": None,
            "<Custom:pulse>": None,
        },
    }


def test_parameter_read_rejects_unbound_role_instead_of_live_focus_fallback(
    application: TargetApplication,
) -> None:
    from zcu_tools.resources.entry.errors import RoleResolutionError

    tab_id = h_tab_new(application.remote, {"adapter_name": "targeted"})["tab_id"]
    with pytest.raises(RoleResolutionError):
        h_tab_parameter_read(
            application.remote,
            {"tab_id": tab_id, "paths": ["qubit.freq"], "module_refs": []},
        )


def test_parameter_read_source_free_tab_accepts_only_empty_queries(
    application: TargetApplication,
) -> None:
    tab_id = h_tab_new(application.remote, {"adapter_name": "roleless"})["tab_id"]
    assert h_tab_parameter_read(
        application.remote, {"tab_id": tab_id, "paths": [], "module_refs": []}
    ) == {"values": {}, "modules": {}}
    for requested in (
        {"paths": ["driver.rate"]},
        {"module_refs": ["driver.pulses.pi"]},
    ):
        with pytest.raises(RemoteError) as caught:
            h_tab_parameter_read(
                application.remote,
                {"tab_id": tab_id, "paths": [], "module_refs": [], **requested},
            )
        assert caught.value.code is ErrorCode.PRECONDITION_FAILED


@pytest.mark.parametrize("field", ["paths", "module_refs"])
@pytest.mark.parametrize("invalid", [None, "driver.rate", [""], ["  "], [False], [12]])
def test_parameter_read_rejects_invalid_query_arrays(
    application: TargetApplication, field: str, invalid: object
) -> None:
    tab_id = h_tab_new(application.remote, {"adapter_name": "targeted"})["tab_id"]
    with pytest.raises(RemoteError) as caught:
        h_tab_parameter_read(
            application.remote,
            {"tab_id": tab_id, "paths": [], "module_refs": [], field: invalid},
        )
    assert caught.value.code is ErrorCode.INVALID_PARAMS


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
