"""Explicit target capture through the shipped remote tab/app/context seam."""

from collections.abc import Generator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pytest
from qtpy.QtCore import QEventLoop, QTimer
from zcu_tools.cfg_model import ConstWaveformCfg, PulseSpec
from zcu_tools.experiment import ExpCfgModel
from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AdapterCfgDefinition,
    AnalysisMode,
    RunRequest,
    SessionEnv,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.events.tab import (
    TabContentChangedPayload,
    TabContentFact,
)
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.remote import ControlOptions, RemoteControlAdapter
from zcu_tools.gui.app.measure.remote.handlers.tab import (
    h_tab_new,
    h_tab_parameter_read,
)
from zcu_tools.gui.app.measure.services.guard import GuardService
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    ScalarSpec,
)
from zcu_tools.gui.cfg.resource import (
    CfgEdit,
    CfgPreconditionError,
    CfgPreconditionReason,
    CfgStaleError,
    CfgStatus,
)
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
)
from zcu_tools.resources.entry import RoleResolutionError, RoleSpec, component_registry

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


class UnusedAliasAdapter(RolelessAdapter):
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        requires_soc=False
    )

    def parameter_roles(self) -> Mapping[str, RoleSpec]:
        return {"driver": RoleSpec("fake/drive/*")}


class RebindAdapter(
    BaseAdapter[ExpCfgModel, RunRecord[ExpCfgModel, dict[str, ExperimentCfgValue]]]
):
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        requires_soc=False, analysis=AnalysisMode.NONE, load_data=False
    )

    experiment_tag: ClassVar[str | None] = "test/rebind"

    @classmethod
    def cfg_definition(cls) -> AdapterCfgDefinition:
        return T1Adapter.cfg_definition()

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return "rebind"

    def parameter_roles(self) -> Mapping[str, RoleSpec]:
        return {
            "driver": RoleSpec("fake/drive/*"),
            "sense": RoleSpec("fake/sensor", via="driver.sense"),
        }

    def make_default_cfg(self, ctx: SessionEnv) -> CfgSchema:
        assert ctx.parameters is not None
        return CfgSchema(
            CfgSectionSpec(
                fields={
                    "rate": ScalarSpec("Rate", float),
                    "seed_rate": ScalarSpec("Seed rate", float),
                }
            ),
            CfgSectionValue(
                fields={
                    "rate": EvalValue("driver.rate"),
                    "seed_rate": DirectValue(ctx.parameters.value("driver.rate")),
                }
            ),
        )

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> RunRecord[ExpCfgModel, dict[str, ExperimentCfgValue]]:
        return RunRecord(cfg=ExpCfgModel(), result=dict(raw_cfg))


@dataclass
class TargetApplication:
    state: State
    controller: Controller
    remote: RemoteControlAdapter


@pytest.fixture
def application(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[TargetApplication]:
    before = registry_state()
    try:
        register_fakes()
        component_registry.register("fake/drive/library", LibraryDriver)
        installed = registry_state()
        registry = Registry()
        registry.register("targeted", TargetAdapter)
        registry.register("dependent", DependentTargetAdapter)
        registry.register("roleless", RolelessAdapter)
        registry.register("unused_alias", UnusedAliasAdapter)
        registry.register("rebind", RebindAdapter)
        state = State(SessionEnv(parameters=None, soc=None, soccfg=None))
        background = BackgroundRunner()
        monkeypatch.setattr(
            "zcu_tools.gui.app.measure.services.app_services.BackgroundRunner",
            lambda: background,
        )
        controller = Controller(
            state, registry, None, BaseEventBus(), project_root=str(tmp_path)
        )
        controller.setup_control.new_entry(EntryRequest("entry"))
        context = controller.context_control
        context.add_component("Q1", "fake/drive/a", {"rate": 10.0, "sense": "R1"})
        context.add_component(
            "Q2",
            "fake/drive/a",
            {"rate": 20.0, "source": "Q1", "energy_a": None, "sense": "R2"},
        )
        context.add_component("S1", "fake/sensor", {"rate": 30.0})
        for name in ("R1", "R2", "R3"):
            context.add_component(name, "fake/sensor", {"rate": 30.0})
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
        assert background.quiesce()
        for tab_id in tuple(state.list_tab_ids()):
            controller.tab_control.close_tab(tab_id)
        assert registry_state() == installed
    finally:
        restore_registry(before)


def rate(application: TargetApplication, tab_id: str) -> object:
    cfg = application.state.get_tab(tab_id).cfg
    return cfg.accept(cfg.observe().ref.revision).values["rate"]


def test_role_rebind_reinfers_only_non_manual_roles_from_birth_focus(
    application: TargetApplication,
) -> None:
    control = application.controller.tab_control
    tab_id = control.new_tab("rebind")
    tab = application.state.get_tab(tab_id)
    application.controller.context_control.set_component_focus("Q2")
    control.set_role(tab_id, "driver", "Q2")
    assert tab.parameters is not None and tab.parameters.roles is not None
    assert tab.parameters.roles.focus == "Q1"
    assert tab.parameters.roles.names == {"driver": "Q2", "sense": "R2"}
    assert tab.parameters.roles.explicit == {"driver": "Q2"}
    control.set_role(tab_id, "sense", "R3")
    control.reset_role(tab_id, "driver")
    assert tab.parameters.roles.names == {"driver": "Q1", "sense": "R3"}
    assert tab.parameters.roles.explicit == {"sense": "R3"}
    control.set_role(tab_id, "driver", None)
    assert tab.parameters.roles.names == {"sense": "R3"}
    assert tab.parameters.roles.explicit == {"driver": None, "sense": "R3"}
    control.reset_role(tab_id, "sense")
    assert tab.parameters.roles.names == {}
    assert tab.parameters.roles.explicit == {"driver": None}
    control.reset_role(tab_id, "driver")
    assert tab.parameters.roles.names == {"driver": "Q1", "sense": "R1"}
    assert tab.parameters.roles.explicit == {}


def test_role_rebind_preserves_cfg_handle_inputs_and_reseeds_defaults(
    application: TargetApplication,
) -> None:
    control = application.controller.tab_control
    tab_id = control.new_tab("rebind")
    tab = application.state.get_tab(tab_id)
    cfg = tab.cfg
    editor = application.controller.cfg_resources.lookup(tab_id)
    cfg.edit(
        cfg.observe().ref.revision,
        (
            CfgEdit(("rate",), EvalValue("driver.rate * 2")),
            CfgEdit(("seed_rate",), DirectValue(7.0)),
        ),
    )
    previous = cfg.observe()
    observed: list[tuple[dict[str, str], object]] = []

    def changed(payload: TabContentChangedPayload) -> None:
        if payload.fact is TabContentFact.ROLES_COMMITTED:
            assert tab.parameters is not None and tab.parameters.roles is not None
            observed.append(
                (dict(tab.parameters.roles.names), rate(application, tab_id))
            )

    application.controller.get_bus().subscribe(TabContentChangedPayload, changed)
    control.set_role(tab_id, "driver", "Q2")
    assert (
        tab.cfg is cfg and application.controller.cfg_resources.lookup(tab_id) is editor
    )
    assert cfg.snapshot_inputs().value.fields["rate"] == EvalValue("driver.rate * 2")
    assert cfg.accept(cfg.observe().ref.revision).values == {
        "rate": 40.0,
        "seed_rate": 7.0,
    }
    assert observed == [({"driver": "Q2", "sense": "R2"}, 40.0)]
    with pytest.raises(CfgStaleError):
        cfg.accept(previous.ref.revision)
    editor.edit(
        cfg.observe().ref.revision, (CfgEdit(("seed_rate",), DirectValue(8.0)),)
    )
    editor.reset(cfg.observe().ref.revision)
    assert cfg.accept(cfg.observe().ref.revision).values == {
        "rate": 20.0,
        "seed_rate": 20.0,
    }


def test_role_rebind_clear_publishes_invalid_but_unused_alias_can_run(
    application: TargetApplication,
) -> None:
    context = application.controller.context_control
    context.new_point(PointCreateRequest("role-clear", "", CommandOrigin("gui")))
    context.add_component("Q1", "fake/drive/a", {"rate": 10.0, "sense": "R1"})
    context.add_component("R1", "fake/sensor", {"rate": 30.0})
    context.set_component_focus("Q1")
    control = application.controller.tab_control
    tab_id = control.new_tab("rebind")
    control.set_role(tab_id, "driver", None)
    tab = application.state.get_tab(tab_id)
    observation = tab.cfg.observe()
    assert observation.status is CfgStatus.INVALID
    assert any("driver" in item.message for item in observation.diagnostics)
    with pytest.raises(CfgPreconditionError):
        GuardService(application.state).acquire_run_permit(
            tab_id, expected_revision=observation.ref.revision
        )
    unused = control.new_tab("unused_alias")
    control.set_role(unused, "driver", None)
    cfg = application.state.get_tab(unused).cfg
    permit = GuardService(application.state).acquire_run_permit(
        unused, expected_revision=cfg.observe().ref.revision
    )
    assert permit.parameters is not None and permit.parameters.roles is not None
    assert permit.parameters.roles.names == {}
    assert permit.accepted_cfg.values == {"rate": 1.0}


@pytest.mark.parametrize(
    "role,component",
    [
        ("missing", "Q2"),
        ("driver", "missing"),
        ("driver", "S1"),
    ],
)
def test_role_rebind_invalid_choice_has_no_partial_commit(
    application: TargetApplication,
    role: str,
    component: str,
) -> None:
    control = application.controller.tab_control
    tab_id = control.new_tab("rebind")
    tab = application.state.get_tab(tab_id)
    capture, observation = tab.parameters, tab.cfg.observe()
    events: list[TabContentChangedPayload] = []
    application.controller.get_bus().subscribe(TabContentChangedPayload, events.append)
    with pytest.raises(RoleResolutionError):
        control.set_role(tab_id, role, component)
    assert tab.parameters is capture and tab.cfg.observe().ref == observation.ref
    assert events == []


@pytest.mark.parametrize("busy", ["running", "analyzing"])
def test_role_rebind_busy_tab_rejects_but_other_idle_tab_can_change(
    application: TargetApplication,
    busy: str,
) -> None:
    control = application.controller.tab_control
    tab_id = control.new_tab("rebind")
    idle = control.new_tab("rebind")
    state, tab = application.state, application.state.get_tab(tab_id)
    capture, observation = tab.parameters, tab.cfg.observe()
    if busy == "running":
        state.set_tab_running(tab_id, running=True)
    else:
        state.set_tab_analyzing(tab_id, analyzing=True)
    try:
        with pytest.raises(FailedPreconditionError):
            control.set_role(tab_id, "driver", "Q2")
        with pytest.raises(FailedPreconditionError):
            control.reset_role(tab_id, "driver")
        assert tab.parameters is capture and tab.cfg.observe().ref == observation.ref
        control.set_role(idle, "driver", "Q2")
        capture_idle = state.get_tab(idle).parameters
        assert capture_idle is not None and capture_idle.roles is not None
        assert capture_idle.roles.names["driver"] == "Q2"
    finally:
        state.set_tab_running(tab_id, running=False)
        state.set_tab_analyzing(tab_id, analyzing=False)


def test_role_rebind_reentrant_command_rejects_before_capture_install(
    application: TargetApplication,
) -> None:
    control = application.controller.tab_control
    tab_id, other = control.new_tab("rebind"), control.new_tab("rebind")
    tab = application.state.get_tab(tab_id)
    old_capture, old_ref = tab.parameters, tab.cfg.observe().ref
    cfg = application.state.get_tab(other).cfg
    failures: list[CfgPreconditionReason] = []

    def observer(_observation) -> None:
        with pytest.raises(CfgPreconditionError) as rejected:
            control.set_role(tab_id, "driver", "Q2")
        failures.append(rejected.value.reason)

    unwatch = cfg.watch(observer)
    failures.clear()
    try:
        cfg.refresh(cfg.observe().ref.revision)
    finally:
        unwatch()
    assert failures == [CfgPreconditionReason.REENTRANT_MUTATION]
    assert tab.parameters is old_capture and tab.cfg.observe().ref == old_ref


def test_role_rebind_roleless_tab_rejects_commands(
    application: TargetApplication,
) -> None:
    control = application.controller.tab_control
    tab_id = control.new_tab("roleless")
    with pytest.raises(FailedPreconditionError):
        control.set_role(tab_id, "driver", None)
    with pytest.raises(FailedPreconditionError):
        control.reset_role(tab_id, "driver")


def test_role_rebind_keeps_historical_panes_and_next_run_snapshot(
    application: TargetApplication,
) -> None:
    context, control = (
        application.controller.context_control,
        application.controller.tab_control,
    )
    context.new_point(PointCreateRequest("saved", "", CommandOrigin("gui")))
    for name, rate_value, sense in (("Q1", 10.0, "R1"), ("Q2", 20.0, "R2")):
        context.add_component(
            name, "fake/drive/a", {"rate": rate_value, "sense": sense}
        )
    for name in ("R1", "R2"):
        context.add_component(name, "fake/sensor", {"rate": 30.0})
    context.set_component_focus("Q1")
    tab_id = control.new_tab("rebind")
    tab = application.state.get_tab(tab_id)
    birth = tab.parameters
    assert birth is not None and birth.storage is not None
    application.state.update_tab_result(tab_id, object(), parameters=birth)
    historical_result = tab.run.result
    application.state.update_tab_analyze(tab_id, object(), None)
    historical_analysis = tab.analysis.result
    control.set_role(tab_id, "driver", "Q2")
    assert tab.run.parameters is birth and tab.run.result is historical_result
    assert tab.analysis.result is historical_analysis
    assert tab.save_context is birth
    permit = GuardService(application.state).acquire_run_permit(
        tab_id, expected_revision=tab.cfg.observe().ref.revision
    )
    assert permit.parameters is tab.parameters and permit.snapshot is not None
    assert permit.snapshot.roles == {"driver": "Q2", "sense": "R2"}
    assert permit.snapshot.entry_id == birth.storage.entry_id
    assert permit.snapshot.point == birth.storage.point
    token = application.controller.start_run(tab_id, tab.cfg.observe().ref)
    loop, poll, deadline = QEventLoop(), QTimer(), QTimer()
    poll.timeout.connect(
        lambda: loop.quit() if not application.state.is_tab_running(tab_id) else None
    )
    deadline.setSingleShot(True)
    deadline.timeout.connect(loop.quit)
    poll.start(5)
    deadline.start(10000)
    loop.exec()
    poll.stop()
    deadline.stop()
    assert not application.state.is_tab_running(tab_id)
    metadata = tab.run.metadata
    assert metadata is not None and tab.run.source_operation_id == token
    assert metadata.snapshot == permit.snapshot
    assert tab.run.parameters is not None and tab.run.parameters.roles is not None
    assert tab.run.parameters.roles.names == metadata.snapshot.roles


def test_auto_partial_alias_creates_invalid_tab_without_focus_fallback(
    application: TargetApplication,
) -> None:
    context = application.controller.context_control
    context.set_component_focus(None)
    reply = h_tab_new(application.remote, {"adapter_name": "targeted"})
    tab_id = reply["tab_id"]
    assert isinstance(tab_id, str)
    tab = application.state.get_tab(tab_id)
    assert application.state.list_tab_ids() == [tab_id]
    assert tab.parameters is not None and tab.parameters.roles is not None
    assert tab.parameters.roles.names == {}
    assert "driver" in tab.parameters.roles.declarations
    observation = tab.cfg.observe()
    assert observation.status is CfgStatus.INVALID
    assert any("driver" in item.message for item in observation.diagnostics)
    with pytest.raises(CfgPreconditionError):
        tab.cfg.accept(observation.ref.revision)
    context.set_component_focus("Q2")
    assert tab.parameters.roles.names == {}
    assert tab.cfg.observe().status is CfgStatus.INVALID


def test_auto_partial_unused_alias_can_acquire_run_permit(
    application: TargetApplication,
) -> None:
    context = application.controller.context_control
    context.new_point(PointCreateRequest("unbound", "", CommandOrigin("gui")))
    context.add_component("Q1", "fake/drive/a", {"rate": 10.0})
    context.add_component("Q2", "fake/drive/a", {"rate": 20.0})
    context.set_component_focus(None)
    reply = h_tab_new(application.remote, {"adapter_name": "unused_alias"})
    tab_id = reply["tab_id"]
    assert isinstance(tab_id, str)
    tab = application.state.get_tab(tab_id)
    assert tab.parameters is not None and tab.parameters.roles is not None
    assert tab.parameters.roles.names == {}
    observation = tab.cfg.observe()
    assert observation.status is CfgStatus.VALID
    permit = GuardService(application.state).acquire_run_permit(
        tab_id, expected_revision=observation.ref.revision
    )
    assert permit.parameters is tab.parameters
    assert permit.accepted_cfg.values == {"rate": 1.0}
    assert permit.request.soc is None


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


def test_explicit_target_allows_unbound_same_kind_peer_reference(
    application: TargetApplication,
) -> None:
    reply = h_tab_new(application.remote, {"adapter_name": "dependent", "target": "Q1"})
    tab_id = reply["tab_id"]
    assert isinstance(tab_id, str)
    tab = application.state.get_tab(tab_id)
    assert tab.parameters is not None and tab.parameters.roles is not None
    assert tab.parameters.roles.names == {"driver": "Q1"}
    observation = tab.cfg.observe()
    assert observation.status is CfgStatus.INVALID
    assert any("target" in item.message for item in observation.diagnostics)
    with pytest.raises(CfgPreconditionError):
        tab.cfg.accept(observation.ref.revision)
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
