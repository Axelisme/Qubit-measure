"""Explicit target capture through the shipped remote tab/app/context seam."""

from collections.abc import Generator, Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest
from zcu_tools.cfg_model import ConstWaveformCfg, PulseSpec
from zcu_tools.gui.app.measure.adapter import (
    SessionEnv,
)
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.remote import ControlOptions, RemoteControlAdapter
from zcu_tools.gui.app.measure.remote.handlers.tab import (
    h_tab_get_cfg,
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
from zcu_tools.gui.cfg.resource import (
    CfgEdit,
)
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import (
    EntryRequest,
)
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
            CfgSectionValue(fields={"rate": EvalValue("$driver.rate")}),
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
        registry.register("roleless", RolelessAdapter)
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
        context.set_alias_rows((AliasRow("driver", "Q1"),))
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


def test_named_aliases_apply_to_parameter_and_library_queries_without_mutating_sources(
    application: TargetApplication,
) -> None:
    context = application.controller.context_control
    context.set_alias_rows(
        (
            AliasRow("other", "Q1"),
            AliasRow("driver", "Q1"),
            AliasRow("sense", "$driver.sense"),
        )
    )
    tab_id = application.controller.tab_control.new_tab("targeted")
    rows = context.get_alias_rows()
    versions = application.state.version.snapshot()
    publication = application.state.get_tab(tab_id).cfg.observe()
    result = h_tab_parameter_read(
        application.remote,
        {
            "tab_id": tab_id,
            "aliases": {"driver": "QL", "sense": "R2"},
            "paths": ["$driver.rate", "$sense.rate", "$other.rate"],
            "module_refs": ["$driver.pulses.pi"],
        },
    )
    assert result == {
        "values": {"$driver.rate": 40.0, "$sense.rate": 30.0, "$other.rate": 10.0},
        "modules": {"$driver.pulses.pi": "QL.pulses.pi"},
    }
    assert context.get_alias_rows() == rows
    assert application.state.version.snapshot() == versions
    assert application.state.get_tab(tab_id).cfg.observe() == publication


def test_parameter_read_uses_current_sources_and_temporary_aliases(
    application: TargetApplication,
) -> None:
    tab_id = h_tab_new(application.remote, {"adapter_name": "targeted"})["tab_id"]
    assert isinstance(tab_id, str)
    context = application.controller.context_control
    context.set_parameter(context.read_context().target, "Q2.rate", 25.0)
    original_rows = context.get_alias_rows()
    sources = h_tab_parameter_read(
        application.remote,
        {
            "tab_id": tab_id,
            "target": "Q2",
            "paths": ["$driver.rate", "$driver.energy_a", "$driver.not_supplied"],
            "module_refs": [],
        },
    )
    assert sources == {
        "values": {"$driver.rate": 25.0, "$driver.energy_a": None},
        "modules": {},
    }
    assert context.get_alias_rows() == original_rows


def test_parameter_read_resolves_library_slots_in_current_temporary_source(
    application: TargetApplication,
) -> None:
    tab_id = h_tab_new(application.remote, {"adapter_name": "targeted"})["tab_id"]
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
            "target": "QL",
            "module_refs": [
                "$driver.module.pi",
                "$driver.pulses.pi",
                "QL.pulses.pi",
                "$driver.pulses.later",
                "<Custom:pulse>",
            ],
        },
    )
    assert sources == {
        "values": {},
        "modules": {
            "$driver.module.pi": None,
            "$driver.pulses.pi": "QL.pulses.pi",
            "QL.pulses.pi": "QL.pulses.pi",
            "$driver.pulses.later": "QL.pulses.later",
            "<Custom:pulse>": None,
        },
    }


def test_parameter_read_rejects_undeclared_alias(
    application: TargetApplication,
) -> None:
    from zcu_tools.resources.entry.errors import RoleResolutionError

    tab_id = h_tab_new(application.remote, {"adapter_name": "targeted"})["tab_id"]
    with pytest.raises(RoleResolutionError):
        h_tab_parameter_read(
            application.remote,
            {"tab_id": tab_id, "paths": ["$qubit.freq"], "module_refs": []},
        )


def test_parameter_read_roleless_tab_can_read_current_concrete_paths(
    application: TargetApplication,
) -> None:
    tab_id = h_tab_new(application.remote, {"adapter_name": "roleless"})["tab_id"]
    assert h_tab_parameter_read(
        application.remote,
        {"tab_id": tab_id, "paths": ["Q2.rate"], "module_refs": ["QL.pulses.pi"]},
    ) == {"values": {"Q2.rate": 20.0}, "modules": {"QL.pulses.pi": "QL.pulses.pi"}}


def test_target_cfg_preview_keeps_publication_and_rejects_stale_ref(
    application: TargetApplication,
) -> None:
    tab_id = application.controller.tab_control.new_tab("targeted")
    cfg = application.state.get_tab(tab_id).cfg
    observed = cfg.edit(cfg.observe().ref.revision, (CfgEdit(("rate",), 17.0),))
    expected = {
        "cfg_id": str(observed.ref.cfg_id),
        "revision": str(observed.ref.revision),
    }
    preview = h_tab_get_cfg(
        application.remote, {"tab_id": tab_id, "target": "Q2", "expected": expected}
    )
    assert preview["status"] == "Valid"
    assert preview["cfg_ref"] == expected
    assert cfg.observe() == observed
    cfg.edit(observed.ref.revision, (CfgEdit(("rate",), 18.0),))
    with pytest.raises(RemoteError) as caught:
        h_tab_get_cfg(
            application.remote, {"tab_id": tab_id, "target": "Q2", "expected": expected}
        )
    assert caught.value.reason == "stale_revision"


@pytest.mark.parametrize("target", ["", " ", 2, False])
def test_target_source_reads_reject_malformed_target(
    application: TargetApplication, target: object
) -> None:
    tab_id = application.controller.tab_control.new_tab("targeted")
    for handler, query in (
        (h_tab_parameter_read, {"paths": [], "module_refs": []}),
        (h_tab_get_cfg, {}),
    ):
        with pytest.raises(RemoteError) as caught:
            handler(application.remote, {"tab_id": tab_id, "target": target, **query})
        assert caught.value.reason == "invalid_target"


def test_target_preview_requires_expected_ref(application: TargetApplication) -> None:
    tab_id = application.controller.tab_control.new_tab("targeted")
    with pytest.raises(RemoteError) as caught:
        h_tab_get_cfg(application.remote, {"tab_id": tab_id, "target": "Q2"})
    assert caught.value.code is ErrorCode.INVALID_PARAMS


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
