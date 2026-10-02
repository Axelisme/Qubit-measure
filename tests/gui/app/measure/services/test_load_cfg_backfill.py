"""Loaded result backfill publishes through the existing cfg resource."""

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
from zcu_tools.device.fake import FakeDeviceInfo
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import AdapterCapabilities
from zcu_tools.gui.app.measure.services.guard import LoadPermit
from zcu_tools.gui.app.measure.services.load import LoadService
from zcu_tools.gui.app.measure.services.tab_cfg import TabCfgResources
from zcu_tools.gui.app.measure.state import Session, SessionEnv, State
from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    ScalarSpec,
)
from zcu_tools.gui.cfg.model import CfgNodeSpec, CfgNodeValue
from zcu_tools.gui.cfg.resource import (
    CfgEdit,
    CfgPreconditionError,
    CfgResource,
    CfgStatus,
)
from zcu_tools.gui.session.state import DeviceState, DeviceStatus
from zcu_tools.resources.context import MetaDict, ModuleLibrary

from tests.gui.app.measure._cfg_fakes import cfg_resources


class RuntimeCfg(ExpCfgModel):
    reps: int = 20


class NullableRuntimeCfg(ExpCfgModel):
    reps: object = None


def make_record(cfg: ExpCfgModel | None) -> RunRecord[ExpCfgModel, object]:
    return RunRecord(cfg=cfg, result=object())


@dataclass
class LoadApp:
    state: State
    resources: TabCfgResources
    cfg: CfgResource
    service: LoadService
    adapter: MagicMock
    options: MagicMock


def make_app(*, device: bool = False) -> LoadApp:
    state = State(
        SessionEnv(md=MetaDict(None), ml=ModuleLibrary(None), soc=None, soccfg=None)
    )
    fields: dict[str, CfgNodeSpec] = {
        "reps": ScalarSpec("Reps", int),
        "note": ScalarSpec("Note", str, required=True),
    }
    values: dict[str, CfgNodeValue | None] = {
        "reps": DirectValue(1),
        "note": DirectValue("old"),
    }
    if device:
        state.put_device(
            DeviceState(
                name="stable",
                type_name="FakeDevice",
                address="fake",
                remember=True,
                status=DeviceStatus.MEMORY_ONLY,
            )
        )
        fields["dev"] = CfgSectionSpec(
            fields={
                "jpa_rf_dev": ScalarSpec(
                    "JPA RF device", str, required=True, choices_source="devices"
                )
            }
        )
        values["dev"] = CfgSectionValue(fields={"jpa_rf_dev": DirectValue("stable")})
    schema = CfgSchema(CfgSectionSpec(fields=fields), CfgSectionValue(fields=values))
    resources = cfg_resources(state)
    cfg = resources.create("tab", lambda: schema)
    adapter = MagicMock()
    adapter.capabilities = AdapterCapabilities(load_data=True)
    adapter.load.return_value = make_record(RuntimeCfg())
    state.add_tab("tab", Session(adapter_name="test", adapter=adapter, cfg=cfg))
    options = MagicMock(return_value=["stable"] if device else [])
    service = LoadService(state, MagicMock(), provide_options=options)
    return LoadApp(state, resources, cfg, service, adapter, options)


@pytest.fixture
def app():
    application = make_app()
    yield application
    application.resources.retire("tab")


@pytest.fixture
def device_app():
    application = make_app(device=True)
    yield application
    application.resources.retire("tab")


def test_load_updates_same_resource_and_preserves_current_input(app: LoadApp) -> None:
    editor = app.resources.lookup("tab")
    initial = editor.observe().ref
    editor.edit(
        initial.revision,
        (
            CfgEdit(("reps",), 7),
            CfgEdit(("note",), "unflushed"),
        ),
    )
    observations = []
    unsubscribe = editor.watch(observations.append)
    before = editor.observe()
    outcome = app.service.load_result(LoadPermit("tab"), "result.hdf5")
    after = editor.observe()
    unsubscribe()

    assert outcome.cfg_backfill == "applied"
    assert app.state.get_tab("tab").run.result is app.adapter.load.return_value
    assert after.ref.cfg_id == before.ref.cfg_id
    assert after.ref.revision == before.ref.revision + 1
    assert after.status is CfgStatus.VALID
    assert app.cfg.snapshot_inputs().value.fields == {
        "reps": DirectValue(20),
        "note": DirectValue("unflushed"),
    }
    assert observations[-1] == after
    editor.edit(after.ref.revision, (CfgEdit(("reps",), 9),))
    assert app.cfg.accept(editor.observe().ref.revision).values["reps"] == 9


@pytest.mark.parametrize("snapshot", [None, ExpCfgModel()])
def test_unavailable_snapshot_keeps_cfg_and_loaded_result(
    app: LoadApp, snapshot
) -> None:
    before = app.cfg.observe()
    app.adapter.load.return_value = make_record(snapshot)
    outcome = app.service.load_result(LoadPermit("tab"), "result.hdf5")
    assert outcome.cfg_backfill == "not_applied"
    assert app.cfg.observe() == before
    assert app.state.get_tab("tab").run.result is app.adapter.load.return_value


def test_invalid_complete_candidate_keeps_input_and_revision(app: LoadApp) -> None:
    app.cfg.edit(app.cfg.observe().ref.revision, (CfgEdit(("note",), ""),))
    before = app.cfg.observe()
    inputs = app.cfg.snapshot_inputs()
    outcome = app.service.load_result(LoadPermit("tab"), "result.hdf5")
    assert outcome.cfg_backfill == "not_applied"
    assert app.cfg.observe() == before
    assert app.cfg.snapshot_inputs() == inputs
    assert app.state.get_tab("tab").run.result is app.adapter.load.return_value


def test_nonoptional_null_snapshot_keeps_resource(app: LoadApp) -> None:
    before = app.cfg.observe()
    app.adapter.load.return_value = make_record(NullableRuntimeCfg())
    outcome = app.service.load_result(LoadPermit("tab"), "result.hdf5")
    assert outcome.cfg_backfill == "not_applied"
    assert app.cfg.observe() == before
    assert app.state.get_tab("tab").run.result is app.adapter.load.return_value


def test_missing_device_option_without_other_fields_keeps_cfg(
    device_app: LoadApp,
) -> None:
    app = device_app
    app.adapter.load.return_value = make_record(
        ExpCfgModel(dev={"stale": FakeDeviceInfo(address="fake", label="jpa_rf_dev")})
    )
    before = app.cfg.observe()
    outcome = app.service.load_result(LoadPermit("tab"), "result.hdf5")
    assert outcome.cfg_backfill == "not_applied"
    assert app.cfg.observe() == before
    assert app.state.get_tab("tab").run.result is app.adapter.load.return_value


def test_missing_device_option_preserves_selector_when_other_fields_apply(
    device_app: LoadApp,
) -> None:
    app = device_app
    app.adapter.load.return_value = make_record(
        RuntimeCfg(dev={"stale": FakeDeviceInfo(address="fake", label="jpa_rf_dev")})
    )
    before = app.cfg.observe()
    outcome = app.service.load_result(LoadPermit("tab"), "result.hdf5")
    assert outcome.cfg_backfill == "applied"
    accepted = app.cfg.accept(app.cfg.observe().ref.revision)
    assert accepted.values["reps"] == 20
    assert accepted.values["dev"] == {"jpa_rf_dev": "stable"}
    assert app.cfg.observe().ref.cfg_id == before.ref.cfg_id


def test_candidate_device_missing_from_fixed_source_rejects_complete_backfill(
    device_app: LoadApp,
) -> None:
    app = device_app
    app.options.return_value = ["stable", "new"]
    app.adapter.load.return_value = make_record(
        RuntimeCfg(dev={"new": FakeDeviceInfo(address="fake", label="jpa_rf_dev")})
    )
    before = app.cfg.observe()
    outcome = app.service.load_result(LoadPermit("tab"), "result.hdf5")
    assert outcome.cfg_backfill == "not_applied"
    assert app.cfg.observe() == before
    assert app.state.get_tab("tab").run.result is app.adapter.load.return_value


@pytest.mark.parametrize(
    "response",
    [
        RuntimeError("device discovery failed"),
        TypeError("device discovery failed"),
        ValueError("device discovery failed"),
        "not an option list",
        {"stable": True},
        {"stable"},
    ],
)
def test_option_failure_keeps_entire_cfg_and_loaded_result(
    device_app: LoadApp, response
) -> None:
    app = device_app
    app.adapter.load.return_value = make_record(
        RuntimeCfg(dev={"new": FakeDeviceInfo(address="fake", label="jpa_rf_dev")})
    )
    if isinstance(response, Exception):
        app.options.side_effect = response
    else:
        app.options.return_value = response
    before = app.cfg.observe()
    inputs = app.cfg.snapshot_inputs()
    outcome = app.service.load_result(LoadPermit("tab"), "result.hdf5")
    assert outcome.cfg_backfill == "not_applied"
    assert app.cfg.observe() == before
    assert app.cfg.snapshot_inputs() == inputs
    assert app.state.get_tab("tab").run.result is app.adapter.load.return_value
    app.cfg.edit(before.ref.revision, (CfgEdit(("reps",), 9),))
    assert app.cfg.accept(app.cfg.observe().ref.revision).values["reps"] == 9


def test_reentrant_load_does_not_publish_cfg_or_replace_result(app: LoadApp) -> None:
    attempts = []
    before = app.cfg.observe()

    def subscriber(observation) -> None:
        if observation.ref == before.ref:
            return
        try:
            app.service.load_result(LoadPermit("tab"), "result.hdf5")
        except CfgPreconditionError as exc:
            attempts.append(exc)

    unsubscribe = app.cfg.watch(subscriber)
    app.cfg.edit(before.ref.revision, (CfgEdit(("reps",), 3),))
    unsubscribe()
    # Loading during publication must be rejected before changing result state.
    assert attempts
    assert app.state.get_tab("tab").run.result is None
    assert app.cfg.accept(app.cfg.observe().ref.revision).values["reps"] == 3
