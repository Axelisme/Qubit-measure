"""Native result Load backfills one captured Form resource atomically."""

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field
from zcu_tools.cfg_model import AliasName
from zcu_tools.device.fake import FakeDeviceInfo
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import AdapterCapabilities, Form
from zcu_tools.gui.app.measure.services.load import LoadService
from zcu_tools.gui.app.measure.services.tab_form import TabFormResource, TabFormSources
from zcu_tools.gui.app.measure.state import Session, SessionEnv, State
from zcu_tools.gui.cfg.resource import CfgPreconditionError, CfgPreconditionReason
from zcu_tools.gui.form import FormEnv
from zcu_tools.gui.session.state import DeviceState, DeviceStatus

from tests.gui.app.measure._load_support import historical_metadata


class RuntimeCfg(ExpCfgModel):
    """A loaded core stores repeats and a nullable named device."""

    reps: int = Field(default=20, ge=1)
    rf_dev: str | None = None


@dataclass
class UnprojectableSnapshot:
    """An adapter-owned result may have a cfg that is not an experiment model."""

    cfg: object


class Note(BaseModel):
    """Editable GUI-only annotation does not change native execution cfg."""

    note: str = Field(default="old", min_length=1)


def keep_cfg(_extras: Note, cfg: RuntimeCfg) -> RuntimeCfg:
    return cfg


@dataclass
class LoadApp:
    state: State
    cfg: TabFormResource
    service: LoadService
    adapter: MagicMock
    form: Form[RuntimeCfg]


def make_app(*, device: bool = False) -> LoadApp:
    state = State(SessionEnv(parameters=None, soc=None, soccfg=None))
    if device:
        state.put_device(
            DeviceState(
                name="stable",
                type_name="FakeDevice",
                address="fake",
                remember=True,
                status=DeviceStatus.MEMORY_ONLY,
                info=FakeDeviceInfo(address="fake", value=1.0),
            )
        )
    form = Form(
        RuntimeCfg,
        {"reps": 1, "rf_dev": AliasName("rf_dev")},
        extras=Note,
        lower=keep_cfg,
    )

    def capture(_aliases: object) -> TabFormSources:
        return TabFormSources(
            FormEnv(
                devices={
                    item.name: item.info
                    for item in state.list_devices()
                    if item.info is not None
                }
            )
        )

    cfg = TabFormResource(form, capture=capture, mutation_allowed=lambda: True)
    adapter = MagicMock()
    adapter.capabilities = AdapterCapabilities(load_data=True)
    adapter.load.return_value = (
        RunRecord(RuntimeCfg(), object()),
        historical_metadata(),
    )
    state.add_tab("tab", Session(adapter_name="test", adapter=adapter, cfg=cfg))

    def cfg_changed() -> None:
        state.version.bump("tab:tab:cfg")

    cfg.subscribe(cfg_changed)
    if device:
        cfg.write({("rf_dev",): "stable"})
    service = LoadService(
        state,
        MagicMock(),
        context=MagicMock(),
        captured_parameters=lambda _aliases: None,
    )
    return LoadApp(state, cfg, service, adapter, form)


@pytest.fixture
def app():
    application = make_app()
    yield application
    application.cfg.revoke()


@pytest.fixture
def device_app():
    application = make_app(device=True)
    yield application
    application.cfg.revoke()


def test_load_updates_same_resource_and_preserves_current_input(app: LoadApp) -> None:
    editor = app.cfg
    editor.write({("reps",): 7, ("extras", "note"): "unflushed"})
    observations: list[dict[str, object]] = []
    unsubscribe = editor.subscribe(lambda: observations.append(dict(editor.input.data)))
    outcome = app.service.load_result("tab", "result.hdf5")
    unsubscribe()
    assert outcome.cfg_backfill == "applied"
    assert app.state.get_tab("tab").run.result is app.adapter.load.return_value[0]
    assert app.state.get_tab("tab").run.metadata == app.adapter.load.return_value[1]
    assert editor.input.data["extras"] == {"note": "unflushed"}
    assert editor.accept().values["reps"] == 20
    assert observations == [dict(editor.input.data)]
    editor.write({("reps",): 9})
    assert app.cfg.accept().values["reps"] == 9


@pytest.mark.parametrize("snapshot", [None, object()])
def test_unavailable_snapshot_keeps_cfg_and_loaded_result(
    app: LoadApp, snapshot: object
) -> None:
    app.adapter.load.return_value = (
        UnprojectableSnapshot(snapshot),
        historical_metadata(),
    )
    before = app.cfg.input.data
    outcome = app.service.load_result("tab", "result.hdf5")
    assert outcome.cfg_backfill == "not_applied"
    assert app.cfg.input.data == before
    assert app.state.get_tab("tab").run.result is app.adapter.load.return_value[0]


def test_invalid_complete_candidate_keeps_input_and_revision(app: LoadApp) -> None:
    app.cfg.write({("extras", "note"): ""})
    before = app.cfg.input.data
    revision = app.state.version.get("tab:tab:cfg")
    outcome = app.service.load_result("tab", "result.hdf5")
    assert outcome.cfg_backfill == "not_applied"
    assert outcome.cfg_rejection is not None
    assert ("extras",) in outcome.cfg_rejection.errors.messages
    assert app.cfg.input.data == before
    assert app.state.version.get("tab:tab:cfg") == revision


def test_nonoptional_null_snapshot_keeps_resource(app: LoadApp) -> None:
    invalid = RuntimeCfg.model_construct(reps=None, rf_dev=None)
    app.adapter.load.return_value = (
        RunRecord(invalid, object()),
        historical_metadata(),
    )
    before = app.cfg.input.data
    outcome = app.service.load_result("tab", "result.hdf5")
    assert outcome.cfg_backfill == "not_applied"
    assert app.cfg.input.data == before


def test_missing_device_option_without_other_fields_keeps_cfg(
    device_app: LoadApp,
) -> None:
    device_app.adapter.load.return_value = (
        RunRecord(RuntimeCfg(reps=1, rf_dev="gone"), object()),
        historical_metadata(),
    )
    before = device_app.cfg.input.data
    outcome = device_app.service.load_result("tab", "result.hdf5")
    assert outcome.cfg_backfill == "not_applied"
    assert device_app.cfg.input.data == before


def test_missing_device_option_rejects_other_fields_too(device_app: LoadApp) -> None:
    device_app.adapter.load.return_value = (
        RunRecord(RuntimeCfg(reps=40, rf_dev="gone"), object()),
        historical_metadata(),
    )
    before = device_app.cfg.input.data
    outcome = device_app.service.load_result("tab", "result.hdf5")
    assert outcome.cfg_backfill == "not_applied"
    assert device_app.cfg.input.data == before
    assert device_app.cfg.accept().values["reps"] == 1


def test_device_added_after_open_requires_reset_before_backfill(
    device_app: LoadApp,
) -> None:
    device_app.state.put_device(
        DeviceState(
            name="late",
            type_name="FakeDevice",
            address="fake",
            remember=True,
            status=DeviceStatus.MEMORY_ONLY,
            info=FakeDeviceInfo(address="fake", value=2.0),
        )
    )
    device_app.adapter.load.return_value = (
        RunRecord(RuntimeCfg(rf_dev="late"), object()),
        historical_metadata(),
    )
    before = device_app.cfg.input.data
    assert (
        device_app.service.load_result("tab", "result.hdf5").cfg_backfill
        == "not_applied"
    )
    assert device_app.cfg.input.data == before
    device_app.cfg.reset()
    assert (
        device_app.service.load_result("tab", "result.hdf5").cfg_backfill == "applied"
    )
    assert device_app.cfg.accept().values["rf_dev"] == "late"


def test_option_failure_keeps_entire_cfg_and_loaded_result(
    app: LoadApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(_env: FormEnv):
        raise RuntimeError("choice declaration failed")

    monkeypatch.setattr(app.form.model, "choices", unavailable)
    before = app.cfg.input.data
    outcome = app.service.load_result("tab", "result.hdf5")
    assert outcome.cfg_backfill == "not_applied"
    assert app.cfg.input.data == before
    assert app.state.get_tab("tab").run.result is app.adapter.load.return_value[0]


def test_reentrant_load_does_not_publish_cfg_or_replace_result(app: LoadApp) -> None:
    failures: list[CfgPreconditionReason] = []

    def callback() -> None:
        try:
            app.service.load_result("tab", "result.hdf5")
        except CfgPreconditionError as exc:
            failures.append(exc.reason)

    unsubscribe = app.cfg.subscribe(callback)
    app.cfg.write({("reps",): 7})
    unsubscribe()
    assert failures == [CfgPreconditionReason.REENTRANT_MUTATION]
    assert app.state.get_tab("tab").run.result is None
    app.adapter.load.assert_not_called()
