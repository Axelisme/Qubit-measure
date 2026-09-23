from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
from zcu_tools.device.fake import FakeDeviceInfo
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.gui.app.main.adapter import AdapterCapabilities
from zcu_tools.gui.app.main.events.tab import TabContentChangedPayload, TabContentFact
from zcu_tools.gui.app.main.services.cfg_editor import CfgEditorError, CfgEditorService
from zcu_tools.gui.app.main.services.guard import LoadPermit
from zcu_tools.gui.app.main.services.load import LoadDataError, LoadService
from zcu_tools.gui.app.main.state import ExpContext, Session, State
from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    ScalarSpec,
)
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.meta_tool import ModuleLibrary


class RuntimeCfg(ExpCfgModel):
    reps: int = 20


class NullableRuntimeCfg(ExpCfgModel):
    reps: object = None


@dataclass
class Result:
    cfg_snapshot: ExpCfgModel | None


@pytest.fixture
def app():
    bus = BaseEventBus()
    host = MagicMock()
    host.get_current_md.return_value = {}
    host.get_current_ml.return_value = None
    host.list_device_names.return_value = []
    host.list_arb_waveforms.return_value = []
    state = State(ExpContext(md=MagicMock(), ml=ModuleLibrary(), soc=None, soccfg=None))
    schema = CfgSchema(
        spec=CfgSectionSpec(
            fields={
                "reps": ScalarSpec("Reps", int),
                "note": ScalarSpec("Note", str, required=True),
            }
        ),
        value=CfgSectionValue(
            fields={"reps": DirectValue(1), "note": DirectValue("old")}
        ),
    )
    adapter = MagicMock()
    adapter.capabilities = AdapterCapabilities(load_data=True)
    adapter.load.return_value = Result(RuntimeCfg())
    state.add_tab(
        "tab", Session(adapter_name="test", adapter=adapter, cfg_schema=schema)
    )
    editors = CfgEditorService(
        host,
        read_port=host,
        write_port=host,
        version_bump=host.bump_editor_version,
        version_drop=host.drop_editor_version,
        bus=bus,
    )
    service = LoadService(state, MagicMock(), cfg_editor=editors, bus=bus)
    yield state, editors, service, adapter, bus
    editor_id = editors.editor_id_for_owner("tab")
    if editor_id is not None:
        editors.teardown(editor_id)


def test_load_adopts_snapshot_and_preserves_unflushed_draft_values(app):
    state, editors, service, adapter, bus = app
    original, _ = editors.open_seeded(state.get_tab("tab").cfg_schema, owner_key="tab")
    old_draft = editors.get_draft(original)
    editors.set_field(original, "reps", 7)
    editors.set_field(original, "note", "unflushed")
    observations = []
    bus.subscribe(
        TabContentChangedPayload,
        lambda event: observations.append(
            (
                event.fact,
                editors.editor_id_for_owner("tab"),
                state.get_tab("tab").cfg_schema,
            )
        ),
    )
    version = state.version.get("tab:tab:cfg")

    outcome = service.load_result(LoadPermit("tab"), "result.hdf5")

    assert outcome.cfg_backfill == "applied"
    assert state.get_tab("tab").run.result is adapter.load.return_value
    assert state.version.get("tab:tab:cfg") == version + 1
    schema = state.get_tab("tab").cfg_schema
    assert schema.value.fields == {
        "reps": DirectValue(20),
        "note": DirectValue("unflushed"),
    }
    replacement = editors.editor_id_for_owner("tab")
    assert observations == [(TabContentFact.CFG_REPLACED, replacement, schema)]
    with pytest.raises(CfgEditorError):
        editors.set_field(original, "reps", 99)
    with pytest.raises(RuntimeError):
        old_draft.snapshot()
    assert editors.snapshot_owner("tab") == schema


@pytest.mark.parametrize("snapshot", [None, ExpCfgModel()])
def test_unavailable_snapshot_keeps_live_config_and_editor(app, snapshot):
    state, editors, service, adapter, bus = app
    before = state.get_tab("tab").cfg_schema
    original, _ = editors.open_seeded(before, owner_key="tab")
    adapter.load.return_value = Result(snapshot)
    events = []
    bus.subscribe(TabContentChangedPayload, events.append)
    outcome = service.load_result(LoadPermit("tab"), "result.hdf5")
    assert outcome.cfg_backfill == "not_applied"
    assert state.get_tab("tab").cfg_schema is before
    assert state.get_tab("tab").run.result is adapter.load.return_value
    assert editors.editor_id_for_owner("tab") == original
    assert events == []


def test_invalid_complete_candidate_leaves_state_and_draft_unchanged(app):
    state, editors, service, adapter, _ = app
    before = state.get_tab("tab").cfg_schema
    original, _ = editors.open_seeded(before, owner_key="tab")
    editors.set_field(original, "note", "")
    draft_before = editors.snapshot_owner("tab")
    version = state.version.get("tab:tab:cfg")
    outcome = service.load_result(LoadPermit("tab"), "result.hdf5")
    assert outcome.cfg_backfill == "not_applied"
    assert state.get_tab("tab").cfg_schema is before
    assert state.version.get("tab:tab:cfg") == version
    assert editors.snapshot_owner("tab") == draft_before
    assert editors.editor_id_for_owner("tab") == original
    assert state.get_tab("tab").run.result is adapter.load.return_value


def test_nonoptional_null_snapshot_does_not_replace_config_or_draft(app):
    state, editors, service, adapter, bus = app
    before = state.get_tab("tab").cfg_schema
    original, _ = editors.open_seeded(before, owner_key="tab")
    adapter.load.return_value = Result(NullableRuntimeCfg())
    events = []
    bus.subscribe(TabContentChangedPayload, events.append)

    outcome = service.load_result(LoadPermit("tab"), "result.hdf5")
    assert outcome.cfg_backfill == "not_applied"
    assert state.get_tab("tab").run.result is adapter.load.return_value
    assert state.get_tab("tab").cfg_schema is before
    assert editors.editor_id_for_owner("tab") == original
    assert editors.snapshot_owner("tab") == before
    assert events == []


@pytest.fixture
def device_app():
    bus = BaseEventBus()
    host = MagicMock()
    host.get_current_md.return_value = {}
    host.get_current_ml.return_value = None
    host.list_device_names.return_value = ["stable"]
    host.list_arb_waveforms.return_value = []
    state = State(ExpContext(md=MagicMock(), ml=ModuleLibrary(), soc=None, soccfg=None))
    schema = CfgSchema(
        spec=CfgSectionSpec(
            fields={
                "reps": ScalarSpec("Reps", int),
                "dev": CfgSectionSpec(
                    fields={
                        "jpa_rf_dev": ScalarSpec(
                            "JPA RF device",
                            str,
                            required=True,
                            choices_source="devices",
                        )
                    }
                ),
            }
        ),
        value=CfgSectionValue(
            fields={
                "reps": DirectValue(1),
                "dev": CfgSectionValue(fields={"jpa_rf_dev": DirectValue("stable")}),
            }
        ),
    )
    adapter = MagicMock()
    adapter.capabilities = AdapterCapabilities(load_data=True)
    state.add_tab(
        "tab", Session(adapter_name="test", adapter=adapter, cfg_schema=schema)
    )
    editors = CfgEditorService(
        host,
        read_port=host,
        write_port=host,
        version_bump=host.bump_editor_version,
        version_drop=host.drop_editor_version,
        bus=bus,
    )
    original, _ = editors.open_seeded(schema, owner_key="tab")
    service = LoadService(state, MagicMock(), cfg_editor=editors, bus=bus)
    yield state, editors, service, adapter, bus, host, original
    current_id = editors.editor_id_for_owner("tab")
    if current_id is not None:
        editors.teardown(current_id)


def test_missing_device_option_keeps_existing_selector_and_editor(device_app):
    state, editors, service, adapter, bus, _, original = device_app
    adapter.load.return_value = Result(
        ExpCfgModel(dev={"stale": FakeDeviceInfo(address="fake", label="jpa_rf_dev")})
    )
    before = state.get_tab("tab").cfg_schema
    events = []
    bus.subscribe(TabContentChangedPayload, events.append)

    outcome = service.load_result(LoadPermit("tab"), "result.hdf5")
    assert outcome.cfg_backfill == "not_applied"
    assert state.get_tab("tab").run.result is adapter.load.return_value
    assert state.get_tab("tab").cfg_schema is before
    assert editors.editor_id_for_owner("tab") == original
    assert editors.get_draft(original).is_valid()
    assert events == []


def test_missing_device_option_is_preserved_when_other_fields_backfill(device_app):
    state, editors, service, adapter, _, _, original = device_app
    adapter.load.return_value = Result(
        RuntimeCfg(dev={"stale": FakeDeviceInfo(address="fake", label="jpa_rf_dev")})
    )
    outcome = service.load_result(LoadPermit("tab"), "result.hdf5")
    assert outcome.cfg_backfill == "applied"
    assert state.get_tab("tab").run.result is adapter.load.return_value
    assert state.get_tab("tab").cfg_schema.value.fields["reps"] == DirectValue(20)
    dev = state.get_tab("tab").cfg_schema.value.fields["dev"]
    assert isinstance(dev, CfgSectionValue)
    assert dev.fields["jpa_rf_dev"] == DirectValue("stable")
    replacement = editors.editor_id_for_owner("tab")
    assert replacement is not None and replacement != original
    assert editors.get_draft(replacement).is_valid()
    assert editors.snapshot_owner("tab") == state.get_tab("tab").cfg_schema


def test_device_choice_disappears_before_draft_preparation(device_app):
    state, editors, service, adapter, bus, host, original = device_app
    host.list_device_names.side_effect = [
        ["stable", "stale"],  # converter sees the saved name
        ["stable"],  # new CfgDraft must reject it before publication
    ]
    adapter.load.return_value = Result(
        RuntimeCfg(dev={"stale": FakeDeviceInfo(address="fake", label="jpa_rf_dev")})
    )
    before = state.get_tab("tab").cfg_schema
    events = []
    bus.subscribe(TabContentChangedPayload, events.append)

    outcome = service.load_result(LoadPermit("tab"), "result.hdf5")
    assert outcome.cfg_backfill == "not_applied"
    assert state.get_tab("tab").run.result is adapter.load.return_value
    assert state.get_tab("tab").cfg_schema is before
    assert editors.editor_id_for_owner("tab") == original
    assert editors.get_draft(original).is_valid()
    assert events == []


def test_prepare_failure_does_not_undo_loaded_result(app, monkeypatch):
    state, editors, service, adapter, _ = app
    before = state.get_tab("tab").cfg_schema
    original, _ = editors.open_seeded(before, owner_key="tab")
    monkeypatch.setattr(
        editors,
        "prepare_replacement",
        MagicMock(side_effect=RuntimeError("allocation failed")),
    )
    outcome = service.load_result(LoadPermit("tab"), "result.hdf5")
    assert outcome.cfg_backfill == "not_applied"
    assert state.get_tab("tab").cfg_schema is before
    assert editors.editor_id_for_owner("tab") == original
    assert state.get_tab("tab").run.result is adapter.load.return_value


def test_headless_load_publishes_editable_owner_despite_broken_view(app):
    state, editors, service, _, bus = app

    def broken_view(event):
        raise RuntimeError("view failed")

    bus.subscribe(TabContentChangedPayload, broken_view)
    outcome = service.load_result(LoadPermit("tab"), "result.hdf5")
    assert outcome.cfg_backfill == "applied"
    replacement = editors.editor_id_for_owner("tab")
    assert replacement is not None
    editors.set_field(replacement, "reps", 30)
    assert editors.snapshot_owner("tab").value.fields["reps"] == DirectValue(30)
    assert state.get_tab("tab").cfg_schema.value.fields["reps"] == DirectValue(20)


def test_loader_failure_preserves_result_cfg_and_editor(app):
    state, editors, service, adapter, _ = app
    before = state.get_tab("tab").cfg_schema
    original, _ = editors.open_seeded(before, owner_key="tab")
    old_result = Result(None)
    state.update_tab_loaded_result("tab", old_result, "old.hdf5")
    adapter.load.side_effect = ValueError("bad file")
    with pytest.raises(LoadDataError):
        service.load_result(LoadPermit("tab"), "bad.hdf5")
    assert state.get_tab("tab").cfg_schema is before
    assert state.get_tab("tab").run.result is old_result
    assert editors.editor_id_for_owner("tab") == original
