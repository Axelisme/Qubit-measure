from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.measure.services.cfg_editor import (
    CfgEditorError,
    CfgEditorService,
)
from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    ScalarSpec,
)
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.resources.context import MetaDict


@pytest.fixture
def publications():
    return []


@pytest.fixture
def metadata():
    return MetaDict()


@pytest.fixture
def service(publications, metadata):
    host = MagicMock()
    host.get_current_md.return_value = metadata
    host.get_current_ml.return_value = None
    host.list_device_names.return_value = []
    host.list_arb_waveforms.return_value = []
    svc = CfgEditorService(
        host,
        read_port=host,
        write_port=host,
        versions=host,
        bus=BaseEventBus(),
        publish_owner=lambda owner, schema: publications.append((owner, schema)),
    )
    yield svc
    editor_id = svc.editor_id_for_owner("tab")
    if editor_id is not None:
        svc.teardown(editor_id)


def seed(value: int) -> CfgSchema:
    return CfgSchema(
        spec=CfgSectionSpec(fields={"reps": ScalarSpec("Repetitions", int)}),
        value=CfgSectionValue(fields={"reps": DirectValue(value)}),
    )


def test_reopening_owner_rejects_stale_client_writes(service):
    original, _ = service.open_seeded(seed(1), owner_key="tab")
    replacement, _ = service.open_seeded(seed(2), owner_key="tab")

    assert service.editor_id_for_owner("tab") == replacement
    with pytest.raises(CfgEditorError):
        service.set_field(original, "reps", 99)
    assert service.get_draft(replacement).snapshot().value.fields[
        "reps"
    ] == DirectValue(2)

    service.set_field(replacement, "reps", 3)
    assert service.get_draft(replacement).snapshot().value.fields[
        "reps"
    ] == DirectValue(3)


def test_prepare_is_invisible_until_activation_and_retirement(service):
    original, _ = service.open_seeded(seed(1), owner_key="tab")
    old_draft = service.get_draft(original)
    events = []
    service.set_change_listener(
        lambda editor_id, event, payload: events.append((editor_id, event))
    )
    prepared = service.prepare_replacement("tab", seed(2))

    assert service.editor_id_for_owner("tab") == original
    assert service.snapshot_owner("tab").value.fields["reps"] == DirectValue(1)
    with pytest.raises(CfgEditorError):
        service.get_draft(prepared.editor_id)
    assert events == []

    retired = service.activate_replacement(prepared)
    assert retired is not None
    assert service.editor_id_for_owner("tab") == prepared.editor_id
    assert service.snapshot_owner("tab").value.fields["reps"] == DirectValue(2)
    with pytest.raises(CfgEditorError):
        service.set_field(original, "reps", 99)
    assert old_draft.snapshot().value.fields["reps"] == DirectValue(1)
    assert events == []

    service.retire_replaced(retired)
    service.retire_replaced(retired)
    assert events == [(original, "editor_closed")]
    with pytest.raises(RuntimeError):
        old_draft.snapshot()
    service.discard_prepared(prepared)
    service.set_field(prepared.editor_id, "reps", 3)
    assert service.snapshot_owner("tab").value.fields["reps"] == DirectValue(3)


def test_only_active_owner_publishes_complete_model_without_a_viewer(
    service, publications
):
    original, _ = service.open_seeded(seed(1), owner_key="tab")
    old_draft = service.get_draft(original)
    assert publications == [("tab", seed(1))]
    service.set_field(original, "reps", 3)
    assert publications[-1] == ("tab", seed(3))
    assert publications[0] == ("tab", seed(1))
    publications.clear()

    prepared = service.prepare_replacement("tab", seed(2))
    assert publications == []
    retired = service.activate_replacement(prepared)
    assert retired is not None
    assert publications == []
    old_draft.root.fields["reps"].set_value(99)
    assert publications == []
    service.set_field(prepared.editor_id, "reps", 4)
    assert publications == [("tab", seed(4))]
    service.retire_replaced(retired)


def test_expression_refresh_publishes_resolved_and_error_state(
    service, publications, metadata
):
    metadata.count = 4
    schema = CfgSchema(
        spec=seed(1).spec,
        value=CfgSectionValue(fields={"reps": EvalValue("count")}),
    )
    service.open_seeded(schema, owner_key="tab")
    first = publications[-1][1].value.fields["reps"]
    assert isinstance(first, EvalValue)
    assert first.resolved == 4
    assert first.error is None

    metadata.count = 7
    service.refresh_expressions()
    refreshed = publications[-1][1].value.fields["reps"]
    assert isinstance(refreshed, EvalValue)
    assert refreshed.resolved == 7
    assert first.resolved == 4

    metadata.count = "invalid"
    service.refresh_expressions()
    invalid = publications[-1][1].value.fields["reps"]
    assert isinstance(invalid, EvalValue)
    assert invalid.resolved is None
    assert invalid.error is not None


def test_discard_prepared_preserves_live_editor(service):
    original, _ = service.open_seeded(seed(1), owner_key="tab")
    prepared = service.prepare_replacement("tab", seed(2))
    service.discard_prepared(prepared)
    service.discard_prepared(prepared)

    with pytest.raises(ValueError, match="consumed"):
        service.activate_replacement(prepared)
    assert service.editor_id_for_owner("tab") == original
    service.set_field(original, "reps", 3)
    assert service.snapshot_owner("tab").value.fields["reps"] == DirectValue(3)


def test_prepare_failure_keeps_live_draft(service):
    original, _ = service.open_seeded(seed(1), owner_key="tab")
    invalid = CfgSchema(
        spec=seed(1).spec,
        value=CfgSectionValue(fields={"reps": DirectValue("not an integer")}),
    )
    with pytest.raises(TypeError):
        service.prepare_replacement("tab", invalid)
    assert service.editor_id_for_owner("tab") == original
    service.set_field(original, "reps", 3)
    assert service.snapshot_owner("tab").value.fields["reps"] == DirectValue(3)


def test_prepare_rejects_invalid_complete_draft_without_revoking_owner(service):
    original, _ = service.open_seeded(seed(1), owner_key="tab")
    invalid = CfgSchema(
        spec=seed(1).spec,
        value=CfgSectionValue(fields={"reps": DirectValue(None)}),
    )

    with pytest.raises(ValueError, match="invalid"):
        service.prepare_replacement("tab", invalid)
    assert service.editor_id_for_owner("tab") == original
    assert service.get_draft(original).is_valid()
    assert service.snapshot_owner("tab") == seed(1)


def test_replacement_can_create_a_headless_owner(service):
    assert service.snapshot_owner("tab") is None
    prepared = service.prepare_replacement("tab", seed(2))
    assert service.activate_replacement(prepared) is None
    assert service.editor_id_for_owner("tab") == prepared.editor_id
    service.set_field(prepared.editor_id, "reps", 4)
    assert service.snapshot_owner("tab").value.fields["reps"] == DirectValue(4)


def test_owner_change_rejects_pending_replacement(service):
    service.open_seeded(seed(1), owner_key="tab")
    prepared = service.prepare_replacement("tab", seed(2))
    latest, _ = service.open_seeded(seed(3), owner_key="tab")
    try:
        with pytest.raises(ValueError, match="owner changed"):
            service.activate_replacement(prepared)
        assert service.editor_id_for_owner("tab") == latest
        assert service.snapshot_owner("tab").value.fields["reps"] == DirectValue(3)
    finally:
        service.discard_prepared(prepared)


def test_later_reopen_does_not_revive_retired_handles(service):
    original, _ = service.open_seeded(seed(1), owner_key="tab")
    second, _ = service.open_seeded(seed(2), owner_key="tab")
    service.teardown(second)
    latest, _ = service.open_seeded(seed(3), owner_key="tab")

    for retired in (original, second):
        with pytest.raises(CfgEditorError):
            service.set_field(retired, "reps", 99)
    assert service.get_draft(latest).snapshot().value.fields["reps"] == DirectValue(3)
