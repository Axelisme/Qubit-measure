from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.main.services.cfg_editor import CfgEditorError, CfgEditorService
from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    ScalarSpec,
)
from zcu_tools.gui.event_bus import BaseEventBus


@pytest.fixture
def service():
    host = MagicMock()
    host.get_current_md.return_value = {}
    host.get_current_ml.return_value = None
    host.list_device_names.return_value = []
    host.list_arb_waveforms.return_value = []
    svc = CfgEditorService(
        host,
        read_port=host,
        write_port=host,
        version_bump=host.bump_editor_version,
        version_drop=host.drop_editor_version,
        bus=BaseEventBus(),
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


def test_later_reopen_does_not_revive_retired_handles(service):
    original, _ = service.open_seeded(seed(1), owner_key="tab")
    second, _ = service.open_seeded(seed(2), owner_key="tab")
    service.teardown(second)
    latest, _ = service.open_seeded(seed(3), owner_key="tab")

    for retired in (original, second):
        with pytest.raises(CfgEditorError):
            service.set_field(retired, "reps", 99)
    assert service.get_draft(latest).snapshot().value.fields["reps"] == DirectValue(3)
