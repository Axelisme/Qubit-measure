"""Independent inspect/library draft lifetime remains separate from tab cfg."""

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
    return CfgEditorService(host, host, host, host, BaseEventBus())


def seed(value: int) -> CfgSchema:
    return CfgSchema(
        CfgSectionSpec(fields={"reps": ScalarSpec("Reps", int)}),
        CfgSectionValue(fields={"reps": DirectValue(value)}),
    )


def test_reopen_owner_revokes_old_draft_and_handle(service: CfgEditorService) -> None:
    original, _ = service.open_seeded(seed(1), owner_key="inspect-owner")
    draft = service.get_draft(original)
    replacement, _ = service.open_seeded(seed(2), owner_key="inspect-owner")
    assert service.editor_id_for_owner("inspect-owner") == replacement
    with pytest.raises(CfgEditorError):
        service.set_field(original, "reps", 99)
    with pytest.raises(RuntimeError):
        draft.snapshot()
    assert service.get_draft(replacement).snapshot().value.fields[
        "reps"
    ] == DirectValue(2)


def test_later_reopen_does_not_revive_retired_handles(
    service: CfgEditorService,
) -> None:
    original, _ = service.open_seeded(seed(1), owner_key="inspect-owner")
    second, _ = service.open_seeded(seed(2), owner_key="inspect-owner")
    service.teardown(second)
    latest, _ = service.open_seeded(seed(3), owner_key="inspect-owner")
    for retired in (original, second):
        with pytest.raises(CfgEditorError):
            service.set_field(retired, "reps", 99)
    assert service.get_draft(latest).snapshot().value.fields["reps"] == DirectValue(3)
