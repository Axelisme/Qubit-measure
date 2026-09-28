"""Non-finite scalar edits remain visible and invalid in the full cfg form."""

from unittest.mock import MagicMock

import pytest
from qtpy.QtWidgets import QLineEdit
from zcu_tools.gui.app.main.cfg_binding import MeasureCfgBindings
from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    ScalarSpec,
)
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.widgets.cfg import CfgFormWidget


@pytest.mark.parametrize(
    ("type_", "text", "valid_text", "valid_value"),
    [
        (float, "nan", "1.5", 1.5),
        (float, "inf", "1.5", 1.5),
        (complex, "nan+2j", "1+2j", 1 + 2j),
        (complex, "1+infj", "1+2j", 1 + 2j),
    ],
)
def test_form_preserves_nonfinite_scalar_error_until_corrected(
    qapp, type_: type, text: str, valid_text: str, valid_value: float | complex
) -> None:
    ctrl = MagicMock()
    ctrl.get_bus.return_value = BaseEventBus()
    ctrl.get_current_md.return_value = MagicMock()
    ctrl.get_current_ml.return_value = MagicMock()
    ctrl.list_arb_waveforms.return_value = []
    schema = CfgSchema(
        spec=CfgSectionSpec(
            fields={"value": ScalarSpec("Value", type_, optional=True)}
        ),
        value=CfgSectionValue(fields={"value": DirectValue(valid_value)}),
    )
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    widget = CfgFormWidget()
    widget.attach(draft)
    try:
        entry = widget.findChild(QLineEdit)
        assert entry is not None
        entry.setText(text)
        observed = widget.read_values().fields["value"]
        assert isinstance(observed, DirectValue)
        assert observed.raw == text
        assert observed.value is None
        assert observed.error is not None
        assert not draft.is_valid()

        entry.setText(valid_text)
        assert widget.read_values().fields["value"] == DirectValue(
            valid_value, raw=valid_text
        )
        assert draft.is_valid()
    finally:
        widget.detach()
        widget.close()
        widget.deleteLater()
        draft.root.teardown()
