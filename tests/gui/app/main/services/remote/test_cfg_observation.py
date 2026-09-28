"""Cfg observation projection of a tab draft for remote readers.

``build_cfg_observation`` turns a live ``CfgDraft`` into the JSON tree that
``tab.get_cfg`` and ``editor.get`` publish: cached values, invalid input text,
reference and centered-range state, and lossless-encoding rejection.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.cfg import CfgSchema, CfgSectionSpec, CfgSectionValue, DirectValue
from zcu_tools.gui.cfg.binding import CfgDraft


def test_cfg_observation_projects_cached_values_and_invalid_text() -> None:
    import json

    from zcu_tools.gui.app.main.services.remote.cfg_observation import (
        build_cfg_observation,
    )
    from zcu_tools.gui.cfg import (
        EvalValue,
        LiteralSpec,
        ScalarSpec,
        SweepSpec,
        make_default_value,
    )
    from zcu_tools.gui.cfg.binding import ScalarField

    spec = CfgSectionSpec(
        fields={
            "fixed": LiteralSpec({"center": 1 + 2j}),
            "frequency": ScalarSpec("Frequency", float, editable=False),
            "count": ScalarSpec("Count", int, required=True),
            "axis": SweepSpec(),
        }
    )
    value = make_default_value(spec).with_field("frequency", EvalValue("frequency"))
    evaluator = MagicMock(return_value=2.5)
    draft = CfgDraft(
        CfgSchema(spec, value),
        evaluate_expression=evaluator,
        provide_options=lambda source_id: (),
        references=MagicMock(),
    )
    try:
        count = draft.root.fields["count"]
        assert isinstance(count, ScalarField)
        count.set_text("1e")
        evaluator.side_effect = AssertionError("A read must not resolve expressions")
        tree = json.loads(json.dumps(build_cfg_observation(draft), allow_nan=False))
        assert tree["kind"] == "section" and not tree["valid"]
        fields = tree["children"]
        assert fields["fixed"]["input"]["resolved"] == {
            "center": {"__complex__": [1.0, 2.0]}
        }
        assert not fields["frequency"]["editable"]
        assert fields["frequency"]["input"] == {
            "mode": "expression",
            "raw": "frequency",
            "resolved": 2.5,
            "error": None,
            "validation_error": None,
        }
        assert fields["count"]["input"]["raw"] == "1e"
        assert fields["count"]["input"]["resolved"] is None
        assert fields["count"]["input"]["error"] is not None
        assert build_cfg_observation(draft, "count") == fields["count"]
        assert build_cfg_observation(draft, "axis.start") == fields["axis"]
        assert build_cfg_observation(draft, "missing") == {}
    finally:
        draft.close()


@pytest.mark.parametrize(
    "value", [object(), {1: "ambiguous key"}, float("inf"), complex(1, float("nan"))]
)
def test_cfg_observation_rejects_values_without_lossless_json_encoding(
    value: object,
) -> None:
    from zcu_tools.gui.app.main.services.remote.cfg_observation import (
        build_cfg_observation,
    )
    from zcu_tools.gui.cfg import LiteralSpec, make_default_value

    spec = CfgSectionSpec(fields={"fixed": LiteralSpec(value)})
    draft = CfgDraft(
        CfgSchema(spec, make_default_value(spec)),
        evaluate_expression=MagicMock(),
        provide_options=lambda source_id: (),
        references=MagicMock(),
    )
    try:
        with pytest.raises((TypeError, ValueError), match="observation"):
            build_cfg_observation(draft)
    finally:
        draft.close()


def test_cfg_observation_projects_reference_and_centered_range_state() -> None:
    from zcu_tools.gui.app.main.services.remote.cfg_observation import (
        build_cfg_observation,
    )
    from zcu_tools.gui.cfg import (
        CenteredSweepSpec,
        ReferenceSpec,
        ReferenceValue,
        ScalarSpec,
        make_custom_reference_key,
        make_default_value,
    )
    from zcu_tools.gui.cfg.binding import CenteredSweepField, ReferenceField

    shape = CfgSectionSpec(label="Pulse", fields={"gain": ScalarSpec("Gain", float)})
    spec = CfgSectionSpec(
        fields={
            "drive": ReferenceSpec("module", [shape], optional=True),
            "axis": CenteredSweepSpec(center_editable=False, locked_center=0.0),
        }
    )
    value = make_default_value(spec)
    value.fields["drive"] = ReferenceValue(
        make_custom_reference_key("Pulse"), CfgSectionValue({"gain": DirectValue(0.25)})
    )
    catalog = MagicMock()
    catalog.keys.return_value = ()
    draft = CfgDraft(
        CfgSchema(spec, value),
        evaluate_expression=MagicMock(),
        provide_options=lambda source_id: (),
        references=catalog,
    )
    try:
        axis = draft.root.fields["axis"]
        assert isinstance(axis, CenteredSweepField)
        axis.set_text("span", "1e")
        tree = build_cfg_observation(draft)
        children = tree["children"]
        assert isinstance(children, dict)
        assert children["axis"]["kind"] == "centered_sweep"
        assert not children["axis"]["center_editable"]
        assert children["axis"]["locked_center"] == 0.0
        assert children["axis"]["inputs"]["span"]["raw"] == "1e"
        assert children["axis"]["inputs"]["span"]["error"] is not None
        drive = children["drive"]
        assert drive["resolved_label"] == "Pulse"
        assert drive["choices"] == ["Pulse"]
        assert drive["children"]["gain"]["input"]["resolved"] == 0.25
        assert build_cfg_observation(draft, "drive.ref") == drive
        reference = draft.root.fields["drive"]
        assert isinstance(reference, ReferenceField)
        reference.set_enabled(False)
        disabled = build_cfg_observation(draft, "drive")
        assert disabled["ref"] is None
        assert disabled["children"] == {}
        assert disabled["valid"]
    finally:
        draft.close()
