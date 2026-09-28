"""Shared builders for cfg form widget tests."""

from __future__ import annotations

from unittest.mock import MagicMock

from zcu_tools.gui.app.main.cfg_binding import MeasureCfgBindings
from zcu_tools.gui.cfg import (
    CfgNodeSpec,
    CfgNodeValue,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    ScalarSpec,
)
from zcu_tools.gui.cfg.binding import ScalarField


def section_schema(
    spec_fields: dict[str, CfgNodeSpec], value_fields: dict[str, CfgNodeValue | None]
) -> CfgSchema:
    return CfgSchema(
        spec=CfgSectionSpec(fields=spec_fields),
        value=CfgSectionValue(fields=value_fields),
    )


def attach_draft(w, schema: CfgSchema, ctrl):
    """Build a caller-owned draft, attach the widget, and return its root field."""
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    w.attach(draft)
    return draft.root


def scalar_field(ctrl: MagicMock, spec: ScalarSpec, initial_val: object) -> ScalarField:
    bindings = MeasureCfgBindings(ctrl)
    return ScalarField(
        spec,
        bindings.evaluate_expression,
        bindings.provide_options,
        initial_val,
    )
