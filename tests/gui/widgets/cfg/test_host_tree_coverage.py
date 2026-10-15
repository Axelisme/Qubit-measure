"""Resource cfg rendering through shared Qt widgets."""

from __future__ import annotations

from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    ScalarSpec,
)


def test_resource_form_shows_device_name_preview_and_unbound_diagnostic(qapp) -> None:
    from qtpy.QtWidgets import QLabel, QLineEdit
    from zcu_tools.gui.app.measure.cfg_binding import MeasureCfgBindings
    from zcu_tools.gui.app.measure.state import State
    from zcu_tools.gui.cfg import EvalValue
    from zcu_tools.gui.cfg.resource import CfgResource
    from zcu_tools.gui.session.aliases import AliasRow
    from zcu_tools.gui.session.types import SessionEnv
    from zcu_tools.gui.widgets.cfg.resource_form import ResourceCfgFormWidget

    from tests.gui.app.measure._cfg_fakes import PublishedHost, parameter_capture

    base = parameter_capture()
    current = base.bind_aliases((AliasRow("flux_dev", "yoko1"),), {})
    state = State(SessionEnv(parameters=current, soc=None, soccfg=None))
    bindings = MeasureCfgBindings(PublishedHost(state))
    schema = CfgSchema(
        CfgSectionSpec(
            fields={"flux_dev": ScalarSpec("Flux", str, choices_source="devices")}
        ),
        CfgSectionValue(fields={"flux_dev": EvalValue("$flux_dev")}),
    )
    resource = CfgResource(
        lambda: schema,
        resolution=lambda: bindings.snapshot(
            (), captured_values={}, parameters=current
        ),
    )
    form = ResourceCfgFormWidget()
    try:
        form.attach(resource)
        assert form.is_valid()
        assert any(label.text() == '= "yoko1"' for label in form.findChildren(QLabel))
        line = next(
            line for line in form.findChildren(QLineEdit) if line.text() == "$flux_dev"
        )
        line.setText("$unknown")
        assert not form.is_valid()
        assert "unknown" in (form.first_invalid_reason() or "")
        line.setText("$flux_dev")
        assert form.is_valid()
        current = base.bind_aliases((AliasRow("flux_dev", "yoko2"),), {})
        resource.refresh()
        assert any(label.text() == '= "yoko2"' for label in form.findChildren(QLabel))
    finally:
        form.detach()
        resource.revoke()
        form.deleteLater()
        qapp.processEvents()
