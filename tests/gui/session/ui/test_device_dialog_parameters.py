"""DeviceDialog apply-time expression, direct-value and validation contracts."""

from __future__ import annotations

import pytest
from qtpy.QtWidgets import QLabel
from zcu_tools.gui.session.device_models import SetupDeviceRequest
from zcu_tools.gui.session.ui.eval_field import EvalNumericField

from tests.gui.session.ui._device_dialog_support import (
    apply_button,
    connected_snapshot,
    device_entry,
    device_list,
    device_panel,
    labelled_field,
    make_device_control,
    make_device_dialog,
    make_parameter_control,
)


def test_eval_apply_resolves_expression_to_float(qapp):
    """Eval mode: apply resolves the expression to a concrete float and passes it
    to start_setup_device. The resolved info.value must be a float, not an EvalRef."""
    from zcu_tools.device.fake import FakeDeviceInfo

    ctrl = make_parameter_control(flx_int=0.5)
    ctrl.list_devices.return_value = [device_entry("fd")]
    info = FakeDeviceInfo(address="none")
    ctrl.get_device_snapshot.return_value = connected_snapshot("fd", info)
    ctrl.get_device_info.return_value = info

    dialog = make_device_dialog(ctrl)
    device_list(dialog).setCurrentRow(0)

    panel = device_panel(dialog)
    # Switch value field to eval mode and type an expression
    labelled_field(panel, "Value:", EvalNumericField).load_expression(
        "general.ext.flx_int"
    )
    ctrl.read_context.reset_mock()

    apply_button(dialog).click()

    ctrl.read_context.assert_called_once()
    ctrl.start_setup_device.assert_called_once()
    (req,) = ctrl.start_setup_device.call_args.args
    assert isinstance(req, SetupDeviceRequest)
    assert req.name == "fd"
    assert isinstance(req.info, FakeDeviceInfo)
    assert req.info.value == pytest.approx(0.5)
    assert isinstance(req.info.value, float)


def test_eval_apply_fast_fails_on_undefined_name(qapp):
    """An undefined expression displays an error without submitting setup."""
    from zcu_tools.device.fake import FakeDeviceInfo

    ctrl = make_parameter_control()
    ctrl.list_devices.return_value = [device_entry("fd")]
    info = FakeDeviceInfo(address="none")
    ctrl.get_device_snapshot.return_value = connected_snapshot("fd", info)
    ctrl.get_device_info.return_value = info

    dialog = make_device_dialog(ctrl)
    device_list(dialog).setCurrentRow(0)

    panel = device_panel(dialog)
    labelled_field(panel, "Value:", EvalNumericField).load_expression("missing_var")

    apply_button(dialog).click()

    ctrl.start_setup_device.assert_not_called()
    assert any("fd:" in label.text() for label in dialog.findChildren(QLabel))


def test_eval_apply_fast_fails_on_syntax_error(qapp):
    """Eval mode: apply with a syntax-error expression fast-fails."""
    from zcu_tools.device.fake import FakeDeviceInfo

    ctrl = make_parameter_control()
    ctrl.list_devices.return_value = [device_entry("fd")]
    info = FakeDeviceInfo(address="none")
    ctrl.get_device_snapshot.return_value = connected_snapshot("fd", info)
    ctrl.get_device_info.return_value = info

    dialog = make_device_dialog(ctrl)
    device_list(dialog).setCurrentRow(0)

    panel = device_panel(dialog)
    labelled_field(panel, "Value:", EvalNumericField).load_expression("r_f[")

    apply_button(dialog).click()

    ctrl.start_setup_device.assert_not_called()


def test_eval_apply_no_context_fast_fails(qapp):
    """When the parameter provider raises (no active context), apply fast-fails."""
    from zcu_tools.device.fake import FakeDeviceInfo

    ctrl = make_device_control()
    ctrl.read_context.side_effect = RuntimeError("no active context")
    ctrl.list_devices.return_value = [device_entry("fd")]
    info = FakeDeviceInfo(address="none")
    ctrl.get_device_snapshot.return_value = connected_snapshot("fd", info)
    ctrl.get_device_info.return_value = info

    dialog = make_device_dialog(ctrl)
    device_list(dialog).setCurrentRow(0)

    panel = device_panel(dialog)
    labelled_field(panel, "Value:", EvalNumericField).load_expression(
        "general.ext.flx_int"
    )

    apply_button(dialog).click()

    ctrl.start_setup_device.assert_not_called()


def test_direct_mode_apply_unchanged(qapp):
    """Direct mode (no eval): apply behaviour is identical to the pre-eval baseline."""
    from zcu_tools.device.fake import FakeDeviceInfo

    ctrl = make_parameter_control()
    ctrl.list_devices.return_value = [device_entry("fd")]
    info = FakeDeviceInfo(address="none", value=3.14)
    ctrl.get_device_snapshot.return_value = connected_snapshot("fd", info)
    ctrl.get_device_info.return_value = info

    dialog = make_device_dialog(ctrl)
    device_list(dialog).setCurrentRow(0)

    apply_button(dialog).click()

    ctrl.start_setup_device.assert_called_once()
    (req,) = ctrl.start_setup_device.call_args.args
    assert isinstance(req, SetupDeviceRequest)
    assert req.name == "fd"
    assert isinstance(req.info, FakeDeviceInfo)
    # value should be the float passed through unchanged
    assert req.info.value == pytest.approx(3.14)
    ctrl.read_context.assert_not_called()


def test_eval_apply_out_of_range_fast_fails(qapp):
    """Eval mode: rampstep expression resolves to 0 (below minimum=1e-9) → fast-fail.

    This is the concrete ZeroDivisionError guard: eval must reject rampstep=0 at
    apply time before the value ever reaches the driver's ramp logic.
    """
    from zcu_tools.device.fake import FakeDeviceInfo

    ctrl = make_parameter_control(flx_int=0.0)  # flx_int evaluates to 0.0
    ctrl.list_devices.return_value = [device_entry("fd")]
    info = FakeDeviceInfo(address="none")
    ctrl.get_device_snapshot.return_value = connected_snapshot("fd", info)
    ctrl.get_device_info.return_value = info

    dialog = make_device_dialog(ctrl)
    device_list(dialog).setCurrentRow(0)

    panel = device_panel(dialog)
    # rampstep_field has minimum=1e-9; expression evaluates to 0.0 → below minimum
    labelled_field(panel, "Ramp step:", EvalNumericField).load_expression(
        "general.ext.flx_int"
    )

    apply_button(dialog).click()

    ctrl.start_setup_device.assert_not_called()
    assert any("rampstep" in label.text() for label in dialog.findChildren(QLabel))


def test_eval_apply_in_range_sends_setup(qapp):
    """Eval mode: rampstep expression resolves within [1e-9, 1e9] → setup dispatched."""
    from zcu_tools.device.fake import FakeDeviceInfo

    ctrl = make_parameter_control(flx_int=0.001)  # 0.001 is within [1e-9, 1e9]
    ctrl.list_devices.return_value = [device_entry("fd")]
    info = FakeDeviceInfo(address="none")
    ctrl.get_device_snapshot.return_value = connected_snapshot("fd", info)
    ctrl.get_device_info.return_value = info

    dialog = make_device_dialog(ctrl)
    device_list(dialog).setCurrentRow(0)

    panel = device_panel(dialog)
    labelled_field(panel, "Ramp step:", EvalNumericField).load_expression(
        "general.ext.flx_int"
    )

    apply_button(dialog).click()

    ctrl.start_setup_device.assert_called_once()
    (req,) = ctrl.start_setup_device.call_args.args
    assert isinstance(req, SetupDeviceRequest)
    assert req.name == "fd"
    assert isinstance(req.info, FakeDeviceInfo)
    assert req.info.rampstep == pytest.approx(0.001)


def test_choice_field_unaffected_by_eval_mode(qapp):
    """The 'output' combo (choice field) is never wrapped in EvalRef regardless
    of whether any numeric field is in eval mode."""
    from zcu_tools.device.fake import FakeDeviceInfo

    ctrl = make_parameter_control()
    ctrl.list_devices.return_value = [device_entry("fd")]
    info = FakeDeviceInfo(address="none", output="on")
    ctrl.get_device_snapshot.return_value = connected_snapshot("fd", info)
    ctrl.get_device_info.return_value = info

    dialog = make_device_dialog(ctrl)
    device_list(dialog).setCurrentRow(0)

    panel = device_panel(dialog)
    labelled_field(panel, "Value:", EvalNumericField).load_expression(
        "general.ext.flx_int"
    )

    apply_button(dialog).click()

    ctrl.start_setup_device.assert_called_once()
    (req,) = ctrl.start_setup_device.call_args.args
    assert req.info.output == "on"  # choice field passes through as plain string
