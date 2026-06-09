"""Headless SetupDialog tests — mock connects a MockSoc; real builds a SetupRequest.

The mock path is exercised end-to-end (OK → MockSoc into the active exp_context).
The real path is NOT connected here (``make_soc_proxy`` would block on a network
timeout); instead the dialog's ``_build_request`` is asserted to carry the
fields, and the mock-toggle's enable/disable of the remote groups is checked.
"""

from __future__ import annotations

from zcu_tools.gui.app.autofluxdep.app import build_core
from zcu_tools.gui.app.autofluxdep.ui.setup_dialog import SetupDialog


def test_setup_dialog_ok_connects_mock_soc(qapp):
    ctrl = build_core()
    dlg = SetupDialog(ctrl)
    assert dlg._mock.isChecked()  # mock is the default path
    dlg._accept()  # simulate OK without exec()
    assert ctrl.state.has_setup
    ctx = ctrl.state.exp_context
    assert ctx.soc is not None and ctx.soccfg is not None
    assert "MockQickSoc" in type(ctx.soc).__name__
    dlg.deleteLater()


def test_mock_toggle_enables_remote_groups(qapp):
    ctrl = build_core()
    dlg = SetupDialog(ctrl)
    # mock ticked → ZCU + flux-device groups disabled (informational)
    assert dlg._mock.isChecked()
    assert not dlg._zcu.isEnabled()
    assert not dlg._flux.isEnabled()
    # untick → both enabled for real-connection entry
    dlg._mock.setChecked(False)
    assert dlg._zcu.isEnabled()
    assert dlg._flux.isEnabled()
    dlg.deleteLater()


def test_build_request_carries_real_fields(qapp):
    ctrl = build_core()
    dlg = SetupDialog(ctrl)
    dlg._mock.setChecked(False)
    dlg._ip.setText("10.0.0.5")
    dlg._port.setValue(9000)
    dlg._flux_addr.setText("USB0::0x0B21::INSTR")
    dlg._params.setText("/tmp/params.json")
    req = dlg._build_request()
    assert req.use_mock is False
    assert req.ip == "10.0.0.5"
    assert req.port == 9000
    assert req.flux_device_address == "USB0::0x0B21::INSTR"
    assert req.params_path == "/tmp/params.json"
    dlg.deleteLater()


def test_real_setup_failure_keeps_dialog_open(qapp, monkeypatch):
    # a real connect that raises must not crash or close the dialog — the error
    # is shown inline and State stays un-setup so the user can fix and retry.
    ctrl = build_core()
    dlg = SetupDialog(ctrl)
    dlg._mock.setChecked(False)
    dlg._flux_addr.setText("dummy")

    def boom(_req):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(ctrl, "setup", boom)
    dlg._accept()
    assert not ctrl.state.has_setup  # setup did not complete
    assert "Setup failed" in dlg._status.text()  # error surfaced inline
    dlg.deleteLater()
