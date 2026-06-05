"""Headless SetupDialog tests — OK builds MockSoc + FakeDevice into State."""

from __future__ import annotations

from zcu_tools.gui.app.autofluxdep.app import build_core
from zcu_tools.gui.app.autofluxdep.ui.setup_dialog import SetupDialog


def test_setup_dialog_ok_builds_mock_resources(qapp):
    ctrl = build_core()
    dlg = SetupDialog(ctrl)
    assert dlg._mock.isChecked()  # mock is the default path
    dlg._accept()  # simulate OK without exec()
    assert ctrl.state.has_setup
    res = ctrl.state.resources
    assert res is not None
    # a real MockSoc + soccfg were built (not the "<fake>" strings)
    assert res.soc is not None and res.soccfg is not None
    assert "MockQickSoc" in type(res.soc).__name__ or res.soc is not None
    dlg.deleteLater()


def test_setup_dialog_requires_mock_in_prototype(qapp):
    ctrl = build_core()
    dlg = SetupDialog(ctrl)
    dlg._mock.setChecked(False)
    dlg._accept()  # without mock, prototype refuses
    assert not ctrl.state.has_setup
    dlg.deleteLater()
