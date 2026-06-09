"""Controller.create_from_role: seed a blank ml entry from a named role,
md-linked defaults lowered to the md's current values.

Uses a real ExpContext (real MetaDict/ModuleLibrary) + a real RoleCatalog so the
factory → lowering → ml-register chain is exercised end to end.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.experiment.v2_gui.registry import register_all_roles
from zcu_tools.gui.app.main.adapter import ContextReadiness, ExpContext
from zcu_tools.gui.app.main.controller import Controller
from zcu_tools.gui.app.main.event_bus import EventBus
from zcu_tools.gui.app.main.registry import Registry
from zcu_tools.gui.app.main.role_catalog import RoleCatalog
from zcu_tools.gui.app.main.state import State
from zcu_tools.gui.session.services.io_manager import IOManager
from zcu_tools.meta_tool import MetaDict, ModuleLibrary


def _make_ctrl(md_values: dict) -> Controller:
    md = MetaDict()
    for k, v in md_values.items():
        setattr(md, k, v)
    ctx = ExpContext(
        md=md,
        ml=ModuleLibrary(),
        soc=None,
        soccfg=None,
        readiness=ContextReadiness.ACTIVE,
    )
    catalog = RoleCatalog()
    register_all_roles(catalog)
    io = IOManager()
    io._em = MagicMock()  # simulate a project being set up
    bus = EventBus()
    return Controller(
        state=State(ctx),
        registry=Registry(),
        io_manager=io,
        view=None,
        bus=bus,
        role_catalog=catalog,
    )


def test_create_module_from_role_uses_md_value(qapp):  # noqa: ARG001
    ctrl = _make_ctrl({"r_f": 6123.0, "res_ch": 1, "ro_ch": 2})
    ctrl.create_from_role("module", "res_probe", "my_ro")

    ml = ctrl.get_current_ml()
    assert "my_ro" in ml.modules
    raw = ml.modules["my_ro"].to_dict()
    # res_probe is a bare pulse: md-linked freq lowered to the md's current
    # value (not a structural 0.0).
    assert raw["freq"] == 6123.0


def test_create_module_from_role_empty_md_falls_back(qapp):  # noqa: ARG001
    ctrl = _make_ctrl({})
    ctrl.create_from_role("module", "res_probe", "ro_blank")

    raw = ctrl.get_current_ml().modules["ro_blank"].to_dict()
    # fallback literal (the factory's default), not a crash.
    assert raw["freq"] == 6000.0


def test_create_from_role_name_clash_fails(qapp):  # noqa: ARG001
    """Create is new-entry semantics: a name clash must fail fast, not silently
    overwrite an existing ml entry."""
    ctrl = _make_ctrl({"r_f": 6000.0})
    ctrl.create_from_role("module", "res_probe", "dup")
    with pytest.raises(RuntimeError, match="already exists"):
        ctrl.create_from_role("module", "qub_probe", "dup")
    # the original entry is untouched (not overwritten by the failed second call)
    assert ctrl.get_current_ml().modules["dup"].to_dict()["type"] == "pulse"


def test_create_waveform_from_role(qapp):  # noqa: ARG001
    ctrl = _make_ctrl({})
    ctrl.create_from_role("waveform", "res_waveform", "ro_wav")
    assert "ro_wav" in ctrl.get_current_ml().waveforms


def test_create_from_blank_module_role(qapp):  # noqa: ARG001
    """A ':blank' role creates a structural-zero entry of that exact shape."""
    ctrl = _make_ctrl({"r_f": 6000.0})
    ctrl.create_from_role("module", "reset/bath:blank", "rb")
    raw = ctrl.get_current_ml().modules["rb"].to_dict()
    assert raw["type"] == "reset/bath"


def test_create_from_blank_waveform_role_uncovered_style(qapp):  # noqa: ARG001
    """A waveform style with no md-aware role (drag) is reachable via :blank."""
    ctrl = _make_ctrl({})
    ctrl.create_from_role("waveform", "drag:blank", "dwav")
    raw = ctrl.get_current_ml().waveforms["dwav"].to_dict()
    assert raw["style"] == "drag"


def test_item_kind_mismatch_raises(qapp):  # noqa: ARG001
    ctrl = _make_ctrl({})
    with pytest.raises(RuntimeError, match="not a waveform"):
        ctrl.create_from_role("waveform", "res_probe", "x")


def test_unknown_role_raises(qapp):  # noqa: ARG001
    ctrl = _make_ctrl({})
    with pytest.raises(KeyError):
        ctrl.create_from_role("module", "no_such_role", "x")


def test_empty_name_raises(qapp):  # noqa: ARG001
    ctrl = _make_ctrl({})
    with pytest.raises(RuntimeError, match="name must not be empty"):
        ctrl.create_from_role("module", "res_probe", "")


def test_no_catalog_wired_raises(qapp):  # noqa: ARG001
    ctx = ExpContext(
        md=MetaDict(),
        ml=ModuleLibrary(),
        soc=None,
        soccfg=None,
        readiness=ContextReadiness.ACTIVE,
    )
    io = IOManager()
    io._em = MagicMock()
    ctrl = Controller(
        state=State(ctx),
        registry=Registry(),
        io_manager=io,
        view=None,
        bus=EventBus(),
    )
    with pytest.raises(RuntimeError, match="No role catalog"):
        ctrl.create_from_role("module", "res_probe", "x")
