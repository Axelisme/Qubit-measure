"""Context/value remote handlers dispatch through ContextControlPort."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from zcu_tools.gui.session.value_lookup import ValueInfo

from ._helpers import dispatch_handler as _dispatch


def _ctrl_with_context_control(context_control: MagicMock) -> SimpleNamespace:
    ctrl = SimpleNamespace(
        context_control=context_control, resources_versions=lambda: {}
    )
    for name in (
        "use_context",
        "new_context",
        "get_current_md",
        "get_current_ml",
        "set_md_attr",
        "del_md_attr",
        "rename_ml_module",
        "rename_ml_waveform",
        "del_ml_module",
        "del_ml_waveform",
        "list_value_sources",
        "read_value_source",
    ):
        setattr(ctrl, name, MagicMock())
    return ctrl


def test_context_switch_and_create_use_context_control_facet() -> None:
    ctx = MagicMock()
    ctx.has_project.return_value = True
    ctx.get_context_labels.return_value = ["base"]
    ctx.get_active_context_label.return_value = "base"
    ctrl = _ctrl_with_context_control(ctx)

    assert _dispatch(ctrl, "context.use", {"label": "base"}) == {
        "label": "base",
        "has_active_context": True,
        "__agent_write_versions": {},
    }
    ctx.use_context.assert_called_once_with("base")
    ctrl.use_context.assert_not_called()

    assert _dispatch(
        ctrl, "context.new", {"bind_device": None, "clone_from": None}
    ) == {
        "label": "base",
        "has_active_context": True,
        "__agent_write_versions": {},
    }
    ctx.new_context.assert_called_once_with(bind_device=None, clone_from=None)
    ctrl.new_context.assert_not_called()


def test_context_md_ml_and_value_handlers_use_context_control_facet() -> None:
    ctx = MagicMock()
    ctx.get_current_md.return_value = {"r_f": 6000.0}
    ctx.get_current_ml.return_value.modules = {"readout": SimpleNamespace(type="pulse")}
    ctx.get_current_ml.return_value.waveforms = {
        "drive": SimpleNamespace(style="const")
    }
    ctx.list_value_sources.return_value = (
        ValueInfo("predictor.loaded", bool, "predictor", "Whether loaded."),
    )
    ctx.read_value_source.return_value = (
        ValueInfo("predictor.loaded", bool, "predictor", "Whether loaded."),
        True,
    )
    ctrl = _ctrl_with_context_control(ctx)

    assert _dispatch(ctrl, "context.md_get", {}) == {"keys": ["r_f"]}
    assert _dispatch(ctrl, "context.md_get_attr", {"key": "r_f"}) == {
        "key": "r_f",
        "value": 6000.0,
    }
    library_index = _dispatch(ctrl, "context.ml_get", {})
    assert set(library_index) == {"modules", "waveforms"}
    modules = library_index["modules"]
    waveforms = library_index["waveforms"]
    assert isinstance(modules, list) and len(modules) == 1
    assert isinstance(waveforms, list) and len(waveforms) == 1
    module, waveform = modules[0], waveforms[0]
    assert isinstance(module, dict) and isinstance(waveform, dict)
    assert {key: module[key] for key in ("name", "kind")} == {
        "name": "readout",
        "kind": "pulse",
    }
    assert {key: waveform[key] for key in ("name", "style")} == {
        "name": "drive",
        "style": "const",
    }
    assert all(
        isinstance(item["description"], str) and item["description"]
        for item in (module, waveform)
    )
    assert _dispatch(ctrl, "value.list", {}) == {
        "values": [
            {
                "key": "predictor.loaded",
                "type": "bool",
                "owner": "predictor",
                "description": "Whether loaded.",
            }
        ]
    }
    assert _dispatch(ctrl, "value.read", {"key": "predictor.loaded"}) == {
        "key": "predictor.loaded",
        "type": "bool",
        "owner": "predictor",
        "description": "Whether loaded.",
        "value": True,
    }

    _dispatch(ctrl, "context.md_set_attr", {"key": "r_f", "value": 6100.0})
    _dispatch(ctrl, "context.md_del_attr", {"key": "r_f"})
    _dispatch(ctrl, "context.ml_rename_module", {"old": "readout", "new": "ro"})
    _dispatch(ctrl, "context.ml_rename_waveform", {"old": "drive", "new": "drv"})
    _dispatch(ctrl, "context.ml_del_module", {"name": "ro"})
    _dispatch(ctrl, "context.ml_del_waveform", {"name": "drv"})

    ctx.set_md_attr.assert_called_once_with("r_f", 6100.0)
    ctx.del_md_attr.assert_called_once_with("r_f")
    ctx.rename_ml_module.assert_called_once_with("readout", "ro")
    ctx.rename_ml_waveform.assert_called_once_with("drive", "drv")
    ctx.del_ml_module.assert_called_once_with("ro")
    ctx.del_ml_waveform.assert_called_once_with("drv")
    ctrl.set_md_attr.assert_not_called()
    ctrl.del_md_attr.assert_not_called()
    ctrl.rename_ml_module.assert_not_called()
    ctrl.rename_ml_waveform.assert_not_called()
    ctrl.del_ml_module.assert_not_called()
    ctrl.del_ml_waveform.assert_not_called()
