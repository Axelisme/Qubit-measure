"""CfgEditorService — headless ml editing sessions.

Drives the service directly (no wire) against a MagicMock controller backed by
a real MetaDict + ModuleLibrary, so we exercise the full open → set_field →
commit path including eval-on-commit and ModuleRef sub-tree re-binding.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.event_bus import EventBus
from zcu_tools.gui.services.cfg_editor import CfgEditorError, CfgEditorService
from zcu_tools.meta_tool import MetaDict, ModuleLibrary


@pytest.fixture()
def ml():
    return ModuleLibrary()


@pytest.fixture()
def md():
    m = MetaDict()
    m.r_f = 6012.5
    return m


@pytest.fixture()
def ctrl(ml, md):
    """MagicMock controller whose set_ml_*_from_raw lands into the real ml."""
    from zcu_tools.program.v2 import ModuleCfgFactory, WaveformCfgFactory

    c = MagicMock()
    c.get_bus.return_value = EventBus()
    c.get_current_md.return_value = md
    c.get_current_ml.return_value = ml
    c.list_device_names.return_value = []
    c.has_soc.return_value = False

    def _set_module(name, raw):
        ml.register_module(**{name: ModuleCfgFactory.from_raw(raw, ml=ml)})

    def _set_waveform(name, raw):
        ml.register_waveform(**{name: WaveformCfgFactory.from_raw(raw, ml=ml)})

    c.set_ml_module_from_raw.side_effect = _set_module
    c.set_ml_waveform_from_raw.side_effect = _set_waveform
    return c


@pytest.fixture()
def service(ctrl):
    return CfgEditorService(ctrl)


def _paths(entries):
    return {e["path"]: e for e in entries}


# ---------------------------------------------------------------------------
# open
# ---------------------------------------------------------------------------


def test_open_blank_module_returns_paths(service):
    editor_id, paths = service.open("module", discriminator="pulse")
    assert editor_id.startswith("editor-")
    keys = _paths(paths)
    assert "freq" in keys
    assert keys["freq"]["kind"] == "scalar"


def test_open_unknown_kind_fails(service):
    with pytest.raises(CfgEditorError):
        service.open("widget", discriminator="pulse")


def test_open_needs_discriminator_or_from_name(service):
    with pytest.raises(CfgEditorError):
        service.open("module")


def test_open_unknown_module_type_fails(service):
    with pytest.raises(CfgEditorError):
        service.open("module", discriminator="nonsense")


def test_open_from_existing_entry_returns_concrete_values(service, ml):
    # Seed an entry, then open it for editing.
    editor_id, _ = service.open("waveform", discriminator="const")
    service.commit(editor_id, "wf_seed")
    assert "wf_seed" in ml.waveforms

    _, paths = service.open("waveform", from_name="wf_seed")
    keys = _paths(paths)
    assert "length" in keys
    # scalar comes back as a concrete value, not an EvalValue/dict.
    assert not isinstance(keys["length"]["value"], dict)


# ---------------------------------------------------------------------------
# set_field + commit (eval-on-commit)
# ---------------------------------------------------------------------------


def test_set_scalar_then_commit_lands_concrete(service, ctrl, ml):
    editor_id, _ = service.open("module", discriminator="pulse")
    res = service.set_field(editor_id, "freq", 5000.0)
    assert res["valid"] is True
    service.commit(editor_id, "agent_pulse")

    ctrl.set_ml_module_from_raw.assert_called_once()
    assert "agent_pulse" in ml.modules
    assert ml.modules["agent_pulse"].to_dict()["freq"] == 5000.0


def test_eval_value_resolved_against_md_on_commit(service, ml, md):
    editor_id, _ = service.open("module", discriminator="pulse")
    service.set_field(editor_id, "freq", {"__kind": "eval", "expr": "r_f"})
    service.commit(editor_id, "agent_eval")

    # commit lowered EvalValue('r_f') against md -> concrete md.r_f.
    assert ml.modules["agent_eval"].to_dict()["freq"] == md.r_f


def test_eval_value_requires_string_expr(service):
    editor_id, _ = service.open("module", discriminator="pulse")
    with pytest.raises(CfgEditorError):
        service.set_field(editor_id, "freq", {"__kind": "eval", "expr": 123})


# ---------------------------------------------------------------------------
# ModuleRef key switch rebinds the sub-tree
# ---------------------------------------------------------------------------


def test_ref_switch_returns_new_subtree(service):
    # A pulse module references a waveform; switching the waveform style
    # rebuilds which sub-fields exist (Const has length only; Gauss adds sigma).
    editor_id, paths = service.open("module", discriminator="pulse")
    keys = _paths(paths)
    assert "waveform.ref" in keys
    assert "waveform.value.sigma" not in keys  # const has no sigma

    # Switch the waveform ref key; set_field returns the rebuilt sub-tree.
    res = service.set_field(editor_id, "waveform.ref", "<Custom:Gauss>")
    sub_keys = _paths(res["paths"])
    # The returned sub-tree is rooted at the changed path and now exposes the
    # gauss-only field that did not exist before the switch.
    assert "waveform.value.sigma" in sub_keys


# ---------------------------------------------------------------------------
# discard + unknown id
# ---------------------------------------------------------------------------


def test_unknown_editor_id_fails(service):
    with pytest.raises(CfgEditorError):
        service.set_field("editor-nope", "freq", 1.0)
    with pytest.raises(CfgEditorError):
        service.commit("editor-nope", "x")
    with pytest.raises(CfgEditorError):
        service.discard("editor-nope")


def test_discard_removes_session(service):
    editor_id, _ = service.open("module", discriminator="pulse")
    service.discard(editor_id)
    with pytest.raises(CfgEditorError):
        service.get(editor_id)


def test_discard_for_client_batch_ignores_unknown(service):
    id1, _ = service.open("module", discriminator="pulse")
    id2, _ = service.open("waveform", discriminator="const")
    service.discard_for_client([id1, id2, "editor-unknown"])
    with pytest.raises(CfgEditorError):
        service.get(id1)
    with pytest.raises(CfgEditorError):
        service.get(id2)


def test_commit_failure_keeps_session(service, ctrl):
    editor_id, _ = service.open("module", discriminator="pulse")
    ctrl.set_ml_module_from_raw.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError):
        service.commit(editor_id, "bad")
    # session survives so the agent can fix and retry.
    assert service.get(editor_id)
