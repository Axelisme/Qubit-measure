"""IOManager tests (real ExperimentManager).

GlobalDeviceManager registry CRUD coverage lives in
``services/test_device_manager.py`` alongside the rest of the device-registry
tests — this file is IOManager-only.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.main.adapter import ExpContext
from zcu_tools.gui.session.services.io_manager import IOManager


def _make_base_ctx(**overrides) -> ExpContext:
    defaults = dict(
        md=MagicMock(),
        ml=MagicMock(),
        soc=object(),
        soccfg=object(),
        database_path="/db",
        predictor=None,
    )
    defaults.update(overrides)
    return ExpContext(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# setup / list_contexts
# ---------------------------------------------------------------------------


def test_iomanager_not_setup_returns_empty_contexts():
    io = IOManager()
    assert io.list_contexts() == []


def test_iomanager_not_setup_raises_on_use_context():
    io = IOManager()
    with pytest.raises(RuntimeError, match="not set up"):
        io.use_context("anything", _make_base_ctx())


def test_iomanager_not_setup_returns_none_label():
    io = IOManager()
    assert io.get_active_label() is None


def test_iomanager_setup_creates_exp_dir(tmp_path):
    io = IOManager()
    io.setup(str(tmp_path))
    assert io.list_contexts() == []


def test_iomanager_contexts_live_under_result_exps(tmp_path):
    io = IOManager()
    result_dir = tmp_path / "result" / "ChipA" / "Q1"
    io.setup(str(result_dir))

    io.new_context(_make_base_ctx())
    label = io.get_active_label()
    assert label is not None

    assert (result_dir / "exps" / label / "meta_info.json").exists()
    assert (result_dir / "exps" / label / "module_cfg.yaml").exists()
    assert not (result_dir / label / "meta_info.json").exists()
    assert io.list_contexts() == [label]


# ---------------------------------------------------------------------------
# use_context
# ---------------------------------------------------------------------------


def test_iomanager_use_context_preserves_soc_and_predictor(tmp_path):
    io = IOManager()
    io.setup(str(tmp_path))

    # create a context first
    ctx0 = _make_base_ctx()
    io.new_context(ctx0)
    label = io.get_active_label()
    assert label is not None
    label_str: str = label

    # now use_context with a different base_ctx that has soc / predictor set
    soc_obj = object()
    pred_obj = object()
    base = _make_base_ctx(soc=soc_obj, predictor=pred_obj)
    result_ctx = io.use_context(label_str, base)

    assert result_ctx.soc is soc_obj
    assert result_ctx.predictor is pred_obj
    assert result_ctx.database_path == base.database_path


def test_iomanager_use_context_updates_md_ml(tmp_path):
    io = IOManager()
    io.setup(str(tmp_path))

    ctx0 = _make_base_ctx()
    io.new_context(ctx0)
    label = io.get_active_label()
    assert label is not None

    base = _make_base_ctx()
    old_md = base.md
    old_ml = base.ml

    result_ctx = io.use_context(label, base)
    # md and ml should come from ExperimentManager, not the old mocks
    assert result_ctx.md is not old_md
    assert result_ctx.ml is not old_ml


# ---------------------------------------------------------------------------
# new_context
# ---------------------------------------------------------------------------


def test_iomanager_new_context_creates_new_label(tmp_path):
    io = IOManager()
    io.setup(str(tmp_path))
    ctx0 = _make_base_ctx()
    io.new_context(ctx0, value=1e-3, unit="A")
    label = io.get_active_label()
    assert label is not None
    assert "mA" in label or "A" in label


def test_iomanager_new_context_result_ctx_has_fresh_md_ml(tmp_path):
    io = IOManager()
    io.setup(str(tmp_path))
    base = _make_base_ctx()
    result_ctx = io.new_context(base)
    assert result_ctx.md is not base.md
    assert result_ctx.ml is not base.ml


def test_iomanager_new_context_preserves_database_path(tmp_path):
    io = IOManager()
    io.setup(str(tmp_path))
    base = _make_base_ctx(database_path="/special/db")
    result_ctx = io.new_context(base)
    assert result_ctx.database_path == "/special/db"


def test_iomanager_new_context_clone_creates_separate_files(tmp_path):
    io = IOManager()
    io.setup(str(tmp_path))
    ctx0 = _make_base_ctx()
    ctx1 = io.new_context(ctx0)
    src_label = io.get_active_label()
    assert src_label is not None
    ctx2 = io.new_context(ctx1, clone_from=src_label)
    assert ctx1.md is not ctx2.md
    assert ctx1.ml is not ctx2.ml
    assert len(io.list_contexts()) == 2
