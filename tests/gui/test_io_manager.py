"""IOManager tests (real ContextManager).

DeviceManager registry CRUD coverage lives in
``services/test_device_manager.py`` alongside the rest of the device-registry
tests — this file is IOManager-only.
"""

from __future__ import annotations

import pytest
from zcu_tools.gui.app.measure.adapter import SessionEnv
from zcu_tools.gui.session.services.io_manager import IOManager

# ---------------------------------------------------------------------------
# setup / list_contexts
# ---------------------------------------------------------------------------


def test_iomanager_not_setup_returns_empty_contexts():
    io = IOManager()
    assert io.list_contexts() == []


def test_iomanager_not_setup_raises_on_use_context():
    io = IOManager()
    with pytest.raises(RuntimeError, match="not set up"):
        io.use_context("anything", SessionEnv(parameters=None, soc=None, soccfg=None))


def test_iomanager_not_setup_returns_none_label():
    io = IOManager()
    assert io.get_active_label() is None


def test_iomanager_setup_creates_exp_dir(tmp_path):
    io = IOManager()
    io.setup(str(tmp_path))
    assert io.list_contexts() == []
