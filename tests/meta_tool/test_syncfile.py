from __future__ import annotations

import pytest
from zcu_tools.meta_tool import MetaDict, ModuleLibrary
from zcu_tools.meta_tool.syncfile import auto_sync


def test_syncfile_has_persistence_false_without_path() -> None:
    assert MetaDict().has_persistence is False
    assert ModuleLibrary().has_persistence is False


def test_syncfile_has_persistence_true_with_path(tmp_path) -> None:
    assert MetaDict(tmp_path / "meta.json").has_persistence is True
    assert ModuleLibrary(tmp_path / "module_cfg.yaml").has_persistence is True


def test_auto_sync_rejects_non_syncfile_receiver() -> None:
    @auto_sync("read")
    def decorated(receiver: object) -> None:
        return None

    with pytest.raises(TypeError, match="Expected first argument to be SyncFile"):
        decorated(object())
