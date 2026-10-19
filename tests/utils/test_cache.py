"""User cache root resolution at the environment/platform boundary."""

import tempfile
from pathlib import Path

import platformdirs
import pytest
from zcu_tools import utils


def test_cache_root_uses_override_and_rechecks_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first cache"
    second = tmp_path / "second cache"
    monkeypatch.setenv("ZCU_TOOLS_CACHE_DIR", str(first))
    assert utils.user_cache_root() == first
    monkeypatch.setenv("ZCU_TOOLS_CACHE_DIR", str(second))
    assert utils.user_cache_root() == second


def test_cache_root_without_override_uses_platform_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ZCU_TOOLS_CACHE_DIR", raising=False)
    platform_root = tmp_path / "platform cache"

    def platform_cache(appname: str, appauthor: str) -> str:
        assert (appname, appauthor) == ("zcu_tools", "zcu_tools")
        return str(platform_root)

    monkeypatch.setattr(platformdirs, "user_cache_dir", platform_cache)
    assert utils.user_cache_root() == platform_root


@pytest.mark.parametrize("override", ["", ".", "relative/cache"])
def test_cache_root_rejects_invalid_override(
    override: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ZCU_TOOLS_CACHE_DIR", override)
    with pytest.raises(ValueError, match="ZCU_TOOLS_CACHE_DIR"):
        utils.user_cache_root()


def test_cache_root_falls_back_to_temp_on_platform_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ZCU_TOOLS_CACHE_DIR", raising=False)

    def fail_platform(*args: object, **kwargs: object) -> str:
        raise RuntimeError("platform lookup failed")

    monkeypatch.setattr(platformdirs, "user_cache_dir", fail_platform)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    assert utils.user_cache_root() == tmp_path / ".zcu_tools"
