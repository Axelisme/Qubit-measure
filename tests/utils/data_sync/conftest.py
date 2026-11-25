import os
from pathlib import Path

import pytest
from zcu_tools.utils.data_sync import rclone, setup

from ._support import bind


@pytest.fixture
def unbound_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (tmp_path / "account").mkdir()
    (tmp_path / "account" / "rclone.conf").touch()
    # Select a test cloud root without launching or downloading rclone.
    monkeypatch.setattr(rclone, "prepare_rclone", lambda cache: cache / "rclone")
    monkeypatch.setattr(
        setup, "select_remote_root", lambda *args, **kwargs: "lab:measurements"
    )
    return checkout


@pytest.fixture
def local_cloud(unbound_checkout: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    binary = os.environ.get("DATA_SYNC_TEST_RCLONE")
    if binary is None:
        pytest.skip("Set DATA_SYNC_TEST_RCLONE to a verified rclone executable")
    tmp_path = unbound_checkout.parent
    cloud = tmp_path / "cloud"
    cloud.mkdir()
    for name in tuple(os.environ):
        if name.upper().startswith("RCLONE_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("RCLONE_CACHE_DIR", str(tmp_path / "rclone-cache"))
    (tmp_path / "account" / "rclone.conf").write_text(
        f"[lab]\ntype = alias\nremote = {cloud.as_posix()}\n", encoding="utf-8"
    )
    # Replace the download only. The copy and listing use the real local backend.
    monkeypatch.setattr(rclone, "prepare_rclone", lambda cache: Path(binary))
    bind(
        unbound_checkout,
        "local/alice",
        monkeypatch,
        labdata=unbound_checkout / "Labdata",
    )
    return cloud / "measurements"
