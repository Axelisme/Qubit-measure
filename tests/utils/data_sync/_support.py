import io
import json
import os
import sys
from pathlib import Path

import pytest
from zcu_tools.utils.data_sync.cli import main


def bind(
    checkout: Path,
    prefix: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    labdata: Path | None = None,
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n"))
    assert (
        sync(
            checkout,
            "setup",
            "--prefix",
            prefix,
            "--labdata",
            str(labdata or checkout.parent / "labdata"),
        )
        == 0
    )


def sync(checkout: Path, *args: str) -> int:
    return main(
        [
            "--config-dir",
            str(checkout.parent / "account"),
            "--cache-dir",
            str(checkout.parent / "cache"),
            *args,
        ],
        checkout=checkout,
    )


def write_file(root: Path, relative: str, content: bytes, *, mtime: int) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    os.utime(path, (mtime, mtime))


def state_dir(labdata: Path) -> Path:
    return labdata / ".data-sync/state"


def only_baseline(labdata: Path) -> Path:
    (path,) = state_dir(labdata).glob("*.json")
    return path


def baseline_files(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))["files"]
