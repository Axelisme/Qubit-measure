from __future__ import annotations

import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
from zcu_tools.utils.datasaver.transport import download_from_server, upload_to_server


@dataclass
class _Response:
    status_code: int
    text: str = ""
    content: bytes = b""


def test_upload_to_server_posts_file(monkeypatch: pytest.MonkeyPatch, tmp_path):
    payload = tmp_path / "data.hdf5"
    payload.write_bytes(b"payload")
    calls: list[tuple[str, dict[str, Any]]] = []

    def post(url: str, *, files: dict[str, Any]) -> _Response:
        calls.append((url, files))
        _filename, fileobj = files["file"]
        assert fileobj.read() == b"payload"
        return _Response(200, "ok")

    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(post=post))

    upload_to_server(str(payload), "127.0.0.1", 4999)

    assert calls[0][0] == "http://127.0.0.1:4999/upload"


def test_upload_to_server_raises_on_http_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    payload = tmp_path / "data.hdf5"
    payload.write_bytes(b"payload")

    def post(url: str, *, files: dict[str, Any]) -> _Response:
        return _Response(500, "server failed")

    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(post=post))

    with pytest.raises(RuntimeError, match="Failed to upload file"):
        upload_to_server(str(payload), "127.0.0.1", 4999)


def test_download_from_server_writes_bare_filename(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    calls: list[tuple[str, dict[str, str]]] = []
    monkeypatch.chdir(tmp_path)

    def post(url: str, *, json: dict[str, str]) -> _Response:
        calls.append((url, json))
        return _Response(200, "ok", b"downloaded")

    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(post=post))

    download_from_server("data.hdf5", "127.0.0.1", 4999)

    assert calls == [
        ("http://127.0.0.1:4999/download", {"path": "data.hdf5"}),
    ]
    assert (tmp_path / "data.hdf5").read_bytes() == b"downloaded"


def test_download_from_server_raises_on_http_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    monkeypatch.chdir(tmp_path)

    def post(url: str, *, json: dict[str, str]) -> _Response:
        return _Response(404, "missing")

    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(post=post))

    with pytest.raises(RuntimeError, match="Failed to download file"):
        download_from_server("missing.hdf5", "127.0.0.1", 4999)

    assert not (tmp_path / "missing.hdf5").exists()
