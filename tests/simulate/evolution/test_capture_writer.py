"""Capture persistence through the public writer and frozen codec."""

import os
import warnings
from contextlib import closing
from typing import override

import pytest
from zcu_tools.simulate import evolution

from ._engine import batch, source


def test_empty_capture_writes_unique_readable_manifests(tmp_path, monkeypatch):
    monkeypatch.setenv("ZCU_TOOLS_CACHE_DIR", str(tmp_path))
    capture = evolution.CaptureBuffer()
    first = evolution.write_capture(capture)
    second = evolution.write_capture(capture)
    assert first is not None and second is not None
    assert first.is_absolute()
    assert first.parent == evolution.capture_directory()
    assert first.parent == tmp_path / "simulator-debug" / "v1"
    assert first != second
    assert evolution.read_frozen_manifest(first) == ()
    assert evolution.read_frozen_manifest(second) == ()


def test_capture_rewrite_snapshots_once_and_keeps_previous_records(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ZCU_TOOLS_CACHE_DIR", str(tmp_path))

    class ObservedBuffer(evolution.CaptureBuffer):
        reads = 0

        @property
        @override
        def records(self) -> tuple[evolution.FrozenRecord, ...]:
            self.reads += 1
            return super().records

    capture = ObservedBuffer()
    with closing(source()) as src:
        with capture.recording(src, (batch(1),), 1e-3, None):
            pass
        first = evolution.write_capture(capture)
        with capture.recording(src, (batch(0),), 1e-3, None):
            pass
        second = evolution.write_capture(capture)
    assert capture.reads == 2
    assert first is not None and second is not None and first != second
    before = evolution.read_frozen_manifest(first)
    after = evolution.read_frozen_manifest(second)
    assert len(before) == 1 and len(after) == 2
    assert before[0].physical_sha256 == after[0].physical_sha256
    assert before[0].metadata == after[0].metadata
    assert len(list(first.parent.iterdir())) == 2


@pytest.mark.parametrize("failure", ["io", "nonfinite", "unsupported"])
def test_publication_failure_warns_once_and_cleans_temporary(
    tmp_path, monkeypatch, failure
):
    monkeypatch.setenv("ZCU_TOOLS_CACHE_DIR", str(tmp_path))
    capture = evolution.CaptureBuffer()
    with (
        closing(source()) as src,
        capture.recording(src, (batch(1),), 1e-3, None) as metadata,
    ):
        if failure == "nonfinite":
            metadata[0]["budget"] = float("nan")
        elif failure == "unsupported":
            # FrozenRecord metadata is opaque; JSON rejects unsupported objects.
            capture.records[0].physical["unsupported"] = object()
    directory = evolution.capture_directory()
    if failure == "io":
        directory.parent.mkdir(parents=True)
        directory.write_text("not a directory")
    with pytest.warns(RuntimeWarning, match="capture could not be written") as seen:
        assert evolution.write_capture(capture) is None
    assert len(seen) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert evolution.write_capture(capture) is None
    if failure != "io":
        assert list(directory.iterdir()) == []


@pytest.mark.parametrize("override", ["", "relative/path"])
def test_invalid_cache_override_propagates(tmp_path, monkeypatch, override):
    monkeypatch.setenv("ZCU_TOOLS_CACHE_DIR", override)
    with pytest.raises(ValueError, match="non-empty absolute"):
        evolution.write_capture(evolution.CaptureBuffer())
    assert list(tmp_path.iterdir()) == []


def test_snapshot_and_unexpected_publication_errors_propagate(tmp_path, monkeypatch):
    monkeypatch.setenv("ZCU_TOOLS_CACHE_DIR", str(tmp_path))

    class BrokenBuffer(evolution.CaptureBuffer):
        @property
        @override
        def records(self) -> tuple[evolution.FrozenRecord, ...]:
            raise TypeError("snapshot failure")

    with pytest.raises(TypeError, match="snapshot failure"):
        evolution.write_capture(BrokenBuffer())

    def fail_replace(*args):
        raise RuntimeError("unexpected publisher failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(RuntimeError, match="unexpected publisher failure"):
        evolution.write_capture(evolution.CaptureBuffer())
    assert list(evolution.capture_directory().iterdir()) == []
