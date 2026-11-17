"""Cached aliases observations through the ResultEntry opener and handle."""

from pathlib import Path

import pytest
from zcu_tools.resources.entry import ResultEntry, document_revision
from zcu_tools.resources.entry.aliases import EntryAliases, save_entry_aliases


@pytest.mark.parametrize("point", [None, "one"])
def test_read_keeps_one_observation_and_returns_isolated_lists(
    entry: ResultEntry, point: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = entry.result_path
    if point is not None:
        entry.new_point(point)
        directory = directory / "points" / point
    (directory / "parameters.yaml").write_text("broken: [")
    (directory / "library.yaml").write_text("broken: [")
    source = directory / "aliases.yaml"
    save_entry_aliases(source, EntryAliases(aliases={"group": ["Q1", "Q2"]}))
    original_read = Path.read_bytes
    reads = 0

    def count_read(path: Path) -> bytes:
        nonlocal reads
        if path == source:
            reads += 1
        return original_read(path)

    # Content cannot reveal duplicate opener reads; observe the filesystem boundary.
    monkeypatch.setattr(Path, "read_bytes", count_read)
    handle = entry.aliases(point)
    observed = handle.read()
    source.unlink()
    group = observed.aliases.aliases["group"]
    assert group == ["Q1", "Q2"]
    if isinstance(group, list):
        group.append("Q3")
    observed.aliases.aliases["new"] = "Q4"
    assert handle.read().aliases.aliases == {"group": ["Q1", "Q2"]}
    assert document_revision(handle) == observed.revision
    assert reads == 1
    assert not source.exists()


def test_refresh_admits_format_observations_but_strict_preserves_cached_state(
    entry: ResultEntry,
) -> None:
    source = entry.result_path / "aliases.yaml"
    handle = entry.aliases()
    assert handle.refresh() is False
    save_entry_aliases(source, EntryAliases(aliases={}))
    assert handle.refresh() is False
    source.unlink()
    assert handle.refresh() is False
    assert document_revision(handle).revision is None
    save_entry_aliases(source, EntryAliases(aliases={}))
    assert handle.refresh() is False
    empty_revision = document_revision(handle)
    source.write_text(
        'format: zcu.aliases\nformat_version: "1.8"\naliases: {} # comment\n'
    )
    assert handle.refresh(strict=True) is False
    assert document_revision(handle) != empty_revision
    source.write_text("format: wrong\n")
    before = handle.read()
    with pytest.raises(ValueError, match="format"):
        handle.refresh(strict=True)
    assert handle.read() == before
    assert handle.refresh() is True
    invalid = handle.read()
    assert invalid.error is not None
    source.write_bytes(b"\xff")
    assert handle.refresh() is True
    assert handle.refresh() is False
    assert document_revision(handle) != invalid.revision
    source.unlink()
    assert handle.refresh() is True
    assert handle.read().aliases.aliases == {}
    assert document_revision(handle).revision is None


def test_save_installs_isolated_input_and_writer_revision_without_rereading(
    entry: ResultEntry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = entry.result_path / "aliases.yaml"
    source.write_text("broken: [")
    handle = entry.aliases()
    authored = EntryAliases(aliases={"group": ["Q1", "Q2"]})
    original_read = Path.read_bytes

    def reject_alias_read(path: Path) -> bytes:
        if path == source:
            raise AssertionError("save must not read old or newly written aliases")
        return original_read(path)

    # Read count is not visible from content; observe the filesystem boundary.
    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "read_bytes", reject_alias_read)
        written = handle.save(authored)
        assert document_revision(handle) == written
    group = authored.aliases["group"]
    if isinstance(group, list):
        group.append("Q3")
    authored.aliases["new"] = "Q4"
    assert handle.read().aliases.aliases == {"group": ["Q1", "Q2"]}
    assert entry.aliases().read().revision == written
    assert entry.aliases().read().aliases == handle.read().aliases


def test_refresh_compares_authored_row_order_and_retains_state_on_io_failure(
    entry: ResultEntry,
) -> None:
    source = entry.result_path / "aliases.yaml"
    save_entry_aliases(source, EntryAliases(aliases={"a": "Q1", "b": "$a"}))
    handle = entry.aliases()
    save_entry_aliases(source, EntryAliases(aliases={"b": "$a", "a": "Q1"}))
    assert handle.refresh() is True
    assert list(handle.read().aliases.aliases) == ["b", "a"]
    before = handle.read()
    source.unlink()
    source.mkdir()
    for strict in (False, True):
        with pytest.raises(OSError, match="directory"):
            handle.refresh(strict=strict)
        assert handle.read() == before
