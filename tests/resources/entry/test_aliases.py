"""Setup and point authored aliases through public entry and codec seams."""

from pathlib import Path

import pytest
from zcu_tools.resources.entry import PointView, ResultEntry
from zcu_tools.resources.entry.aliases import (
    EntryAliases,
    load_entry_aliases,
    save_entry_aliases,
)


@pytest.fixture
def authored() -> EntryAliases:
    return EntryAliases(aliases={"spare": "", "drive": "Q1", "sense": "$drive.sense"})


def test_missing_setup_aliases_are_an_empty_table(entry: ResultEntry) -> None:
    assert entry.load_aliases() == EntryAliases(aliases={})


def test_missing_point_aliases_do_not_fall_back_to_setup(
    entry: ResultEntry, authored: EntryAliases
) -> None:
    entry.new_point("legacy")
    entry.save_aliases(authored)
    assert entry.load_aliases(point="legacy") == EntryAliases(aliases={})
    assert entry.load_aliases() == authored


def test_alias_round_trip_keeps_setup_and_points_independent(
    entry: ResultEntry, entry_roots: tuple[Path, Path], authored: EntryAliases
) -> None:
    entry.new_point("one")
    entry.new_point("two")
    entry.save_aliases(authored)
    point_table = EntryAliases(aliases={"drive": "Q2", "spare": ""})
    entry.save_aliases(point_table, point="one")
    loaded = entry.load_aliases(point="one")
    assert list(loaded.aliases.items()) == [("drive", "Q2"), ("spare", "")]
    loaded.aliases["drive"] = "not persisted"
    results, database = entry_roots
    reopened = ResultEntry.open("entry", result_root=results, database_root=database)
    assert reopened.load_aliases(point="one") == point_table
    assert reopened.load_aliases(point="two") == EntryAliases(aliases={})
    assert list(reopened.load_aliases().aliases.items()) == list(
        authored.aliases.items()
    )
    reopened.save_aliases(EntryAliases(aliases={}), point="one")
    assert entry.load_aliases(point="one") == EntryAliases(aliases={})
    assert entry.load_aliases() == authored


@pytest.mark.parametrize("operation", ["load", "save"])
@pytest.mark.parametrize(
    "label,error",
    [
        ("../escape", ValueError),
        ("missing", FileNotFoundError),
        ("partial", FileNotFoundError),
    ],
)
def test_point_alias_io_requires_a_safe_complete_point(
    entry: ResultEntry, operation: str, label: str, error: type[Exception]
) -> None:
    directory = entry.result_path / "points" / "partial"
    directory.mkdir()
    (directory / "point.yaml").write_text("{}\n", encoding="utf-8")
    with pytest.raises(error):
        if operation == "load":
            entry.load_aliases(point=label)
        else:
            entry.save_aliases(EntryAliases(aliases={}), point=label)


def test_point_aliases_accept_forward_minor_and_preserve_raw_rows(
    entry: ResultEntry, authored: EntryAliases
) -> None:
    entry.new_point("one")
    source = entry.result_path / "points/one/aliases.yaml"
    source.write_text(
        'format: zcu.aliases\nformat_version: "1.8"\naliases:\n'
        '  spare: ""\n  drive: Q1\n  sense: $drive.sense\n',
        encoding="utf-8",
    )
    assert list(entry.load_aliases(point="one").aliases.items()) == list(
        authored.aliases.items()
    )


@pytest.mark.parametrize(
    "version,body",
    [("2.0", "{}"), ("1.0", "[]"), ("1.0", "{drive: null}"), ("1.0", "{drive: 7}")],
)
def test_point_aliases_reject_invalid_documents_without_setup_fallback(
    entry: ResultEntry, authored: EntryAliases, version: str, body: str
) -> None:
    entry.save_aliases(authored)
    entry.new_point("one")
    source = entry.result_path / "points/one/aliases.yaml"
    source.write_text(
        f'format: zcu.aliases\nformat_version: "{version}"\naliases: {body}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        entry.load_aliases(point="one")
    assert entry.load_aliases() == authored


def test_new_point_copies_setup_aliases_only_at_creation(
    entry: ResultEntry, authored: EntryAliases
) -> None:
    entry.new_point("unseeded")
    assert (
        load_entry_aliases(entry.result_path / "points/unseeded/aliases.yaml") is None
    )
    entry.save_aliases(authored)
    entry.new_point("seeded")
    source = entry.result_path / "aliases.yaml"
    destination = entry.result_path / "points/seeded/aliases.yaml"
    assert load_entry_aliases(destination) == authored
    entry.save_aliases(EntryAliases(aliases={"drive": "Q2"}))
    assert load_entry_aliases(destination) == authored
    assert (
        load_entry_aliases(entry.result_path / "points/unseeded/aliases.yaml") is None
    )
    save_entry_aliases(destination, EntryAliases(aliases={}))
    assert load_entry_aliases(source) == EntryAliases(aliases={"drive": "Q2"})


@pytest.mark.parametrize("foreign", [False, True], ids=["same-entry", "cross-entry"])
@pytest.mark.parametrize("via_view", [False, True], ids=["label", "view"])
def test_clone_copies_published_source_point_aliases_not_either_setup(
    entry: ResultEntry,
    entry_roots: tuple[Path, Path],
    authored: EntryAliases,
    foreign: bool,
    via_view: bool,
) -> None:
    results, database = entry_roots
    source_entry = (
        ResultEntry.create("source", result_root=results, database_root=database)
        if foreign
        else entry
    )
    source_view = source_entry.new_point("original")
    source_aliases = source_entry.result_path / "points/original/aliases.yaml"
    save_entry_aliases(source_aliases, authored)
    source_entry.save_aliases(EntryAliases(aliases={"drive": "source setup"}))
    entry.save_aliases(EntryAliases(aliases={"drive": "destination setup"}))
    clone_from: str | PointView = (
        source_view if via_view else ("source/original" if foreign else "original")
    )
    entry.new_point("cloned", clone_from=clone_from)
    destination = entry.result_path / "points/cloned/aliases.yaml"
    assert load_entry_aliases(destination) == authored
    save_entry_aliases(source_aliases, EntryAliases(aliases={"drive": "later source"}))
    assert load_entry_aliases(destination) == authored
    save_entry_aliases(destination, EntryAliases(aliases={}))
    assert load_entry_aliases(source_aliases) == EntryAliases(
        aliases={"drive": "later source"}
    )
    source_aliases.unlink()
    entry.new_point("missing", clone_from=clone_from)
    assert load_entry_aliases(entry.result_path / "points/missing/aliases.yaml") is None
    assert entry.load_aliases() == EntryAliases(aliases={"drive": "destination setup"})
