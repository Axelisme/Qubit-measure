"""SaveLayout destinations and validation through the public entry facade."""

import re
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import pytest
from zcu_tools.resources.entry import (
    ArtifactKey,
    Output,
    ResultEntry,
    SaveLayout,
    new_run_id,
)

SAVED_AT = datetime(2026, 10, 6, 0, 30, tzinfo=timezone(timedelta(hours=8)))


def layout(
    tmp_path: Path, *, run_id: str = "opaque_run%id", point: str | None = None
) -> SaveLayout:
    return SaveLayout(
        result_path=tmp_path / "results" / "entry",
        database_path=tmp_path / "Database" / "entry",
        run_id=run_id,
        point=point,
        saved_at=SAVED_AT,
    )


def tree_bytes(root: Path) -> dict[Path, bytes]:
    """Snapshot synthetic published files to observe writes, not implementation state."""
    return {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize(
    ("key", "point", "expected"),
    [
        (
            ArtifactKey("run", "primary", "data"),
            None,
            (
                ("data_h5", "Database/entry/runs/opaque_run%id/data.h5"),
                (
                    "labber",
                    "Database/entry/Labber/2026/10/Data_1006/opaque_run%id_1.hdf5",
                ),
            ),
        ),
        (
            ArtifactKey("analysis", "other", "data"),
            "point",
            (
                ("data_h5", "Database/entry/runs/opaque_run%id/data.h5"),
                (
                    "labber",
                    "Database/entry/Labber/2026/10/Data_1006/opaque_run%id_1.hdf5",
                ),
            ),
        ),
        (
            ArtifactKey("run", "artifact", "figure", "spectrum"),
            "point",
            (
                (
                    "png",
                    "Database/entry/runs/opaque_run%id/figures/run_artifact_spectrum.png",
                ),
                (
                    "png",
                    "results/entry/points/point/figures/opaque%5Frun%25id_run_artifact_spectrum.png",
                ),
            ),
        ),
        (
            ArtifactKey("caller", "artifact", "figure", "spectrum"),
            None,
            (
                (
                    "png",
                    "Database/entry/runs/opaque_run%id/figures/caller_artifact_spectrum.png",
                ),
                (
                    "png",
                    "results/entry/figures/opaque%5Frun%25id_caller_artifact_spectrum.png",
                ),
            ),
        ),
        (
            ArtifactKey("analysis", "fit", "analysis", "coefficients"),
            None,
            (
                (
                    "json",
                    "Database/entry/runs/opaque_run%id/analysis/analysis_fit_coefficients.json",
                ),
            ),
        ),
        (
            ArtifactKey("caller", "fit", "analysis", "residuals"),
            "point",
            (
                (
                    "json",
                    "Database/entry/runs/opaque_run%id/analysis/caller_fit_residuals.json",
                ),
            ),
        ),
        (
            ArtifactKey("a_b", "c", "figure", "x"),
            None,
            (
                ("png", "Database/entry/runs/opaque_run%id/figures/a%5Fb_c_x.png"),
                ("png", "results/entry/figures/opaque%5Frun%25id_a%5Fb_c_x.png"),
            ),
        ),
        (
            ArtifactKey("a", "b_c", "figure", "x"),
            None,
            (
                ("png", "Database/entry/runs/opaque_run%id/figures/a_b%5Fc_x.png"),
                ("png", "results/entry/figures/opaque%5Frun%25id_a_b%5Fc_x.png"),
            ),
        ),
        (
            ArtifactKey("%5F_%", "n_", "analysis", "m_%"),
            None,
            (
                (
                    "json",
                    "Database/entry/runs/opaque_run%id/analysis/%255F%5F%25_n%5F_m%5F%25.json",
                ),
            ),
        ),
        (
            ArtifactKey("未知", "任意", "figure", "圖"),
            "工作點",
            (
                ("png", "Database/entry/runs/opaque_run%id/figures/未知_任意_圖.png"),
                (
                    "png",
                    "results/entry/points/工作點/figures/opaque%5Frun%25id_未知_任意_圖.png",
                ),
            ),
        ),
    ],
)
def test_output_table_is_exact_ordered_and_does_not_publish(
    tmp_path: Path,
    key: ArtifactKey,
    point: str | None,
    expected: tuple[tuple[str, str], ...],
) -> None:
    destinations = layout(tmp_path, point=point).outputs(key)
    assert tuple((item.format, item.path) for item in destinations) == tuple(
        (format_name, tmp_path / relative_path)
        for format_name, relative_path in expected
    )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("member", ["figure", "analysis"])
def test_members_of_one_artifact_have_distinct_destinations(
    tmp_path: Path, member: str
) -> None:
    key = ArtifactKey(
        "post_analysis", "fit", "figure" if member == "figure" else "analysis", "first"
    )
    first = layout(tmp_path).outputs(key)
    second = layout(tmp_path).outputs(replace(key, member_name="second"))
    extension = "png" if member == "figure" else "json"
    folder = "figures" if member == "figure" else "analysis"
    assert (
        first[0].path
        == tmp_path
        / f"Database/entry/runs/opaque_run%id/{folder}/post%5Fanalysis_fit_first.{extension}"
    )
    assert (
        second[0].path
        == tmp_path
        / f"Database/entry/runs/opaque_run%id/{folder}/post%5Fanalysis_fit_second.{extension}"
    )
    assert set(first).isdisjoint(second)


@pytest.mark.parametrize("run_id", ["opaque_run%id", "opaque_1", "名稱"])
def test_labber_avoids_existing_files_but_does_not_reserve(
    tmp_path: Path, run_id: str
) -> None:
    save_layout = layout(tmp_path, run_id=run_id)
    key = ArtifactKey("run", "data", "data")
    folder = tmp_path / "Database/entry/Labber/2026/10/Data_1006"
    assert save_layout.outputs(key)[1] == Output("labber", folder / f"{run_id}_1.hdf5")
    folder.mkdir(parents=True)
    (folder / f"{run_id}_1.hdf5").write_bytes(b"first")
    (folder / f"{run_id}_2.hdf5").write_bytes(b"second")
    before = tree_bytes(tmp_path)
    expected = (
        Output("data_h5", tmp_path / "Database/entry/runs" / run_id / "data.h5"),
        Output("labber", folder / f"{run_id}_3.hdf5"),
    )
    assert save_layout.outputs(key) == expected
    assert save_layout.outputs(key) == expected
    assert tree_bytes(tmp_path) == before


@pytest.mark.parametrize("member", ["figure", "analysis"])
def test_fixed_paths_are_returned_even_if_already_published(
    tmp_path: Path, member: str
) -> None:
    key = ArtifactKey(
        "run", "fit", "figure" if member == "figure" else "analysis", "first"
    )
    save_layout = layout(tmp_path)
    expected = save_layout.outputs(key)
    for destination in expected:
        destination.path.parent.mkdir(parents=True, exist_ok=True)
        destination.path.write_bytes(b"existing")
    before = tree_bytes(tmp_path)
    assert save_layout.outputs(key) == expected
    assert tree_bytes(tmp_path) == before


def test_relative_roots_become_absolute_without_requiring_entry_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    save_layout = SaveLayout(
        result_path=Path("results/entry"),
        database_path=Path("Database/entry"),
        run_id="no-meaning",
        point=None,
        saved_at=SAVED_AT,
    )
    assert save_layout.outputs(ArtifactKey("run", "data", "data")) == (
        Output("data_h5", tmp_path / "Database/entry/runs/no-meaning/data.h5"),
        Output(
            "labber",
            tmp_path / "Database/entry/Labber/2026/10/Data_1006/no-meaning_1.hdf5",
        ),
    )
    assert list(tmp_path.iterdir()) == []


UNSAFE = [
    "",
    ".",
    "..",
    "a/b",
    "a\\b",
    "a\x00b",
    "/absolute",
    "C:\\absolute",
    "C:drive",
    "\\anchor",
]


@pytest.mark.parametrize("field", ["run_id", "point"])
@pytest.mark.parametrize("value", UNSAFE)
def test_layout_rejects_unsafe_segments_at_construction(
    tmp_path: Path, field: str, value: str
) -> None:
    before = tree_bytes(tmp_path)
    with pytest.raises(ValueError, match=field):
        layout(
            tmp_path,
            run_id=value if field == "run_id" else "run",
            point=value if field == "point" else None,
        )
    assert tree_bytes(tmp_path) == before


@pytest.mark.parametrize("field", ["section", "name", "member_name"])
@pytest.mark.parametrize("value", UNSAFE)
def test_artifact_key_rejects_unsafe_segments(
    tmp_path: Path, field: str, value: str
) -> None:
    before = tree_bytes(tmp_path)
    with pytest.raises(ValueError, match=field):
        ArtifactKey(
            value if field == "section" else "run",
            value if field == "name" else "fit",
            "figure",
            value if field == "member_name" else "first",
        )
    assert tree_bytes(tmp_path) == before


@pytest.mark.parametrize("member", ["figure", "analysis"])
def test_member_name_is_required_for_non_data_members(member: str) -> None:
    with pytest.raises(ValueError, match="member_name"):
        ArtifactKey("run", "fit", "figure" if member == "figure" else "analysis")


def test_data_member_must_not_have_a_member_name() -> None:
    with pytest.raises(ValueError, match="member_name"):
        ArtifactKey("run", "data", "data", "not-allowed")


@pytest.mark.parametrize("member", ["unknown"])
def test_unknown_member_fails_at_the_public_boundary(
    member: Literal["data", "figure", "analysis"],
) -> None:
    with pytest.raises(ValueError, match="member"):
        ArtifactKey("run", "fit", member=member)


@pytest.mark.parametrize(
    ("root", "relative_link", "key", "point"),
    [
        ("Database", "runs/opaque_run%id", ArtifactKey("run", "data", "data"), None),
        (
            "Database",
            "runs/opaque_run%id/figures",
            ArtifactKey("run", "fit", "figure", "first"),
            None,
        ),
        (
            "Database",
            "runs/opaque_run%id/analysis",
            ArtifactKey("run", "fit", "analysis", "first"),
            None,
        ),
        (
            "Database",
            "Labber/2026/10/Data_1006",
            ArtifactKey("run", "data", "data"),
            None,
        ),
        (
            "results",
            "points/point",
            ArtifactKey("run", "fit", "figure", "first"),
            "point",
        ),
        ("results", "figures", ArtifactKey("run", "fit", "figure", "first"), None),
    ],
)
def test_outputs_reject_resolved_parent_escape_without_publishing(
    tmp_path: Path, root: str, relative_link: str, key: ArtifactKey, point: str | None
) -> None:
    save_layout = layout(tmp_path, point=point)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "evidence").write_bytes(b"untouched")
    link = tmp_path / root / "entry" / relative_link
    link.parent.mkdir(parents=True)
    link.symlink_to(outside, target_is_directory=True)
    before = tree_bytes(tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        save_layout.outputs(key)
    assert tree_bytes(tmp_path) == before


@pytest.mark.parametrize("offset", [0, 8, -5])
def test_run_id_converts_aware_time_to_utc(offset: int) -> None:
    at = datetime(2026, 10, 6, 2, 0, tzinfo=timezone(timedelta(hours=offset)))
    prefix = {0: "20261006T020000Z", 8: "20261005T180000Z", -5: "20261006T070000Z"}[
        offset
    ]
    assert re.fullmatch(prefix + r"-[0-9a-f]{6}", new_run_id(at=at))


def test_run_id_keeps_four_digit_years_for_early_aware_times() -> None:
    assert re.fullmatch(
        r"00010101T000000Z-[0-9a-f]{6}",
        new_run_id(at=datetime(1, 1, 1, tzinfo=timezone.utc)),
    )


def test_default_run_id_uses_current_utc_second() -> None:
    before = datetime.now(timezone.utc).replace(microsecond=0)
    run_id = new_run_id()
    after = datetime.now(timezone.utc).replace(microsecond=0)
    assert re.fullmatch(r"[0-9]{8}T[0-9]{6}Z-[0-9a-f]{6}", run_id)
    assert (
        before
        <= datetime.strptime(run_id[:16], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        <= after
    )


def test_naive_run_id_time_is_rejected() -> None:
    with pytest.raises(ValueError, match="at"):
        new_run_id(at=datetime(2026, 10, 6, 2, 0))


def test_naive_save_time_is_rejected_without_publishing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="saved_at"):
        SaveLayout(
            result_path=tmp_path / "results",
            database_path=tmp_path / "Database",
            run_id="run",
            point=None,
            saved_at=datetime(2026, 10, 6, 2, 0),
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("reopen", [False, True])
def test_result_entry_exposes_readonly_exact_paths_for_layout(
    entry: ResultEntry, entry_roots: tuple[Path, Path], reopen: bool
) -> None:
    results, database = entry_roots
    if reopen:
        entry = ResultEntry.open("entry", result_root=results, database_root=database)
    before = tree_bytes(results.parent)
    assert entry.result_path == results / "entry"
    assert entry.database_path == database / "entry"
    save_layout = SaveLayout(
        result_path=entry.result_path,
        database_path=entry.database_path,
        run_id="run",
        point=None,
        saved_at=SAVED_AT,
    )
    assert save_layout.outputs(ArtifactKey("run", "data", "data"))[0] == Output(
        "data_h5", database / "entry/runs/run/data.h5"
    )
    for attribute in ("result_path", "database_path"):
        with pytest.raises(AttributeError):
            setattr(entry, attribute, results / "elsewhere")
    assert tree_bytes(results.parent) == before
