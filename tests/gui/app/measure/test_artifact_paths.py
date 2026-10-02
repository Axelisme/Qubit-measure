from pathlib import Path

import pytest
from zcu_tools.gui.app.measure.artifact_paths import (
    named_image_path,
    require_distinct_paths,
)
from zcu_tools.gui.app.measure.artifact_tracker import ArtifactKey, ArtifactKind


@pytest.mark.parametrize(
    ("name", "encoded"),
    [
        ("fit", "fit"),
        ("Fit", "%46it"),
        ("a/b", "a%2Fb"),
        ("a\\b", "a%5Cb"),
        ("%2F", "%252%46"),
        ("e\u0301", "e%CC%81"),
        ("\u00e9", "%C3%A9"),
        ("..", "%2E%2E"),
        ("CON", "%43%4F%4E"),
    ],
)
def test_named_image_path_preserves_distinct_names_without_directory_escape(
    tmp_path: Path, name: str, encoded: str
) -> None:
    key = ArtifactKey(ArtifactKind.ANALYSIS, name)
    path = Path(named_image_path(str(tmp_path / "scan.png"), key))
    assert path == tmp_path / f"scan__analysis__{encoded}.png"
    assert path.parent == tmp_path
    assert not path.exists()


def test_post_image_and_extensionless_base_use_distinct_stage(tmp_path: Path) -> None:
    key = ArtifactKey(ArtifactKind.POST_ANALYSIS, "fit")
    assert named_image_path(str(tmp_path / "scan"), key) == str(
        tmp_path / "scan__post__fit"
    )


def test_name_encoding_and_collision_check_are_case_stable(tmp_path: Path) -> None:
    names = ("fit", "Fit", "%46it", "a/b", "a%2Fb", "e\u0301", "\u00e9")
    paths = tuple(
        named_image_path(
            str(tmp_path / "scan.png"), ArtifactKey(ArtifactKind.ANALYSIS, name)
        )
        for name in names
    )
    require_distinct_paths(paths)
    assert len({Path(path).name.casefold() for path in paths}) == len(names)


@pytest.mark.parametrize(
    "path_names",
    [("FIT.png", "fit.PNG"), ("\u00e9.png", "e\u0301.PNG"), ("a.png", "./a.png")],
)
def test_case_unicode_and_resolved_path_collisions_fail_before_io(
    tmp_path: Path, path_names: tuple[str, str]
) -> None:
    paths = tuple(str(tmp_path / name) for name in path_names)
    with pytest.raises(ValueError, match="distinct paths"):
        require_distinct_paths(paths)
    assert not list(tmp_path.iterdir())


def test_invalid_name_and_empty_path_fail_before_io(tmp_path: Path) -> None:
    key = ArtifactKey(ArtifactKind.ANALYSIS, "\ud800")
    with pytest.raises(ValueError, match="UTF-8"):
        named_image_path(str(tmp_path / "scan.png"), key)
    with pytest.raises(ValueError, match="base path"):
        named_image_path("", ArtifactKey(ArtifactKind.ANALYSIS, "fit"))
    with pytest.raises(ValueError, match="Data"):
        named_image_path(str(tmp_path / "scan.png"), ArtifactKey(ArtifactKind.DATA))
    with pytest.raises(ValueError, match="not be empty"):
        require_distinct_paths(("",))
    assert not list(tmp_path.iterdir())
