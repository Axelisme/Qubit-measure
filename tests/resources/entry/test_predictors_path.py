"""Entry predictor-directory binding has no filesystem side effects."""

from pathlib import Path

import pytest
from zcu_tools.resources.entry import ResultEntry


def test_predictors_path_is_read_only_and_does_not_create_directory(
    entry: ResultEntry,
) -> None:
    expected = entry.result_path / "predictors"

    assert entry.predictors_path == expected
    assert not expected.exists()
    # Dynamic assignment tests runtime rejection without typed illegal assignment or suppression.
    property_name = "predictors_path"
    with pytest.raises(AttributeError):
        setattr(entry, property_name, Path("other"))
    assert entry.predictors_path == expected

    expected.mkdir()
    saved = expected / "existing.json"
    saved.write_text("retained", encoding="utf-8")
    assert entry.predictors_path == expected
    assert saved.read_text(encoding="utf-8") == "retained"
