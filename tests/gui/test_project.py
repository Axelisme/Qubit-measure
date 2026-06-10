"""Unit tests for gui/project.py shared wire helpers (Fix 4 regression).

``project_info_payload`` and ``is_real_project`` are pure functions that carry
no Qt or library dependency — no qapp fixture required.
"""

from __future__ import annotations

import pytest
from zcu_tools.gui.project import (
    DEFAULT_CHIP,
    DEFAULT_QUBIT,
    ProjectInfo,
    is_real_project,
    project_info_payload,
)

# ---------------------------------------------------------------------------
# project_info_payload
# ---------------------------------------------------------------------------


def test_project_info_payload_returns_four_fields():
    project = ProjectInfo(chip_name="Q5_2D", qub_name="Q1")
    payload = project_info_payload(project)
    assert set(payload.keys()) == {
        "chip_name",
        "qub_name",
        "result_dir",
        "database_path",
    }


def test_project_info_payload_values_match():
    project = ProjectInfo(chip_name="Q5_2D", qub_name="Q1")
    payload = project_info_payload(project)
    assert payload["chip_name"] == "Q5_2D"
    assert payload["qub_name"] == "Q1"
    # ProjectInfo derives these in __post_init__; the payload just forwards them.
    assert payload["result_dir"] == project.result_dir
    assert payload["database_path"] == project.database_path


def test_project_info_payload_with_explicit_paths():
    project = ProjectInfo(
        chip_name="A", qub_name="B", result_dir="/r", database_path="/d"
    )
    payload = project_info_payload(project)
    assert payload["result_dir"] == "/r"
    assert payload["database_path"] == "/d"


# ---------------------------------------------------------------------------
# is_real_project
# ---------------------------------------------------------------------------


def test_is_real_project_false_for_defaults():
    """A fresh ProjectInfo with the unknown_* placeholders is NOT a real project."""
    project = ProjectInfo()  # defaults to DEFAULT_CHIP / DEFAULT_QUBIT
    assert is_real_project(project) is False


def test_is_real_project_false_when_both_are_default():
    project = ProjectInfo(chip_name=DEFAULT_CHIP, qub_name=DEFAULT_QUBIT)
    assert is_real_project(project) is False


def test_is_real_project_true_for_real_chip_qub():
    project = ProjectInfo(chip_name="Q5_2D", qub_name="Q1")
    assert is_real_project(project) is True


def test_is_real_project_true_when_only_qubit_differs_from_default():
    """chip=real, qub=real → True (the pair is not the default pair)."""
    project = ProjectInfo(chip_name="MyChip", qub_name="Q2")
    assert is_real_project(project) is True


@pytest.mark.parametrize(
    "chip_name,qub_name",
    [
        (DEFAULT_CHIP, DEFAULT_QUBIT),
        ("", "Q1"),
        ("MyChip", ""),
        ("", ""),
    ],
)
def test_is_real_project_false_for_empty_or_placeholder(chip_name, qub_name):
    """Empty names and placeholder-only combos are all 'not a real project'."""
    project = ProjectInfo(chip_name=chip_name, qub_name=qub_name)
    assert is_real_project(project) is False
