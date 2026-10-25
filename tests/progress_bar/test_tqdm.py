"""Progress and description lifecycle through the default progress-bar factory."""

from typing import Literal

import pytest
from zcu_tools.progress_bar import BaseProgressBar, make_pbar


@pytest.mark.parametrize("disabled", [True, False], ids=["disabled", "enabled"])
def test_progress_lifecycle_independent_of_display(disabled: bool) -> None:
    bar = make_pbar(total=12, initial=2, desc="Rounds", disable=disabled)
    try:
        assert (bar.n, bar.total, bar.desc) == (2, 12, "Rounds")
        bar.update(0.5)
        assert bar.n == 2.5
        bar.set_progress(12)
        bar.set_progress(12)
        assert bar.n == 12
        bar.set_progress(4)
        bar.update()
        assert bar.n == 5
        bar.reset()
        assert bar.n == 0
        bar.total = 20
        bar.set_description("Acquiring")
        bar.update(3)
        bar.refresh()
        assert (bar.n, bar.total, bar.desc) == (3, 20, "Acquiring: ")
    finally:
        bar.close()
    assert (bar.n, bar.total, bar.desc) == (3, 20, "Acquiring: ")


@pytest.mark.parametrize("disabled", [True, False], ids=["disabled", "enabled"])
@pytest.mark.parametrize("description_arg", ["omitted", "keyword", "positional"])
def test_description_lifecycle(
    disabled: bool, description_arg: Literal["omitted", "keyword", "positional"]
) -> None:
    if description_arg == "omitted":
        bar: BaseProgressBar = make_pbar(total=2, disable=disabled)
        initial_description = ""
    elif description_arg == "keyword":
        bar = make_pbar(total=2, desc="Rounds", disable=disabled)
        initial_description = "Rounds"
    else:
        bar = make_pbar(None, "Rounds", total=2, disable=disabled)
        initial_description = "Rounds"

    try:
        assert bar.desc == initial_description
        bar.set_description("Acquiring")
        assert bar.desc == "Acquiring: "
        bar.set_description("")
        assert bar.desc == ""
    finally:
        bar.close()
    assert bar.desc == ""
