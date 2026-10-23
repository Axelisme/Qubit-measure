"""Description lifecycle through the default progress-bar factory."""

from typing import Literal

import pytest
from zcu_tools.progress_bar import BaseProgressBar, make_pbar


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
