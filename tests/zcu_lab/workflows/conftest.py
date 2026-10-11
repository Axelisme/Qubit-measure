"""Function-scoped native simulator inputs for workflow seam tests."""

import pytest

from ._sim_inputs import Inputs, make_inputs


@pytest.fixture
def inputs() -> Inputs:
    return make_inputs()
