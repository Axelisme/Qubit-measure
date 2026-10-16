"""The records are an explicit prerequisite, not a collection-time import."""

import pytest

import zcu_tools.simulate.evolution as evolution


@pytest.fixture(autouse=True)
def engine_contract_available() -> None:
    assert hasattr(evolution, "SecularModel"), "engine-side secular records are missing"
