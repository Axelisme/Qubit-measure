"""GUI simulation parallelism policy with caller-supplied CPU availability."""

import pytest
from zcu_tools.gui.app.measure import startup_context


@pytest.mark.parametrize(
    ("cpu_count", "expected"), [(None, 1), (0, 1), (1, 1), (2, 1), (8, 7)]
)
def test_simulation_workers_reserves_one_cpu(
    cpu_count: int | None, expected: int
) -> None:
    assert startup_context.simulation_workers(cpu_count) == expected
