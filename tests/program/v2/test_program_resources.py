from tests.program._timeline import compile_modules
"""Shared dmem allocation and route ownership through concrete program APIs."""

from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
import pytest
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.program.v2.timeline import TimelineProgramV2


def _program(
    initialize: Callable[[TimelineProgramV2], None],
):
    class Native(TimelineProgramV2):
        def _initialize(self, cfg):
            initialize(self)

        def _body(self, cfg):
            pass

    return Native(make_mock_soccfg(2, 2), reps=1, initial_delay=0.0, relax_delay=0.0)


@pytest.mark.parametrize("values", [[], [7, -1, 42]])
def test_concrete_programs_compile_empty_and_nonempty_dmem(values: Sequence[int]):
    def initialize(prog: TimelineProgramV2):
        assert prog.add_dmem(values[:2]) == 0
        assert prog.add_dmem(values[2:]) == len(values[:2])

    prog = _program(initialize)
    words = prog.compile_datamem()
    assert prog.binprog is not None
    if values:
        assert words is not None
        assert words.dtype == np.int32
        np.testing.assert_array_equal(words, values)
        # Both concrete programs inherit MyProgramV2's transport normalization.
        np.testing.assert_array_equal(prog.binprog["dmem"], values)
    else:
        assert words is None
        assert prog.binprog["dmem"] is None


