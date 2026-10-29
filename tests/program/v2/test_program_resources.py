"""Shared dmem allocation and route ownership through concrete program APIs."""

from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
import pytest
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.program.v2.modular import ModularProgramV2
from zcu_tools.program.v2.timeline import TimelineProgramV2


def _program(
    kind: Literal["timeline", "modular"],
    initialize: Callable[[TimelineProgramV2 | ModularProgramV2], None],
):
    class Native(TimelineProgramV2):
        def _initialize(self, cfg):
            initialize(self)

        def _body(self, cfg):
            pass

    class Modular(ModularProgramV2):
        def _initialize(self, cfg):
            initialize(self)

    soccfg = make_mock_soccfg(2, 2)
    if kind == "timeline":
        return Native(
            soccfg, reps=1, initial_delay=None, final_delay=None, final_wait=None
        )
    return Modular(soccfg, modules=[], reps=1)


@pytest.mark.parametrize("kind", ["timeline", "modular"])
@pytest.mark.parametrize("values", [[], [7, -1, 42]])
def test_concrete_programs_compile_empty_and_nonempty_dmem(kind, values: Sequence[int]):
    def initialize(prog: TimelineProgramV2 | ModularProgramV2):
        assert prog.add_dmem(values[:2]) == 0
        assert prog.add_dmem(values[2:]) == len(values[:2])

    prog = _program(kind, initialize)
    words = prog.compile_datamem()
    assert prog.binprog is not None
    if values:
        assert words is not None
        assert words.dtype == np.int32
        np.testing.assert_array_equal(words, values)
        # ModularProgramV2 retains its transport list normalization; native
        # TimelineProgramV2 retains the QICK int32 array binary contract.
        np.testing.assert_array_equal(prog.binprog["dmem"], values)
    else:
        assert words is None
        assert prog.binprog["dmem"] is None


@pytest.mark.parametrize("kind", ["timeline", "modular"])
def test_route_protection_is_atomic_and_unprotected_routes_remain_native(kind):
    def initialize(prog: TimelineProgramV2 | ModularProgramV2):
        prog.declare_gen(0, nqz=1)
        prog.declare_readout(0, length=0.4)
        with pytest.raises(ValueError, match="readout route 1 is not declared"):
            prog.protect_readout_routes("missing", 0, [0, 1])
        # The failed protection must not reserve generator or ADC zero.
        prog.protect_readout_routes("readout", 0, [0])
        with pytest.raises(ValueError, match="readout: generator route 0 is protected"):
            prog.declare_gen(0, nqz=1)
        with pytest.raises(ValueError, match="readout: readout route 0 is protected"):
            prog.declare_readout(0, length=0.4)
        with pytest.raises(ValueError, match="already protected by readout"):
            prog.protect_readout_routes("other", 0, [0])
        prog.declare_gen(1, nqz=1)
        prog.declare_gen(1, nqz=2)
        prog.declare_readout(1, length=0.2)
        prog.declare_readout(1, length=0.4)

    prog = _program(kind, initialize)
    assert prog.gen_chs[1]["nqz"] == 2
    assert prog.ro_chs[1]["length"] == 123
