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
        return Native(soccfg, reps=1, initial_delay=0.0, relax_delay=0.0)
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
        # Both concrete programs inherit MyProgramV2's transport normalization.
        np.testing.assert_array_equal(prog.binprog["dmem"], values)
    else:
        assert words is None
        assert prog.binprog["dmem"] is None


def test_modular_route_protection_is_atomic_and_unprotected_routes_remain_native():
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

    prog = _program("modular", initialize)
    assert prog.gen_chs[1]["nqz"] == 2
    assert prog.ro_chs[1]["length"] == 123


def test_timeline_direct_declarations_share_settings_and_preserve_explicit_protection():
    def initialize(prog: TimelineProgramV2 | ModularProgramV2):
        prog.declare_gen(0, nqz=1)
        prog.declare_readout(0, length=0.4)
        with pytest.raises(ValueError, match="readout route 1 is not declared"):
            prog.protect_readout_routes("missing", 0, [0, 1])
        prog.protect_readout_routes("readout", 0, [0])
        # Equivalent positional/default forms share even a protected declaration.
        prog.declare_gen(0, 1)
        prog.declare_readout(0, 0.4, phase=0, sel="product")
        with pytest.raises(ValueError, match="already protected by readout"):
            prog.protect_readout_routes("other", 0, [0])
        with pytest.raises(ValueError, match="nqz"):
            prog.declare_gen(0, nqz=2)
        with pytest.raises(ValueError, match="ro_length"):
            prog.declare_readout(0, length=0.2)
        prog.declare_gen(1, nqz=1)
        prog.declare_readout(1, length=0.2)
        with pytest.raises(ValueError, match="nqz"):
            prog.declare_gen(1, nqz=2)
        with pytest.raises(ValueError, match="ro_length"):
            prog.declare_readout(1, length=0.4)

    prog = _program("timeline", initialize)
    assert prog.gen_chs[0]["nqz"] == 1
    assert prog.ro_chs[0]["length"] == 123
    assert prog.gen_chs[1]["nqz"] == 1
    assert prog.ro_chs[1]["length"] == 61
