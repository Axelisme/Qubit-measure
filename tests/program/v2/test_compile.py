"""Smoke tests for ModularProgramV2 compilation without hardware.

These tests verify that the program class can be constructed and compiled
using a mock QickConfig built from a plain dict, exercising the real
_initialize / _body / compile pipeline.
"""
from __future__ import annotations

import pytest
from zcu_tools.program.v2.base import ProgramV2Cfg
from zcu_tools.program.v2.ir import IRBranch, IRDelay
from zcu_tools.program.v2.modular import ModularProgramV2
from zcu_tools.program.v2.modules import Delay, Module, SoftDelay
from zcu_tools.program.v2.sweep import SweepCfg

from .conftest import make_mock_soccfg

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prog(
    modules=None,
    sweep=None,
    n_gens: int = 2,
    n_readouts: int = 1,
    **cfg_kwargs,
) -> ModularProgramV2:
    soccfg = make_mock_soccfg(n_gens=n_gens, n_readouts=n_readouts)
    cfg = ProgramV2Cfg(**cfg_kwargs)
    return ModularProgramV2(soccfg, cfg, modules=modules or [], sweep=sweep)


# ---------------------------------------------------------------------------
# Compile-phase tests
# ---------------------------------------------------------------------------


def test_empty_program_compiles():
    prog = _make_prog()
    assert prog.binprog is not None


def test_binprog_has_pmem():
    prog = _make_prog()
    assert "pmem" in prog.binprog


def test_delay_module_compiles():
    prog = _make_prog(modules=[Delay("wait", delay=1.0)])
    assert prog.binprog is not None


def test_softdelay_module_compiles():
    prog = _make_prog(modules=[SoftDelay("soft_wait", delay=0.5)])
    assert prog.binprog is not None


def test_multiple_delays_compile():
    modules = [
        SoftDelay("pre", delay=0.2),
        Delay("mid", delay=1.0),
        SoftDelay("post", delay=0.3),
    ]
    prog = _make_prog(modules=modules)
    assert prog.binprog is not None


def test_sweep_int_compiles():
    prog = _make_prog(sweep=[("my_loop", 10)])
    assert prog.binprog is not None


def test_sweep_cfg_compiles():
    sweep_cfg = SweepCfg(start=0.0, stop=9.0, expts=10, step=1.0)
    prog = _make_prog(sweep=[("freq_sweep", sweep_cfg)])
    assert prog.binprog is not None


def test_reps_propagated():
    prog = _make_prog(reps=4)
    assert prog.reps == 4


def test_rounds_propagated():
    prog = _make_prog(rounds=3)
    assert prog.cfg_model.rounds == 3


def test_datamem_none_when_empty():
    prog = _make_prog()
    assert prog.compile_datamem() is None


# ---------------------------------------------------------------------------
# Parametrised channel counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_gens,n_readouts", [(1, 1), (2, 1), (4, 2)])
def test_various_channel_counts(n_gens, n_readouts):
    prog = _make_prog(n_gens=n_gens, n_readouts=n_readouts)
    assert prog.binprog is not None


class _BadIrBranchModule(Module):
    name = "bad_branch"

    def init(self, prog: ModularProgramV2) -> None:
        pass

    def ir_run(self, builder, t, prog):
        # Intentionally emit invalid IR to verify pipeline fail-fast behavior.
        builder._emit(IRBranch(compare_reg="sel", arms=(IRDelay(0.1),)))
        return t


def test_compile_fails_fast_on_ir_pass_error() -> None:
    with pytest.raises(RuntimeError, match="IR pass validation failed"):
        _make_prog(modules=[_BadIrBranchModule()])
