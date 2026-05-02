"""Smoke tests for ModularProgramV2 compilation without hardware.

These tests verify that the program class can be constructed and compiled
using a mock QickConfig built from a plain dict, exercising the real
_initialize / _body / compile pipeline.
"""
from __future__ import annotations

import pytest
from zcu_tools.program.v2.base import ProgramV2Cfg
from zcu_tools.program.v2.ir import base as ir_base
from zcu_tools.program.v2.ir.pipeline import PipeLineContext
from zcu_tools.program.v2.modular import ModularProgramV2
from zcu_tools.program.v2.modules import Delay, SoftDelay
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


def _addr_inc(inst: dict) -> int:
    return 2 if inst.get("CMD") == "WAIT" else 1


def _expected_p_addr(prog: ModularProgramV2) -> int:
    if not prog.prog_list:
        return 0
    return max(inst["P_ADDR"] + _addr_inc(inst) for inst in prog.prog_list)


def _expected_line(prog: ModularProgramV2) -> int:
    tracked_labels = sum(1 for info in prog.meta_infos if info.get("kind") == "label")
    return len(prog.prog_list) + tracked_labels


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


def test_compile_keeps_labels_out_of_prog_list():
    prog = _make_prog(sweep=[("my_loop", 10)])

    assert prog.labels
    assert all(not ("LABEL" in inst and "CMD" not in inst) for inst in prog.prog_list)


def test_compile_refreshes_program_cursors():
    prog = _make_prog(sweep=[("my_loop", 10)])

    assert prog.p_addr == _expected_p_addr(prog)
    assert prog.line == _expected_line(prog)


def test_compile_refreshes_program_cursors_with_wait():
    prog = _make_prog(modules=[Delay("wait", delay=1.0)])

    assert any(inst.get("CMD") == "WAIT" for inst in prog.prog_list)
    assert prog.p_addr == _expected_p_addr(prog)
    assert prog.line == _expected_line(prog)


def test_reps_propagated():
    prog = _make_prog(reps=4)
    assert prog.reps == 4


def test_rounds_propagated():
    prog = _make_prog(rounds=3)
    assert prog.cfg_model.rounds == 3


def test_datamem_none_when_empty():
    prog = _make_prog()
    assert prog.compile_datamem() is None


def test_compile_passes_pmem_budget_into_pipeline(monkeypatch):
    seen: dict[str, object] = {}

    class _NoopPipeline:
        def __call__(self, ir):
            return ir, PipeLineContext()

    def fake_make_default_pipeline(pmem_capacity):
        seen["pmem_capacity"] = pmem_capacity
        return _NoopPipeline()

    monkeypatch.setattr(ir_base, "make_default_pipeline", fake_make_default_pipeline)

    prog = _make_prog()

    assert "pmem_capacity" in seen
    assert seen["pmem_capacity"] == prog.tproccfg["pmem_size"]


# ---------------------------------------------------------------------------
# Parametrised channel counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_gens,n_readouts", [(1, 1), (2, 1), (4, 2)])
def test_various_channel_counts(n_gens, n_readouts):
    prog = _make_prog(n_gens=n_gens, n_readouts=n_readouts)
    assert prog.binprog is not None
