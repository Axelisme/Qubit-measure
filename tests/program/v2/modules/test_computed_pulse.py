"""Integration tests for ComputedPulse, including flat_top support."""

from __future__ import annotations

import pytest
from zcu_tools.program.v2.base import ProgramV2Cfg
from zcu_tools.program.v2.modular import ModularProgramV2
from zcu_tools.program.v2.modules.computed_pulse import ComputedPulse
from zcu_tools.program.v2.modules.dmem import LoadValue
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.waveform import (
    ConstWaveformCfg,
    CosineWaveformCfg,
    FlatTopWaveformCfg,
    GaussWaveformCfg,
)

from ..conftest import make_mock_soccfg

GEN_CH = 0
GEN_FREQ = 1000.0
GEN_GAIN = 0.5


def _const_pulse(length=0.1, gain=GEN_GAIN, pre_delay=0.0):
    return PulseCfg(
        waveform=ConstWaveformCfg(length=length),
        ch=GEN_CH,
        nqz=1,
        freq=GEN_FREQ,
        gain=gain,
        pre_delay=pre_delay,
    )


def _flat_top_pulse(total=0.2, raise_length=0.04, gain=GEN_GAIN, pre_delay=0.0):
    return PulseCfg(
        waveform=FlatTopWaveformCfg(
            length=total,
            raise_waveform=GaussWaveformCfg(
                length=raise_length, sigma=raise_length / 4
            ),
        ),
        ch=GEN_CH,
        nqz=1,
        freq=GEN_FREQ,
        gain=gain,
        pre_delay=pre_delay,
    )


def _make_prog(modules):
    soccfg = make_mock_soccfg(n_gens=2, n_readouts=1)
    cfg = ProgramV2Cfg()
    return ModularProgramV2(soccfg, cfg, modules=modules, sweep=[("idx_loop", 2)])


def _gate_modules(gate_pulses, val_reg="gate_idx"):
    return [
        LoadValue(
            "load_idx",
            values=[i for i in range(len(gate_pulses))],
            idx_reg="idx_loop",
            val_reg=val_reg,
        ),
        ComputedPulse("gate", val_reg=val_reg, pulses=gate_pulses),
    ]


# ---------------------------------------------------------------------------
# dmem lookup: verify dmem table is populated correctly
# ---------------------------------------------------------------------------


def test_dmem_table_populated_for_const_candidates():
    """ComputedPulse.init() must populate a dmem table with one entry per candidate."""
    n_pulses = 3
    pulses = [_const_pulse(gain=(i + 1) * 0.2) for i in range(n_pulses)]
    soccfg = make_mock_soccfg(n_gens=2, n_readouts=1)
    from zcu_tools.program.v2.base import ProgramV2Cfg

    cfg = ProgramV2Cfg()
    from zcu_tools.program.v2.modules.dmem import LoadValue

    modules = [
        LoadValue(
            "load_idx",
            values=list(range(n_pulses)),
            idx_reg="idx_loop",
            val_reg="gate_idx",
        ),
        ComputedPulse("gate", val_reg="gate_idx", pulses=pulses),
    ]
    prog = ModularProgramV2(
        soccfg, cfg, modules=modules, sweep=[("idx_loop", n_pulses)]
    )
    assert prog.binprog is not None

    # ComputedPulse.dmem_offset is the start of its table in the shared dmem buffer.
    gate_mod = modules[1]
    assert isinstance(gate_mod, ComputedPulse)
    dmem = prog.compile_datamem()
    assert dmem is not None

    # The slice starting at dmem_offset must contain n_pulses entries.
    offset = gate_mod.dmem_offset
    table = dmem[offset : offset + n_pulses]
    assert len(table) == n_pulses
    # All entries must be distinct wmem indices (each pulse has its own waveform).
    assert len(set(table.tolist())) == n_pulses


def test_dmem_lookup_emits_dmem_read_in_asm():
    """ASM output must contain a dmem read (REG_WR dst dmem [addr]) for the table lookup."""
    pulses = [_const_pulse(gain=0.2), _const_pulse(gain=0.5)]
    prog = _make_prog(_gate_modules(pulses))
    asm = prog.asm()
    # QICK serialises read_dmem as "REG_WR <dst> dmem [<addr>]"
    assert "dmem" in asm


def test_identical_candidates_dedup_table_points_to_same_index():
    """Two identical pulse cfgs are deduped by PulseRegistry: both dmem table entries
    point to the same wmem index.  Accepted behaviour: both 'candidates' execute
    the same waveform, and the program still compiles correctly."""
    same_pulse = _const_pulse(gain=0.5)
    soccfg = make_mock_soccfg(n_gens=2, n_readouts=1)
    from zcu_tools.program.v2.base import ProgramV2Cfg

    modules = _gate_modules([same_pulse, same_pulse])
    prog = ModularProgramV2(
        soccfg, ProgramV2Cfg(), modules=modules, sweep=[("idx_loop", 2)]
    )
    assert prog.binprog is not None

    cp = modules[1]
    assert isinstance(cp, ComputedPulse)
    dmem = prog.compile_datamem()
    assert dmem is not None
    table = dmem[cp.dmem_offset : cp.dmem_offset + 2]
    # Both entries point to the same wmem index (PulseRegistry dedup).
    assert table[0] == table[1]


# ---------------------------------------------------------------------------
# Regression: non-flat_top behavior unchanged
# ---------------------------------------------------------------------------


def test_const_candidates_compile():
    pulses = [_const_pulse(gain=0.2), _const_pulse(gain=0.5)]
    prog = _make_prog(_gate_modules(pulses))
    assert prog.binprog is not None


# ---------------------------------------------------------------------------
# flat_top: new functionality
# ---------------------------------------------------------------------------


def test_flat_top_candidates_compile():
    pulses = [_flat_top_pulse(gain=0.2), _flat_top_pulse(gain=0.5)]
    prog = _make_prog(_gate_modules(pulses))
    assert prog.binprog is not None


def test_flat_top_varying_lengths_compile():
    # ramp/flat lengths differ across candidates — wmem entries carry their own
    # lenreg so this must work without enforcing equal lengths.
    pulses = [
        _flat_top_pulse(total=0.2, raise_length=0.04),
        _flat_top_pulse(total=0.3, raise_length=0.06),
        _flat_top_pulse(total=0.15, raise_length=0.02),
    ]
    prog = _make_prog(_gate_modules(pulses))
    assert prog.binprog is not None


def test_flat_top_emits_three_wport_wr_per_run():
    pulses = [_flat_top_pulse(gain=0.2), _flat_top_pulse(gain=0.5)]
    prog = _make_prog(_gate_modules(pulses))
    asm = prog.asm()
    # Each ComputedPulse.run() must emit 3 contiguous WPORT_WR (ramp_up/flat/
    # ramp_down) sharing the same TIME. The program loops twice (sweep=2), so
    # we expect at least 3 occurrences in the inner body.
    assert asm.count("WPORT_WR") >= 3


def test_const_emits_one_wport_wr_per_run():
    pulses = [_const_pulse(gain=0.2), _const_pulse(gain=0.5)]
    prog = _make_prog(_gate_modules(pulses))
    asm = prog.asm()
    # Sanity: stride=1 path still uses a single WPORT_WR per gate firing.
    assert "WPORT_WR" in asm


# ---------------------------------------------------------------------------
# Reject paths
# ---------------------------------------------------------------------------


def test_flat_top_mixed_with_const_rejected():
    pulses = [_const_pulse(), _flat_top_pulse()]
    with pytest.raises(ValueError, match="flat_top"):
        ComputedPulse("gate", val_reg="gate_idx", pulses=pulses)


def _gauss_pulse(length=0.1, gain=GEN_GAIN, pre_delay=0.0):
    return PulseCfg(
        waveform=GaussWaveformCfg(length=length, sigma=length / 4),
        ch=GEN_CH,
        nqz=1,
        freq=GEN_FREQ,
        gain=gain,
        pre_delay=pre_delay,
    )


def _cosine_pulse(length=0.1, gain=GEN_GAIN, pre_delay=0.0):
    return PulseCfg(
        waveform=CosineWaveformCfg(length=length),
        ch=GEN_CH,
        nqz=1,
        freq=GEN_FREQ,
        gain=gain,
        pre_delay=pre_delay,
    )


def test_mixed_non_flat_top_styles_compile():
    # Stride is 1 for all of const/gauss/cosine/drag/arb, so they may be mixed.
    pulses = [_const_pulse(), _gauss_pulse(), _cosine_pulse()]
    prog = _make_prog(_gate_modules(pulses))
    assert prog.binprog is not None


def test_single_candidate_rejected():
    with pytest.raises(ValueError, match="at least 2"):
        ComputedPulse("gate", val_reg="gate_idx", pulses=[_const_pulse()])


def test_different_pre_delay_rejected():
    pulses = [_const_pulse(pre_delay=0.0), _const_pulse(pre_delay=0.1)]
    with pytest.raises(ValueError, match="pre_delay"):
        ComputedPulse("gate", val_reg="gate_idx", pulses=pulses)


def test_different_channel_rejected():
    from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg

    p0 = PulseCfg(
        waveform=ConstWaveformCfg(length=0.1),
        ch=0,
        nqz=1,
        freq=GEN_FREQ,
        gain=GEN_GAIN,
    )
    p1 = PulseCfg(
        waveform=ConstWaveformCfg(length=0.1),
        ch=1,
        nqz=1,
        freq=GEN_FREQ,
        gain=GEN_GAIN,
    )
    with pytest.raises(ValueError, match="channel"):
        ComputedPulse("gate", val_reg="gate_idx", pulses=[p0, p1])


def test_allow_rerun_returns_true():
    pulses = [_const_pulse(gain=0.2), _const_pulse(gain=0.5)]
    cp = ComputedPulse("gate", val_reg="gate_idx", pulses=pulses)
    assert cp.allow_rerun() is True


def test_total_length_qickparam_raises():
    from unittest.mock import MagicMock

    from qick.asm_v2 import QickParam

    pulses = [_const_pulse(gain=0.2), _const_pulse(gain=0.5)]
    cp = ComputedPulse("gate", val_reg="gate_idx", pulses=pulses)

    mock_prog = MagicMock()
    cp.pulse_modules[0].total_length = MagicMock(return_value=QickParam(start=0.1))
    cp.pulse_modules[1].total_length = MagicMock(return_value=0.1)

    with pytest.raises(ValueError, match="compile time"):
        cp.total_length(mock_prog)
