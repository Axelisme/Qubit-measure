from __future__ import annotations

import re

import pytest
from qick.asm_v2 import QickParam
from zcu_tools.program.v2.base import ProgramV2Cfg
from zcu_tools.program.v2.modular import ModularProgramV2
from zcu_tools.program.v2.modules import ComputedPulse, Pulse, ScanWith
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg

from ..conftest import make_mock_soccfg

GEN_CH = 0
GEN_FREQ = 1000.0
GEN_GAIN = 0.5
GEN_NQZ = 1


def _make_prog(modules=None, sweep=None, **cfg_kwargs) -> ModularProgramV2:
    soccfg = make_mock_soccfg(n_gens=2, n_readouts=1)
    cfg = ProgramV2Cfg(**cfg_kwargs)
    return ModularProgramV2(soccfg, cfg, modules=modules or [], sweep=sweep)


def _pulse_cfg(
    *,
    ch: int = GEN_CH,
    length: float = 0.1,
    pre_delay: float | QickParam = 0.0,
    post_delay: float | QickParam = 0.0,
    gain: float = GEN_GAIN,
) -> PulseCfg:
    return PulseCfg(
        waveform=ConstWaveformCfg(length=length),
        ch=ch,
        nqz=GEN_NQZ,
        freq=GEN_FREQ,
        gain=gain,
        pre_delay=pre_delay,
        post_delay=post_delay,
    )


def test_requires_at_least_two_candidates():
    with pytest.raises(ValueError, match="at least 2"):
        ComputedPulse("cp", val_reg="gate_idx", pulses=[_pulse_cfg()])


def test_requires_one_non_none_candidate():
    with pytest.raises(ValueError, match="at least one non-None"):
        ComputedPulse("cp", val_reg="gate_idx", pulses=[None, None])


def test_none_candidate_becomes_dummy_zero_gain():
    cp = ComputedPulse(
        "cp",
        val_reg="gate_idx",
        pulses=[None, _pulse_cfg(gain=0.7, length=0.2), _pulse_cfg(length=0.1)],
    )
    _make_prog(modules=[cp], sweep=[("gate_idx", 3)])
    assert cp._pulse_modules[0].cfg is not None
    assert cp._pulse_modules[0].cfg.gain == 0.0


def test_generated_asm_contains_expected_computed_pulse_flow():
    cp = ComputedPulse(
        "cp",
        val_reg="gate_idx",
        pulses=[
            _pulse_cfg(length=0.10),
            _pulse_cfg(length=0.24),
            _pulse_cfg(length=0.16),
        ],
    )
    prog = _make_prog(modules=[cp], sweep=[("gate_idx", 3)])
    asm = prog.asm()

    # 1) runtime wave address compute from gate_idx
    addr_match = re.search(r"REG_WR r\d+ op -op\(r\d+ \+ #0\)", asm)
    assert addr_match is not None
    # 2) wave output through register-addressed wmem
    assert "WPORT_WR p0 wmem [&r0] @0" in asm
    # 3) runtime no longer looks up candidate-specific padding.
    assert "REG_WR r1 dmem" not in asm
    assert "TIME inc_ref r1" not in asm

    assert addr_match.start() < asm.index("WPORT_WR p0 wmem [&r0] @0")


def test_generated_asm_handles_nonzero_base():
    cp = ComputedPulse(
        "cp",
        val_reg="gate_idx",
        pulses=[
            _pulse_cfg(length=0.12),
            _pulse_cfg(length=0.22),
            _pulse_cfg(length=0.18),
        ],
    )
    # warm pulse pushes computed pulse waveform base away from 0
    warm = Pulse("warm", _pulse_cfg(length=0.05))
    scan = ScanWith("gate", [0, 1, 2], "gate_idx")
    scan.add_content(cp)

    prog = _make_prog(modules=[warm, scan], sweep=[])
    asm = prog.asm()

    assert cp._wmem_base > 0
    assert "WPORT_WR p0 wmem [&r" in asm

    # verify runtime address compute uses non-zero immediate
    assert re.search(rf"REG_WR r\d+ op -op\(r\d+ \+ #{cp._wmem_base}\)", asm)


def test_generated_asm_loop_block_relative_order():
    cp = ComputedPulse(
        "cp",
        val_reg="gate_idx",
        pulses=[
            _pulse_cfg(length=0.10),
            _pulse_cfg(length=0.24),
            _pulse_cfg(length=0.16),
        ],
    )
    scan = ScanWith("gate", [0, 1, 2], "gate_idx")
    scan.add_content(cp)
    prog = _make_prog(modules=[scan], sweep=[])
    asm_lines = [line.strip() for line in prog.asm().splitlines() if line.strip()]

    start = asm_lines.index("gate_count:")
    end = asm_lines.index("REG_WR s15 label gate_count")
    block = asm_lines[start : end + 1]

    patterns = [
        r"REG_WR r\d+ op -op\(r\d+\)",  # load ScanWith gate_idx from loop counter
        r"REG_WR r\d+ dmem \[&r\d+\]",  # read gate_idx value from dmem
        r"REG_WR r\d+ op -op\(r\d+ \+ #\d+\)",  # compute wmem addr
        r"WPORT_WR p0 wmem \[&r\d+\] @0",  # play selected waveform
        r"TIME #\d+ inc_ref",  # Repeat close_loop fixed overhead
        r"REG_WR s15 label gate_count",  # jump back target for loop
    ]

    pos = -1
    for pattern in patterns:
        next_pos = None
        for i in range(pos + 1, len(block)):
            if re.fullmatch(pattern, block[i]):
                next_pos = i
                break
        assert next_pos is not None, (
            f"Pattern not found in order: {pattern}, block={block}"
        )
        pos = next_pos


def test_rejects_pre_delay_mismatch():
    with pytest.raises(ValueError, match="identical pre_delay"):
        ComputedPulse(
            "cp",
            val_reg="gate_idx",
            pulses=[_pulse_cfg(pre_delay=0.0), _pulse_cfg(pre_delay=0.01)],
        )


def test_rejects_channel_mismatch():
    with pytest.raises(ValueError, match="same channel"):
        ComputedPulse(
            "cp",
            val_reg="gate_idx",
            pulses=[_pulse_cfg(ch=0), _pulse_cfg(ch=1)],
        )


def test_rejects_swept_pre_delay():
    swept = QickParam(start=0.0, spans={"gate_idx": 0.02})
    with pytest.raises(NotImplementedError, match="swept pre_delay"):
        ComputedPulse(
            "cp",
            val_reg="gate_idx",
            pulses=[_pulse_cfg(pre_delay=swept), _pulse_cfg(pre_delay=0.0)],
        )


def test_fail_fast_when_wave_indices_not_contiguous():
    # Duplicate pulse cfg => PulseRegistry reuses one pulse_id => repeated wave idx.
    same = _pulse_cfg(length=0.1)
    cp = ComputedPulse(
        "cp", val_reg="gate_idx", pulses=[same, same, _pulse_cfg(length=0.2)]
    )
    with pytest.raises(ValueError, match="indices must be contiguous"):
        _make_prog(modules=[cp], sweep=[("gate_idx", 3)])
