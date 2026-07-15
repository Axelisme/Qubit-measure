from __future__ import annotations

import pytest
from qick.asm_v2 import AsmInst, WriteReg
from qick.tprocv2_assembler import Assembler
from zcu_tools.program.v2.macro.wmem import ConfigReadoutFromRegs, PulseFromRegs


class _Pulse:
    def __init__(self, wave_names: list[str]) -> None:
        self._wave_names = wave_names

    def get_wavenames(self, exclude_special: bool = False) -> list[str]:
        if exclude_special:
            return [name for name in self._wave_names if name not in {"dummy", "phrst"}]
        return list(self._wave_names)


class _Prog:
    def __init__(self, wave_names: list[str]) -> None:
        self.pulses = {"readout": _Pulse(wave_names)}
        self.wave2idx = {name: i + 4 for i, name in enumerate(wave_names)}
        self.soccfg = {
            "gens": [{"tproc_ch": 7}],
            "readouts": [{"tproc_ctrl": 9}],
        }

    def _get_reg(self, name: str) -> str:
        return {"freq_word": "s1", "gain_word": "s2"}.get(name, name)


def _compile_instruction(inst: AsmInst) -> list[int]:
    command = {**inst.inst, "LINE": 1, "P_ADDR": 1}
    _binary_text, binary_words = Assembler.list2bin([command])
    return binary_words[0]


def _pulse_macro(**kwargs) -> PulseFromRegs:
    macro = PulseFromRegs(ch=0, name="readout", t=0.0, **kwargs)
    macro.t_regs["t"] = 12
    return macro


def _readout_macro(**kwargs) -> ConfigReadoutFromRegs:
    macro = ConfigReadoutFromRegs(ch=0, name="readout", t=0.0, **kwargs)
    macro.t_regs["t"] = 12
    return macro


def test_pulse_from_regs_emits_read_patch_port_sequence() -> None:
    macro = _pulse_macro(freq_reg="freq_word", gain_reg="gain_word")

    insts = macro.expand(_Prog(["readout_wave"]))

    assert [inst.inst["CMD"] for inst in insts] == [
        "REG_WR",
        "REG_WR",
        "REG_WR",
        "WPORT_WR",
    ]
    assert insts[0].inst == {
        "CMD": "REG_WR",
        "DST": "r_wave",
        "SRC": "wmem",
        "ADDR": "&4",
    }
    assert insts[1].inst == {
        "CMD": "REG_WR",
        "DST": "w0",
        "SRC": "op",
        "OP": "s1",
    }
    assert insts[2].inst == {
        "CMD": "REG_WR",
        "DST": "w3",
        "SRC": "op",
        "OP": "s2",
    }
    assert insts[3].inst == {
        "CMD": "WPORT_WR",
        "DST": "7",
        "SRC": "r_wave",
        "TIME": "@12",
    }


def test_readout_from_regs_uses_readout_control_port() -> None:
    macro = _readout_macro(freq_reg="freq_word")

    insts = macro.expand(_Prog(["readout_wave"]))

    assert insts[-1].inst == {
        "CMD": "WPORT_WR",
        "DST": "9",
        "SRC": "r_wave",
        "TIME": "@12",
    }


def test_pulse_from_regs_compiles_like_qick_wave_register_write() -> None:
    prog = _Prog(["readout_wave"])
    actual = _pulse_macro(freq_reg="freq_word").expand(prog)[1]
    oracle = WriteReg(dst="w0", src="freq_word").expand(prog)[0]

    assert _compile_instruction(actual) == _compile_instruction(oracle)


@pytest.mark.parametrize("macro_factory", [_pulse_macro, _readout_macro])
def test_wave_from_regs_ignores_special_waves(macro_factory) -> None:
    macro = macro_factory(freq_reg="freq_word")

    insts = macro.expand(_Prog(["phrst", "readout_wave"]))

    assert insts[0].inst["ADDR"] == "&5"


@pytest.mark.parametrize("macro_factory", [_pulse_macro, _readout_macro])
def test_wave_from_regs_requires_single_non_special_wave(macro_factory) -> None:
    macro = macro_factory(freq_reg="freq_word")

    with pytest.raises(NotImplementedError, match="exactly one"):
        macro.expand(_Prog(["wave0", "wave1"]))


@pytest.mark.parametrize("macro_factory", [_pulse_macro, _readout_macro])
def test_wave_from_regs_requires_a_runtime_register(macro_factory) -> None:
    macro = macro_factory()

    with pytest.raises(ValueError, match="requires at least one"):
        macro.expand(_Prog(["readout_wave"]))
