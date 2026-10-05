"""Repeat and ComputedPulse lower to the pulses they play at each sweep point.

A register-driven ``Repeat`` takes its count from a LoadValue table, and a
``ComputedPulse`` takes its candidate index from one; both are static per sweep
point, so the Bloch timeline holds the unrolled / selected pulses.
"""

from __future__ import annotations

import math

import pytest
from zcu_tools.program.v2.modules.base import Module
from zcu_tools.program.v2.modules.computed_pulse import ComputedPulse
from zcu_tools.program.v2.modules.control import Branch, Repeat
from zcu_tools.program.v2.modules.delay import Delay
from zcu_tools.program.v2.modules.dmem import LoadValue, LoadWord
from zcu_tools.program.v2.modules.pulse import Pulse, PulseCfg
from zcu_tools.program.v2.modules.readout import DirectReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.sim import SimParams, bloch
from zcu_tools.program.v2.sim.lowering import (
    LoweredPoint,
    UnsupportedModuleError,
    lower_point,
)

# Effectively infinite T1/T2: rotations stay unitary.
_SIM = SimParams(
    EJ=8.5,
    EC=1.0,
    EL=0.5,
    flux_period=0.002,
    flux_half=0.001,
    T1=1.0e9,
    T2=1.0e9,
    T2_star=1.0e9,
    bare_rf=7.2,
    g=0.08,
    Ql=5000.0,
    Qi=50000.0,
    snr=10.0,
    pi_gain_len=0.4,
)
_F_QUBIT_GHZ = 4.0
_F_QUBIT_MHZ = 4000.0


def _cycles2us(cycles: int) -> float:
    return cycles * 0.01


def _readout() -> Module:
    return DirectReadoutCfg(ro_ch=0, ro_length=1.0, ro_freq=7200.0).build("ro")


def _pulse_cfg(
    *, gain: float, length: float = 0.4, freq: float = _F_QUBIT_MHZ
) -> PulseCfg:
    # gain * length == pi_gain_len (0.4) is a pi rotation.
    return PulseCfg(
        waveform=ConstWaveformCfg(length=length),
        ch=0,
        nqz=1,
        freq=freq,
        phase=0.0,
        gain=gain,
    )


def _lower(
    modules: list[Module], sweep: list[tuple[str, int]], point: dict[str, int]
) -> LoweredPoint:
    return lower_point(modules, sweep, _SIM, _F_QUBIT_GHZ, point, _cycles2us)


def _excited(lowered: LoweredPoint) -> float:
    return bloch.excited_population(
        bloch.evolve(bloch.ground_state(0.0), lowered.segments)
    )


def _counted_repeat(counts: list[int], body: Module) -> list[Module]:
    return [
        LoadValue("load_count", values=counts, idx_reg="times", val_reg="count"),
        Repeat("loop", n="count").add_content(body),
        _readout(),
    ]


class TestRepeat:
    @pytest.mark.parametrize("times", [0, 1, 2, 3, 4])
    def test_register_count_unrolls_body(self, times: int) -> None:
        half_pi = Pulse("x90", _pulse_cfg(gain=0.5))
        modules = _counted_repeat([0, 1, 2, 3, 4], half_pi)

        lowered = _lower(modules, [("times", 5)], {"times": times})

        assert len(lowered.segments) == times
        assert _excited(lowered) == pytest.approx(
            math.sin(times * math.pi / 4) ** 2, abs=1e-6
        )

    def test_int_count_and_nested_repeat_multiply(self) -> None:
        inner = Repeat("inner", n=2).add_content(Pulse("x90", _pulse_cfg(gain=0.5)))
        outer = Repeat("outer", n=2).add_content(inner)

        lowered = _lower([outer, _readout()], [], {})

        assert len(lowered.segments) == 4
        assert _excited(lowered) == pytest.approx(0.0, abs=1e-6)

    def test_pulse_inside_repeat_defines_the_frame(self) -> None:
        modules = _counted_repeat(
            [1], Pulse("x180", _pulse_cfg(gain=1.0, freq=_F_QUBIT_MHZ + 1.0))
        )
        modules.insert(-1, Delay("wait", 0.5))

        lowered = _lower(modules, [("times", 1)], {"times": 0})

        idle = lowered.segments[-1]
        assert idle.omega == 0.0
        assert idle.delta == pytest.approx(-2.0 * math.pi)

    def test_repeat_inside_selected_branch_unrolls(self) -> None:
        repeat = Repeat("loop", n=2).add_content(Pulse("x90", _pulse_cfg(gain=0.5)))
        modules = [Branch("ge", [], repeat), _readout()]

        lowered = _lower(modules, [("ge", 2)], {"ge": 1})

        assert _excited(lowered) == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize(
        "body",
        [_readout(), Branch("ge", [], Pulse("x180", _pulse_cfg(gain=1.0)))],
        ids=["readout", "branch"],
    )
    def test_readout_or_branch_inside_repeat_raises(self, body: Module) -> None:
        repeat = Repeat("loop", n=2).add_content(body)

        with pytest.raises(UnsupportedModuleError, match="only evolution modules"):
            _lower([repeat, _readout()], [("ge", 2)], {"ge": 0})

    def test_register_count_without_table_raises(self) -> None:
        repeat = Repeat("loop", n="count").add_content(
            Pulse("x90", _pulse_cfg(gain=0.5))
        )

        with pytest.raises(UnsupportedModuleError, match="no LoadValue populates"):
            _lower([repeat, _readout()], [], {})

    def test_register_count_from_load_word_raises(self) -> None:
        modules = [
            LoadWord("load_count", values=[1], idx_reg="times", val_reg="count"),
            Repeat("loop", n="count").add_content(Pulse("x90", _pulse_cfg(gain=0.5))),
            _readout(),
        ]

        with pytest.raises(UnsupportedModuleError, match="LoadWord"):
            _lower(modules, [("times", 1)], {"times": 0})


def _gates(candidates: list[PulseCfg], table: list[int]) -> list[Module]:
    return [
        LoadValue("load_gate", values=table, idx_reg="gate", val_reg="gate_idx"),
        ComputedPulse("gate", val_reg="gate_idx", pulses=candidates),
        _readout(),
    ]


class TestComputedPulse:
    @pytest.mark.parametrize(
        ("gate", "excited"), [(0, 0.0), (1, 0.5), (2, 1.0)], ids=["I", "X90", "X180"]
    )
    def test_table_selects_candidate(self, gate: int, excited: float) -> None:
        candidates = [_pulse_cfg(gain=g) for g in (0.0, 0.5, 1.0)]
        modules = _gates(candidates, [0, 1, 2])

        lowered = _lower(modules, [("gate", 3)], {"gate": gate})

        assert _excited(lowered) == pytest.approx(excited, abs=1e-6)

    @pytest.mark.parametrize("gate", [0, 1])
    def test_shorter_candidate_idles_to_longest_length(self, gate: int) -> None:
        candidates = [
            _pulse_cfg(gain=1.0, length=0.4),
            _pulse_cfg(gain=1.0, length=0.2),
        ]
        modules = _gates(candidates, [0, 1])

        lowered = _lower(modules, [("gate", 2)], {"gate": gate})

        assert sum(s.t for s in lowered.segments) == pytest.approx(0.4)

    def test_candidate_defines_the_frame(self) -> None:
        candidates = [_pulse_cfg(gain=g, freq=_F_QUBIT_MHZ + 1.0) for g in (0.5, 1.0)]
        modules = _gates(candidates, [1])
        modules.insert(-1, Delay("wait", 0.5))

        lowered = _lower(modules, [("gate", 1)], {"gate": 0})

        idle = lowered.segments[-1]
        assert idle.omega == 0.0
        assert idle.delta == pytest.approx(-2.0 * math.pi)

    def test_candidate_index_out_of_range_raises(self) -> None:
        candidates = [_pulse_cfg(gain=g) for g in (0.5, 1.0)]
        modules = _gates(candidates, [5])

        with pytest.raises(UnsupportedModuleError, match="out of range"):
            _lower(modules, [("gate", 1)], {"gate": 0})
