"""Tests for sim/lowering.py — module tree -> Bloch timeline lowering.

Covers:
- Pulse -> drive segment(s): const (single segment, hand-computed Omega/Delta/
  phase/t), gauss / flat_top (segment count + preserved total duration), and
  pre/post_delay -> idle segments.
- Delay / DelayAuto / SoftDelay -> free-evolution segment.
- Reset modules -> their unconditional pulse sequence; NoneReset -> nothing.
- Readout modules -> ReadoutPlan (fixed f_ro + swept-readout flag).
- End-to-end pi / pi-half rotation through bloch.evolve (the load-bearing
  correctness check that the whole gain -> Omega -> rotation chain is right).
- T1 dmem path: each sweep point's free-segment duration equals the LoadValue
  cycle recovered via cycles2us.
- Sweep resolution: amp_rabi (gain), len_rabi (length), t2ramsey (delay).
- detune_offset: a static global frame shift added to every segment's delta
  (drive + idle); detune_offset=0 reproduces the unshifted timeline.
- Deterministic Branch: the sub-sequence selected by point[compare_reg] is
  lowered (incl. a qubit pulse inside the branch defining the frame); a
  measurement-conditional / out-of-range / readout-bearing / nested branch
  fast-fails.
- Fast-fail: unknown / unsupported-control-flow modules raise.

A fixed identity-like cycles2us (cycle -> cycle * 0.01 µs) is used so the dmem
test asserts exact durations without depending on a real soccfg.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from qick.asm_v2 import QickParam
from zcu_tools.program.v2.modules.base import Module
from zcu_tools.program.v2.modules.control import Branch
from zcu_tools.program.v2.modules.delay import Delay, DelayAuto, SoftDelay
from zcu_tools.program.v2.modules.dmem import LoadValue
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import (
    DirectReadoutCfg,
    PulseReadoutCfg,
)
from zcu_tools.program.v2.modules.reset import (
    BathResetCfg,
    NoneResetCfg,
    PulseResetCfg,
    TwoPulseResetCfg,
)
from zcu_tools.program.v2.modules.waveform import (
    ArbWaveformCfg,
    ConstWaveformCfg,
    FlatTopWaveformCfg,
    GaussWaveformCfg,
)
from zcu_tools.program.v2.sim import SimParams, bloch
from zcu_tools.program.v2.sim.lowering import (
    _SHAPED_PULSE_SEGMENTS,
    UnsupportedModuleError,
    lower_point,
)
from zcu_tools.program.v2.sweep import SweepCfg
from zcu_tools.program.v2.utils import sweep2param

# Long T1/T2 so the unit pulse tests see (essentially) unitary rotations; the
# end-to-end pi tests rely on this to compare against the ideal angle.
_SIM = SimParams(
    EJ=8.5,
    EC=1.0,
    EL=0.5,
    flux_period=0.002,
    flux_half=0.001,
    T1=1.0e9,
    T2=1.0e9,
    T2_star=1.0e9,  # T2_star == T2 => gamma=0 (pure homogeneous; T2 == 2*T1 at limit)
    bare_rf=7.2,
    g=0.08,
    Ql=5000.0,
    Qi=50000.0,
    snr=10.0,
    pi_gain_len=0.4,
)

# Qubit transition handed in by the engine (GHz).  Drives are detuned relative
# to this value via delta = 2*pi*(f_qubit_MHz - f_drive_MHz).
_F_QUBIT_GHZ = 4.0


def _identity_cycles2us(cycles: int) -> float:
    """Deterministic cycle->µs map used by the dmem tests (1 cycle = 0.01 µs)."""

    return cycles * 0.01


def _readout() -> Module:
    """A minimal fixed-frequency readout module (every timeline needs one)."""

    return DirectReadoutCfg(ro_ch=0, ro_length=1.0, ro_freq=7200.0).build("ro")


def _const_pulse(
    *,
    gain: float | QickParam = 1.0,
    length: float | QickParam = 0.4,
    freq: float | QickParam = 4000.0,
    phase: float | QickParam = 0.0,
    pre_delay: float = 0.0,
    post_delay: float = 0.0,
) -> Module:
    """Build a const-waveform Pulse module with the given (possibly swept) fields."""

    cfg = PulseCfg(
        waveform=ConstWaveformCfg(length=length),
        ch=0,
        nqz=1,
        freq=freq,
        phase=phase,
        gain=gain,
        pre_delay=pre_delay,
        post_delay=post_delay,
    )
    return cfg.build("p")


class TestConstPulseSegment:
    """A const pulse maps to exactly one drive segment with hand-checked values."""

    def test_single_segment_values(self) -> None:
        pulse = _const_pulse(gain=1.0, length=0.4, freq=4000.0, phase=0.0)
        lp = lower_point(
            [pulse, _readout()], None, _SIM, _F_QUBIT_GHZ, {}, _identity_cycles2us
        )
        assert len(lp.segments) == 1
        seg = lp.segments[0]
        # Omega = (pi / pi_gain_len) * gain * 1 (const envelope peak).
        assert seg.omega == pytest.approx(math.pi / _SIM.pi_gain_len * 1.0)
        # On resonance (drive freq == f_qubit), detuning is zero.
        assert seg.delta == pytest.approx(0.0)
        assert seg.phase == pytest.approx(0.0)
        assert seg.t == pytest.approx(0.4)
        assert seg.t1 == _SIM.T1
        assert seg.t2 == _SIM.T2

    def test_detuning_sign_qubit_minus_drive(self) -> None:
        # Drive 1 MHz below the qubit -> delta = +2*pi*1 rad/µs (qubit minus drive).
        pulse = _const_pulse(freq=3999.0)
        lp = lower_point(
            [pulse, _readout()], None, _SIM, _F_QUBIT_GHZ, {}, _identity_cycles2us
        )
        assert lp.segments[0].delta == pytest.approx(2.0 * math.pi * 1.0)

    def test_phase_degrees_to_radians(self) -> None:
        pulse = _const_pulse(phase=90.0)
        lp = lower_point(
            [pulse, _readout()], None, _SIM, _F_QUBIT_GHZ, {}, _identity_cycles2us
        )
        assert lp.segments[0].phase == pytest.approx(math.pi / 2.0)

    def test_pre_and_post_delay_become_idle_segments(self) -> None:
        pulse = _const_pulse(pre_delay=0.1, post_delay=0.2)
        lp = lower_point(
            [pulse, _readout()], None, _SIM, _F_QUBIT_GHZ, {}, _identity_cycles2us
        )
        assert len(lp.segments) == 3
        pre, drive, post = lp.segments
        assert pre.omega == 0.0 and pre.t == pytest.approx(0.1)
        assert drive.omega > 0.0 and drive.t == pytest.approx(0.4)
        assert post.omega == 0.0 and post.t == pytest.approx(0.2)


class TestShapedPulseSegments:
    """Gauss / flat_top pulses discretize into many segments preserving duration."""

    def test_gauss_segment_count_and_duration(self) -> None:
        cfg = PulseCfg(
            waveform=GaussWaveformCfg(length=0.4, sigma=0.1),
            ch=0,
            nqz=1,
            freq=4000.0,
            gain=1.0,
        )
        lp = lower_point(
            [cfg.build("g"), _readout()],
            None,
            _SIM,
            _F_QUBIT_GHZ,
            {},
            _identity_cycles2us,
        )
        drive = [s for s in lp.segments if s.omega > 0.0 or s.t > 0.0]
        assert len(lp.segments) == _SHAPED_PULSE_SEGMENTS
        # Discretized sub-segments tile the full pulse length.
        assert sum(s.t for s in lp.segments) == pytest.approx(0.4)
        # Envelope peaks below the const value (Gaussian < 1 away from center).
        peak = max(s.omega for s in drive)
        assert peak == pytest.approx(math.pi / _SIM.pi_gain_len, rel=0.05)

    def test_flat_top_has_full_amplitude_flat_region(self) -> None:
        cfg = PulseCfg(
            waveform=FlatTopWaveformCfg(
                length=0.6,
                raise_waveform=GaussWaveformCfg(length=0.2, sigma=0.05),
            ),
            ch=0,
            nqz=1,
            freq=4000.0,
            gain=1.0,
        )
        lp = lower_point(
            [cfg.build("ft"), _readout()],
            None,
            _SIM,
            _F_QUBIT_GHZ,
            {},
            _identity_cycles2us,
        )
        # The flat top sits at the const-equivalent full Omega.
        full_omega = math.pi / _SIM.pi_gain_len
        assert any(s.omega == pytest.approx(full_omega) for s in lp.segments)
        # Ramp segments stay strictly below the flat-top amplitude.
        assert all(s.omega <= full_omega + 1e-9 for s in lp.segments)
        # Total duration is preserved.
        assert sum(s.t for s in lp.segments) == pytest.approx(0.6)

    def test_arb_waveform_segments_use_asset_duration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        time = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        idata = np.array([0.0, 0.5, 1.0], dtype=np.float64)

        class Info:
            duration = 2.0

        def fake_get(name: str):
            assert name == "arb_drive"
            return idata, None, time

        monkeypatch.setattr(
            "zcu_tools.meta_tool.arb_waveform.ArbWaveformDatabase.inspect",
            lambda name: Info(),
        )
        monkeypatch.setattr(
            "zcu_tools.meta_tool.arb_waveform.ArbWaveformDatabase.get", fake_get
        )

        cfg = PulseCfg(
            waveform=ArbWaveformCfg(data="arb_drive"),
            ch=0,
            nqz=1,
            freq=4000.0,
            gain=1.0,
        )
        lp = lower_point(
            [cfg.build("arb"), _readout()],
            None,
            _SIM,
            _F_QUBIT_GHZ,
            {},
            _identity_cycles2us,
        )

        assert len(lp.segments) == _SHAPED_PULSE_SEGMENTS
        assert sum(s.t for s in lp.segments) == pytest.approx(2.0)

        midpoints = (
            (np.arange(_SHAPED_PULSE_SEGMENTS, dtype=np.float64) + 0.5)
            * 2.0
            / _SHAPED_PULSE_SEGMENTS
        )
        expected_amp = np.interp(midpoints, time, idata, left=0.0, right=0.0)
        omega_scale = math.pi / _SIM.pi_gain_len
        actual_amp = np.array([seg.omega / omega_scale for seg in lp.segments])
        np.testing.assert_allclose(actual_amp, expected_amp)


class TestDelaySegments:
    """Delay / DelayAuto / SoftDelay -> a single free-evolution segment."""

    @pytest.mark.parametrize(
        "module",
        [
            Delay("d", 0.3),
            DelayAuto("d", t=0.3),
            SoftDelay("d", 0.3),
        ],
    )
    def test_delay_is_free_segment(self, module: Module) -> None:
        lp = lower_point(
            [module, _readout()], None, _SIM, _F_QUBIT_GHZ, {}, _identity_cycles2us
        )
        assert len(lp.segments) == 1
        seg = lp.segments[0]
        assert seg.omega == 0.0
        assert seg.delta == 0.0
        assert seg.t == pytest.approx(0.3)
        assert seg.t1 == _SIM.T1
        assert seg.t2 == _SIM.T2


class TestIdleFrameDetuning:
    """Idle segments precess at the single-frame detuning set by the qubit pulses.

    The frame carrier is the top-level qubit Pulse frequency; an idle/Delay
    segment must carry ``delta = 2*pi*(f_qubit - f_ref)`` so Ramsey fringes can
    accumulate.  On resonance that detuning is 0 (T1 behaviour); off resonance it
    is non-zero (the Ramsey-fringe fix).
    """

    def test_idle_zero_when_pulse_on_resonance(self) -> None:
        # Qubit pulse at f_qubit (4000 MHz) -> frame detuning 0 -> static idle.
        modules = [
            _const_pulse(freq=4000.0, length=0.2),
            Delay("wait", 0.5),
            _readout(),
        ]
        lp = lower_point(modules, None, _SIM, _F_QUBIT_GHZ, {}, _identity_cycles2us)
        (idle,) = [s for s in lp.segments if s.omega == 0.0]
        assert idle.delta == pytest.approx(0.0)

    def test_idle_carries_frame_detuning_off_resonance(self) -> None:
        # Qubit pulse 3 MHz above f_qubit -> f_ref = f_qubit + 3 -> idle delta =
        # 2*pi*(f_qubit - f_ref) = -2*pi*3 rad/µs (the precession that makes
        # Ramsey fringes; the bug had this hardcoded to 0).
        detuning_mhz = 3.0
        modules = [
            _const_pulse(freq=4000.0 + detuning_mhz, length=0.2),
            Delay("wait", 0.5),
            _readout(),
        ]
        lp = lower_point(modules, None, _SIM, _F_QUBIT_GHZ, {}, _identity_cycles2us)
        (idle,) = [s for s in lp.segments if s.omega == 0.0]
        assert idle.delta == pytest.approx(-2.0 * math.pi * detuning_mhz)
        # The drive segment and the idle share the same frame detuning (single
        # rotating frame): both equal 2*pi*(f_qubit - f_ref).
        (drive,) = [s for s in lp.segments if s.omega > 0.0]
        assert drive.delta == pytest.approx(idle.delta)

    def test_swept_pi2_phase_does_not_change_frame(self) -> None:
        # Mechanism B (real t2ramsey): the second pi/2 carries a swept phase ramp
        # but the same on-resonance freq, so the frame stays on resonance and the
        # idle delta is 0 — the fringe lives in the resolved per-point phase, not
        # the idle precession.
        dsweep = SweepCfg(start=0.0, stop=2.0, expts=3, step=1.0)
        delay_param = sweep2param("t2_delay", dsweep)
        detune_phase = 360.0 * 3.0 * delay_param
        modules = [
            _const_pulse(freq=4000.0, length=0.2),
            Delay("t2_delay", delay=delay_param),
            _const_pulse(freq=4000.0, length=0.2, phase=detune_phase),
            _readout(),
        ]
        for i, expected_phase_deg in enumerate([0.0, 1080.0, 2160.0]):
            lp = lower_point(
                modules,
                [("t2_delay", dsweep)],
                _SIM,
                _F_QUBIT_GHZ,
                {"t2_delay": i},
                _identity_cycles2us,
            )
            (idle,) = [s for s in lp.segments if s.omega == 0.0]
            assert idle.delta == pytest.approx(0.0)
            # The second pi/2's phase advances linearly with the delay index.
            drives = [s for s in lp.segments if s.omega > 0.0]
            assert drives[-1].phase == pytest.approx(math.radians(expected_phase_deg))

    def test_ambiguous_frame_fast_fails(self) -> None:
        # Two qubit pulses at different frequencies have no single rotating frame.
        modules = [
            _const_pulse(freq=4000.0, length=0.2),
            Delay("wait", 0.5),
            _const_pulse(freq=4005.0, length=0.2),
            _readout(),
        ]
        with pytest.raises(UnsupportedModuleError, match="single rotating frame"):
            lower_point(modules, None, _SIM, _F_QUBIT_GHZ, {}, _identity_cycles2us)


class TestResetSegments:
    """Active resets lower to their pulse sequence; NoneReset emits nothing."""

    def test_none_reset_emits_no_segment(self) -> None:
        reset = NoneResetCfg().build("reset")
        lp = lower_point(
            [reset, _readout()], None, _SIM, _F_QUBIT_GHZ, {}, _identity_cycles2us
        )
        assert lp.segments == []

    def test_pulse_reset_is_single_drive(self) -> None:
        pulse_cfg = PulseCfg(
            waveform=ConstWaveformCfg(length=0.4),
            ch=0,
            nqz=1,
            freq=4000.0,
            gain=1.0,
        )
        reset = PulseResetCfg(pulse_cfg=pulse_cfg).build("reset")
        lp = lower_point(
            [reset, _readout()], None, _SIM, _F_QUBIT_GHZ, {}, _identity_cycles2us
        )
        assert len(lp.segments) == 1
        assert lp.segments[0].omega > 0.0

    def test_two_pulse_reset_is_two_drives(self) -> None:
        p = PulseCfg(
            waveform=ConstWaveformCfg(length=0.4),
            ch=0,
            nqz=1,
            freq=4000.0,
            gain=1.0,
        )
        reset = TwoPulseResetCfg(pulse1_cfg=p, pulse2_cfg=p).build("reset")
        lp = lower_point(
            [reset, _readout()], None, _SIM, _F_QUBIT_GHZ, {}, _identity_cycles2us
        )
        assert len(lp.segments) == 2

    def test_bath_reset_is_three_drives(self) -> None:
        p = PulseCfg(
            waveform=ConstWaveformCfg(length=0.4),
            ch=0,
            nqz=1,
            freq=4000.0,
            gain=1.0,
        )
        reset = BathResetCfg(cavity_tone_cfg=p, qubit_tone_cfg=p, pi2_cfg=p).build(
            "reset"
        )
        lp = lower_point(
            [reset, _readout()], None, _SIM, _F_QUBIT_GHZ, {}, _identity_cycles2us
        )
        assert len(lp.segments) == 3


class TestReadoutPlan:
    """Readout modules produce a resolved semantic ReadoutPlan per point."""

    def test_direct_readout_fixed_freq(self) -> None:
        lp = lower_point(
            [
                DirectReadoutCfg(
                    ro_ch=0, ro_length=1.0, ro_freq=7200.0, trig_offset=0.25
                ).build("ro")
            ],
            None,
            _SIM,
            _F_QUBIT_GHZ,
            {},
            _identity_cycles2us,
        )
        assert lp.readout.f_ro_ghz == pytest.approx(7.2)
        assert lp.readout.ro_length_us == pytest.approx(1.0)
        assert lp.readout.trig_offset_us == pytest.approx(0.25)
        assert lp.readout.readout_gain == pytest.approx(1.0)
        assert lp.readout.pulse_cfg is None
        assert lp.readout.pulse_length_us is None

    def test_pulse_readout_reads_nested_metadata(self) -> None:
        pulse_cfg = PulseCfg(
            waveform=ConstWaveformCfg(length=1.0),
            ch=0,
            nqz=1,
            freq=7200.0,
            gain=0.25,
            pre_delay=0.15,
        )
        ro_cfg = DirectReadoutCfg(
            ro_ch=0, ro_length=1.5, ro_freq=7200.0, trig_offset=0.35
        )
        ro = PulseReadoutCfg(pulse_cfg=pulse_cfg, ro_cfg=ro_cfg).build("ro")
        lp = lower_point([ro], None, _SIM, _F_QUBIT_GHZ, {}, _identity_cycles2us)
        assert lp.readout.f_ro_ghz == pytest.approx(7.2)
        assert lp.readout.ro_length_us == pytest.approx(1.5)
        assert lp.readout.trig_offset_us == pytest.approx(0.35)
        assert lp.readout.readout_gain == pytest.approx(0.25)
        assert lp.readout.pulse_cfg is not None
        assert lp.readout.pulse_cfg.gain == pytest.approx(0.25)
        assert lp.readout.pulse_cfg.waveform.length == pytest.approx(1.0)
        assert lp.readout.pulse_length_us == pytest.approx(1.0)
        assert lp.readout.pulse_pre_delay_us == pytest.approx(0.15)

    def test_swept_readout_freq_resolves_per_point(self) -> None:
        # onetone resonator spectroscopy: ro_freq is a sweep.  Lowering resolves
        # it to *this* sweep point's value, so the engine reads back the swept
        # probe frequency point-by-point without a swept/fixed branch.
        ro_sweep = SweepCfg(start=7100.0, stop=7300.0, expts=3, step=100.0)
        ro_param = sweep2param("ro_freq", ro_sweep)
        ro = DirectReadoutCfg(ro_ch=0, ro_length=1.0, ro_freq=ro_param).build("ro")
        for idx, expected_mhz in ((0, 7100.0), (1, 7200.0), (2, 7300.0)):
            lp = lower_point(
                [ro],
                [("ro_freq", ro_sweep)],
                _SIM,
                _F_QUBIT_GHZ,
                {"ro_freq": idx},
                _identity_cycles2us,
            )
            assert lp.readout.f_ro_ghz == pytest.approx(expected_mhz / 1e3)

    def test_swept_pulse_readout_metadata_resolves_per_point(self) -> None:
        ro_sweep = SweepCfg(start=0.8, stop=1.2, expts=3, step=0.2)
        gain_sweep = SweepCfg(start=0.1, stop=0.3, expts=3, step=0.1)
        length_sweep = SweepCfg(start=0.5, stop=0.9, expts=3, step=0.2)
        freq_sweep = SweepCfg(start=7100.0, stop=7300.0, expts=3, step=100.0)

        pulse_cfg = PulseCfg(
            waveform=ConstWaveformCfg(length=sweep2param("pulse_length", length_sweep)),
            ch=0,
            nqz=1,
            freq=sweep2param("ro_freq", freq_sweep),
            gain=sweep2param("gain", gain_sweep),
        )
        ro_cfg = DirectReadoutCfg(
            ro_ch=0,
            ro_length=sweep2param("ro_length", ro_sweep),
            ro_freq=sweep2param("ro_freq", freq_sweep),
        )
        ro = PulseReadoutCfg(pulse_cfg=pulse_cfg, ro_cfg=ro_cfg).build("ro")
        sweep = [
            ("ro_freq", freq_sweep),
            ("gain", gain_sweep),
            ("pulse_length", length_sweep),
            ("ro_length", ro_sweep),
        ]

        lp = lower_point(
            [ro],
            sweep,
            _SIM,
            _F_QUBIT_GHZ,
            {"ro_freq": 2, "gain": 1, "pulse_length": 0, "ro_length": 2},
            _identity_cycles2us,
        )

        assert lp.readout.f_ro_ghz == pytest.approx(7.3)
        assert lp.readout.readout_gain == pytest.approx(0.2)
        assert lp.readout.pulse_length_us == pytest.approx(0.5)
        assert lp.readout.ro_length_us == pytest.approx(1.2)


class TestEndToEndRotation:
    """The load-bearing check: gain*length == pi_gain_len drives a full pi rotation."""

    def test_pi_rotation_inverts_population(self) -> None:
        # gain * length = 1.0 * 0.4 = pi_gain_len -> theta = pi.
        pulse = _const_pulse(gain=1.0, length=0.4, freq=4000.0)
        lp = lower_point(
            [pulse, _readout()], None, _SIM, _F_QUBIT_GHZ, {}, _identity_cycles2us
        )
        v = bloch.evolve(bloch.ground_state(0.0), lp.segments)
        assert bloch.excited_population(v) == pytest.approx(1.0, abs=1e-6)

    def test_half_pi_gives_equal_superposition(self) -> None:
        # gain * length = pi_gain_len / 2 -> theta = pi/2 -> P_e = 0.5.
        pulse = _const_pulse(gain=1.0, length=0.2, freq=4000.0)
        lp = lower_point(
            [pulse, _readout()], None, _SIM, _F_QUBIT_GHZ, {}, _identity_cycles2us
        )
        v = bloch.evolve(bloch.ground_state(0.0), lp.segments)
        assert bloch.excited_population(v) == pytest.approx(0.5, abs=1e-6)


class TestT1DmemPath:
    """The non-uniform T1 path recovers each point's delay from the dmem table."""

    def test_free_segment_matches_recovered_cycles(self) -> None:
        cycles = [10, 20, 40, 80]
        modules = [
            LoadValue(
                "load_t1_delay",
                values=cycles,
                idx_reg="length_idx",
                val_reg="t1_delay_cycle",
                auto_compress=False,
            ),
            DelayAuto("t1_delay", t="t1_delay_cycle"),
            _readout(),
        ]
        sweep = [("length_idx", len(cycles))]
        for i, cycle in enumerate(cycles):
            lp = lower_point(
                modules,
                sweep,
                _SIM,
                _F_QUBIT_GHZ,
                {"length_idx": i},
                _identity_cycles2us,
            )
            free = [s for s in lp.segments if s.omega == 0.0]
            assert len(free) == 1
            assert free[0].t == pytest.approx(_identity_cycles2us(cycle))

    def test_compressed_dmem_table_fast_fails(self) -> None:
        # auto_compress=True with >= 30 small values triggers compression, which
        # the verbatim-only lowering cannot recover.
        cycles = list(range(40))
        modules = [
            LoadValue(
                "load",
                values=cycles,
                idx_reg="length_idx",
                val_reg="t1_delay_cycle",
                auto_compress=True,
            ),
            DelayAuto("t1_delay", t="t1_delay_cycle"),
            _readout(),
        ]
        with pytest.raises(UnsupportedModuleError, match="compressed"):
            lower_point(
                modules,
                [("length_idx", len(cycles))],
                _SIM,
                _F_QUBIT_GHZ,
                {"length_idx": 0},
                _identity_cycles2us,
            )


class TestSweepResolution:
    """Each swept field resolves to the correct per-point value."""

    def test_amp_rabi_gain_sweep(self) -> None:
        gsweep = SweepCfg(start=0.0, stop=1.0, expts=3, step=0.5)
        gain_param = sweep2param("gain", gsweep)
        pulse = _const_pulse(gain=gain_param)
        for i, gain in enumerate([0.0, 0.5, 1.0]):
            lp = lower_point(
                [pulse, _readout()],
                [("gain", gsweep)],
                _SIM,
                _F_QUBIT_GHZ,
                {"gain": i},
                _identity_cycles2us,
            )
            expected = math.pi / _SIM.pi_gain_len * gain
            assert lp.segments[0].omega == pytest.approx(expected)

    def test_len_rabi_length_sweep(self) -> None:
        lsweep = SweepCfg(start=0.1, stop=0.5, expts=3, step=0.2)
        length_param = sweep2param("length", lsweep)
        pulse = _const_pulse(length=length_param)
        for i, length in enumerate([0.1, 0.3, 0.5]):
            lp = lower_point(
                [pulse, _readout()],
                [("length", lsweep)],
                _SIM,
                _F_QUBIT_GHZ,
                {"length": i},
                _identity_cycles2us,
            )
            assert lp.segments[0].t == pytest.approx(length)

    def test_t2ramsey_delay_sweep(self) -> None:
        dsweep = SweepCfg(start=0.0, stop=2.0, expts=3, step=1.0)
        delay_param = sweep2param("t2_delay", dsweep)
        modules = [
            _const_pulse(length=0.2),  # pi/2 pulse
            Delay("t2_delay", delay=delay_param),
            _readout(),
        ]
        for i, delay in enumerate([0.0, 1.0, 2.0]):
            lp = lower_point(
                modules,
                [("t2_delay", dsweep)],
                _SIM,
                _F_QUBIT_GHZ,
                {"t2_delay": i},
                _identity_cycles2us,
            )
            free = [s for s in lp.segments if s.omega == 0.0]
            assert free[0].t == pytest.approx(delay)


class TestDetuneOffset:
    """A static global frame shift adds to every segment's delta (drive + idle)."""

    def test_offset_shifts_drive_and_idle_deltas(self) -> None:
        # On-resonance pulse (frame detuning 0) + an idle delay: with an offset
        # every segment's delta equals exactly the offset.
        offset = 0.7
        modules = [
            _const_pulse(freq=4000.0, length=0.2),
            Delay("wait", 0.5),
            _readout(),
        ]
        lp = lower_point(
            modules,
            None,
            _SIM,
            _F_QUBIT_GHZ,
            {},
            _identity_cycles2us,
            detune_offset=offset,
        )
        evolution = [s for s in lp.segments if s.t > 0.0]
        for seg in evolution:
            assert seg.delta == pytest.approx(offset)

    def test_offset_adds_to_existing_frame_detuning(self) -> None:
        # Off-resonance pulse: each delta is the frame detuning plus the offset.
        detuning_mhz = 3.0
        offset = -0.4
        modules = [
            _const_pulse(freq=4000.0 + detuning_mhz, length=0.2),
            Delay("wait", 0.5),
            _readout(),
        ]
        base = lower_point(modules, None, _SIM, _F_QUBIT_GHZ, {}, _identity_cycles2us)
        shifted = lower_point(
            modules,
            None,
            _SIM,
            _F_QUBIT_GHZ,
            {},
            _identity_cycles2us,
            detune_offset=offset,
        )
        assert len(base.segments) == len(shifted.segments)
        for b, s in zip(base.segments, shifted.segments):
            assert s.delta == pytest.approx(b.delta + offset)

    def test_offset_covers_pre_post_idle_and_reset(self) -> None:
        # Pre/post idle of a pulse and reset-pulse segments all pick up the offset.
        offset = 0.25
        reset = PulseResetCfg(
            pulse_cfg=PulseCfg(
                waveform=ConstWaveformCfg(length=0.4),
                ch=0,
                nqz=1,
                freq=4000.0,
                gain=1.0,
            )
        ).build("reset")
        modules = [
            reset,
            _const_pulse(freq=4000.0, length=0.2, pre_delay=0.1, post_delay=0.2),
            _readout(),
        ]
        lp = lower_point(
            modules,
            None,
            _SIM,
            _F_QUBIT_GHZ,
            {},
            _identity_cycles2us,
            detune_offset=offset,
        )
        # On resonance, frame detuning is 0 so every segment's delta == offset.
        for seg in lp.segments:
            assert seg.delta == pytest.approx(offset)

    def test_zero_offset_is_identical_to_default(self) -> None:
        modules = [
            _const_pulse(freq=4003.0, length=0.2),
            Delay("wait", 0.5),
            _readout(),
        ]
        default = lower_point(
            modules, None, _SIM, _F_QUBIT_GHZ, {}, _identity_cycles2us
        )
        explicit_zero = lower_point(
            modules,
            None,
            _SIM,
            _F_QUBIT_GHZ,
            {},
            _identity_cycles2us,
            detune_offset=0.0,
        )
        assert len(default.segments) == len(explicit_zero.segments)
        for a, b in zip(default.segments, explicit_zero.segments):
            assert a.delta == pytest.approx(b.delta)
            assert a.omega == pytest.approx(b.omega)
            assert a.t == pytest.approx(b.t)


class TestDeterministicBranch:
    """A Branch selected by a registered sweep-loop counter lowers its sub-sequence.

    Models the real g/e prep ``Branch("ge", [], Pulse(pi))`` from
    twotone/dispersive.py: branch 0 is empty (ground), branch 1 is a pi pulse
    (excited).  ``point["ge"]`` picks the branch deterministically.
    """

    def _ge_modules(self) -> list[Module]:
        # gain * length = pi_gain_len -> a pi pulse; branch 0 empty, branch 1 the pi.
        pi_pulse = _const_pulse(gain=1.0, length=0.4, freq=4000.0)
        return [
            Branch("ge", [], pi_pulse),
            _readout(),
        ]

    def test_branch_zero_selects_empty_sequence(self) -> None:
        lp = lower_point(
            self._ge_modules(),
            [("ge", 2)],
            _SIM,
            _F_QUBIT_GHZ,
            {"ge": 0},
            _identity_cycles2us,
        )
        assert lp.segments == []

    def test_branch_one_selects_pi_pulse(self) -> None:
        lp = lower_point(
            self._ge_modules(),
            [("ge", 2)],
            _SIM,
            _F_QUBIT_GHZ,
            {"ge": 1},
            _identity_cycles2us,
        )
        assert len(lp.segments) == 1
        # A pi pulse inverts the population end-to-end.
        v = bloch.evolve(bloch.ground_state(0.0), lp.segments)
        assert bloch.excited_population(v) == pytest.approx(1.0, abs=1e-6)

    def test_compare_by_distinct_from_name(self) -> None:
        # compare_by routes selection to a different sweep axis than the name.
        pulse = _const_pulse(gain=1.0, length=0.4, freq=4000.0)
        modules = [Branch("sel", [], pulse, compare_by="ge"), _readout()]
        empty = lower_point(
            modules, [("ge", 2)], _SIM, _F_QUBIT_GHZ, {"ge": 0}, _identity_cycles2us
        )
        driven = lower_point(
            modules, [("ge", 2)], _SIM, _F_QUBIT_GHZ, {"ge": 1}, _identity_cycles2us
        )
        assert empty.segments == []
        assert len(driven.segments) == 1

    def test_branch_internal_pulse_defines_frame(self) -> None:
        # The only qubit pulse lives inside the selected branch and is detuned
        # 3 MHz; _frame_detuning must recurse into the branch to take it as f_ref,
        # so the surrounding idle precesses at that detuning (not 0).
        detuning_mhz = 3.0
        pulse = _const_pulse(freq=4000.0 + detuning_mhz, length=0.2)
        modules = [
            Branch("ge", [], pulse),
            Delay("wait", 0.5),
            _readout(),
        ]
        lp = lower_point(
            modules,
            [("ge", 2)],
            _SIM,
            _F_QUBIT_GHZ,
            {"ge": 1},
            _identity_cycles2us,
        )
        (idle,) = [s for s in lp.segments if s.omega == 0.0]
        assert idle.delta == pytest.approx(-2.0 * math.pi * detuning_mhz)

    def test_branch_empty_branch_has_no_frame_pulse(self) -> None:
        # When the empty branch is selected and no other qubit pulse exists, the
        # idle is static (frame detuning 0) — the branch-internal pulse is absent.
        pulse = _const_pulse(freq=4003.0, length=0.2)
        modules = [
            Branch("ge", [], pulse),
            Delay("wait", 0.5),
            _readout(),
        ]
        lp = lower_point(
            modules,
            [("ge", 2)],
            _SIM,
            _F_QUBIT_GHZ,
            {"ge": 0},
            _identity_cycles2us,
        )
        (idle,) = [s for s in lp.segments if s.omega == 0.0]
        assert idle.delta == pytest.approx(0.0)

    def test_three_way_branch_selects_each(self) -> None:
        # Models reset/rabi_check's 3-way Branch; assert each index picks its body.
        p1 = _const_pulse(gain=1.0, length=0.4, freq=4000.0)
        p2a = _const_pulse(gain=1.0, length=0.4, freq=4000.0)
        p2b = _const_pulse(gain=0.5, length=0.4, freq=4000.0)
        modules = [
            Branch("sel", [], p1, [p2a, p2b]),
            _readout(),
        ]
        counts = {0: 0, 1: 1, 2: 2}
        for idx, n_segments in counts.items():
            lp = lower_point(
                modules,
                [("sel", 3)],
                _SIM,
                _F_QUBIT_GHZ,
                {"sel": idx},
                _identity_cycles2us,
            )
            assert len(lp.segments) == n_segments


class TestBranchFastFail:
    """Branches that break the deterministic-counter assumption raise."""

    def test_branch_without_sweep_axis_raises(self) -> None:
        # compare_reg "ge" is not in point: not a registered sweep-loop counter
        # (the measurement-conditional shape this lowering cannot resolve).
        pulse = _const_pulse(gain=1.0)
        branch = Branch("ge", [], pulse)
        with pytest.raises(UnsupportedModuleError, match="not a sweep axis"):
            lower_point(
                [branch, _readout()],
                None,
                _SIM,
                _F_QUBIT_GHZ,
                {},
                _identity_cycles2us,
            )

    def test_branch_index_out_of_range_raises(self) -> None:
        pulse = _const_pulse(gain=1.0)
        branch = Branch("ge", [], pulse)  # 2 branches, index 5 is out of range
        with pytest.raises(UnsupportedModuleError, match="out of range"):
            lower_point(
                [branch, _readout()],
                [("ge", 2)],
                _SIM,
                _F_QUBIT_GHZ,
                {"ge": 5},
                _identity_cycles2us,
            )

    def test_readout_inside_branch_raises(self) -> None:
        branch = Branch("ge", [], _readout())
        with pytest.raises(UnsupportedModuleError, match="readout inside a branch"):
            lower_point(
                [branch, _readout()],
                [("ge", 2)],
                _SIM,
                _F_QUBIT_GHZ,
                {"ge": 1},
                _identity_cycles2us,
            )

    def test_nested_branch_raises(self) -> None:
        # A Branch inside a Branch falls through _lower_module's fast-fail: no
        # real experiment nests branches, and nesting needs selection policy the
        # single-level helper omits.
        inner = Branch("inner", [], _const_pulse(gain=1.0))
        outer = Branch("ge", [], inner)
        with pytest.raises(UnsupportedModuleError):
            lower_point(
                [outer, _readout()],
                [("ge", 2), ("inner", 2)],
                _SIM,
                _F_QUBIT_GHZ,
                {"ge": 1, "inner": 1},
                _identity_cycles2us,
            )


class TestFastFail:
    """Unsupported (control-flow) modules raise rather than being approximated."""

    def test_missing_readout_raises(self) -> None:
        with pytest.raises(UnsupportedModuleError, match="no readout"):
            lower_point(
                [_const_pulse()],
                None,
                _SIM,
                _F_QUBIT_GHZ,
                {},
                _identity_cycles2us,
            )

    def test_duplicate_readout_raises(self) -> None:
        with pytest.raises(UnsupportedModuleError, match="more than one readout"):
            lower_point(
                [_readout(), _readout()],
                None,
                _SIM,
                _F_QUBIT_GHZ,
                {},
                _identity_cycles2us,
            )
