"""White-box optimization contracts for sim/engine.py.

These tests intentionally spy on private SimEngine helpers and routing decisions.
Public simulator physics and shape behavior stays in ``test_engine.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.program.v2.base import ProgramV2Cfg
from zcu_tools.program.v2.mocksoc import make_mock_soc
from zcu_tools.program.v2.modular import ModularProgramV2
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import DirectReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.sim import engine as engine_module
from zcu_tools.program.v2.sim.bloch import Segment
from zcu_tools.program.v2.sim.engine import (
    SimCancelledError,
    SimEngine,
    _PointModel,
    _PointReadout,
)
from zcu_tools.program.v2.sim.lowering import LoweredPoint, ReadoutPlan
from zcu_tools.program.v2.sweep import SweepCfg
from zcu_tools.program.v2.utils import sweep2param

from .test_engine import (
    _RESET_RELAX_DELAY,
    _SIM,
    _f_qubit_mhz,
    _pi_pulse_prog,
    _readout,
    _rf_g_mhz,
)


def test_engine_compute_round_supports_multiple_read_triggers() -> None:
    """nreads > 1 broadcasts deterministic readout physics over trigger slots."""

    prog = _pi_pulse_prog(relax_delay=_RESET_RELAX_DELAY, reps=3)
    ((_, ro),) = prog.ro_chs.items()
    ro["trigs"] = 2

    engine = SimEngine(prog, _SIM)
    acc = engine.compute_round(0)[0]
    _s_g, _s_e, p_e, signal_scale, noise_scale, gain_noise_scale = (
        engine._ensure_signal()
    )

    assert acc.shape == (3, 2, 2)
    assert p_e.shape == (3, 2)
    assert signal_scale.shape == (2,)
    assert noise_scale.shape == (2,)
    assert gain_noise_scale.shape == (2,)
    np.testing.assert_allclose(p_e[:, 0], p_e[:, 1])
    np.testing.assert_allclose(signal_scale[0], signal_scale[1])
    np.testing.assert_allclose(noise_scale[0], noise_scale[1])
    np.testing.assert_allclose(gain_noise_scale[0], gain_noise_scale[1])


def test_engine_lazy_compute_respects_early_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R-2: an early-stopping run never computes the rounds it does not poll.

    With lazy poll-time compute the soc asks the engine for round N only when it
    polls round N.  A stop_checker that fires after the first round halts the
    round loop, so compute_round is called exactly once even though 5 rounds were
    configured — proving the unpolled rounds' physics is never computed.
    """

    calls: list[int] = []
    completed_rounds = 0
    real_compute_round = SimEngine.compute_round

    def spy_compute_round(self: SimEngine, round_idx: int):
        nonlocal completed_rounds
        calls.append(round_idx)
        result = real_compute_round(self, round_idx)
        completed_rounds += 1
        return result

    monkeypatch.setattr(SimEngine, "compute_round", spy_compute_round)

    soc, soccfg = make_mock_soc(sim=_SIM)
    f_qubit = _f_qubit_mhz()

    pulse = PulseCfg(
        ch=0,
        nqz=1,
        gain=0.1,
        freq=f_qubit,
        phase=0.0,
        waveform=ConstWaveformCfg(length=1.0),
    ).build("qub")
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=20, rounds=5),
        modules=[pulse, _readout(_rf_g_mhz())],
    )

    # The stop_checker fires after the first round has been computed, so this
    # remains a round-boundary early-stop test rather than an intra-round cancel
    # test (covered separately below).
    prog.acquire(
        soc,
        progress=False,
        stop_checkers=[lambda: completed_rounds >= 1],
    )

    assert calls == [0], (
        f"expected exactly one round computed (early stop), got rounds {calls}"
    )


def test_acquire_stop_checker_is_checked_only_after_mock_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acquire-level stop_checkers keep hardware-like round-boundary semantics."""

    calls: list[int] = []
    real_compute_round = SimEngine.compute_round

    def spy_compute_round(self: SimEngine, round_idx: int):
        calls.append(round_idx)
        return real_compute_round(self, round_idx)

    monkeypatch.setattr(SimEngine, "compute_round", spy_compute_round)

    soc, soccfg = make_mock_soc(sim=_SIM)
    pulse = PulseCfg(
        ch=0,
        nqz=1,
        gain=0.1,
        freq=_f_qubit_mhz(),
        phase=0.0,
        waveform=ConstWaveformCfg(length=1.0),
    ).build("qub")
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=20, rounds=5),
        modules=[pulse, _readout(_rf_g_mhz())],
    )

    def stop_at_round_boundary() -> bool:
        return True

    prog.acquire(
        soc,
        progress=False,
        stop_checkers=[stop_at_round_boundary],
    )

    assert calls == [0], (
        "acquire-level stop checker should stop after the first completed round, "
        f"not before mock round compute; got rounds {calls}"
    )


def test_acquire_stop_checker_does_not_cancel_inside_mock_signal_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acquire-level stop_checkers do not interrupt one mock round mid-compute."""

    readout_calls = 0

    def fake_operating_signal(self: SimEngine) -> tuple[float, float, float]:
        return (4.0, 7.0, 7.01)

    def fake_point_readout_model(
        self: SimEngine,
        lowered: object,
        f_qubit_ghz: float,
        rf_g: float,
        rf_e: float,
        n_samples: int,
        sample_times_us: NDArray[np.float64],
    ) -> _PointReadout:
        del self, lowered, f_qubit_ghz, rf_g, rf_e, n_samples, sample_times_us
        nonlocal readout_calls
        readout_calls += 1
        return _PointReadout(
            s_g=np.array([1.0 + 0.0j], dtype=np.complex128),
            s_e=np.array([0.0 + 0.0j], dtype=np.complex128),
            signal_scale=1.0,
            noise_std_scale=1.0,
            gain_noise_std_scale=0.0,
        )

    monkeypatch.setattr(SimEngine, "_operating_signal", fake_operating_signal)
    monkeypatch.setattr(SimEngine, "_point_readout_model", fake_point_readout_model)

    soc, soccfg = make_mock_soc(sim=_SIM.model_copy(update={"poll_latency": 0.0}))
    sw = SweepCfg(start=7000.0, stop=7010.0, expts=8, step=10.0 / 7)
    ro_param = sweep2param("ro_freq", sw)
    readout = DirectReadoutCfg(ro_ch=0, ro_length=1.0, ro_freq=ro_param).build("ro")
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=1, rounds=5),
        modules=[readout],
        sweep=[("ro_freq", sw)],
    )

    def stop_after_two_points() -> bool:
        return readout_calls >= 2

    prog.acquire(soc, progress=False, stop_checkers=[stop_after_two_points])

    assert readout_calls == sw.expts


def test_engine_cancel_during_detune_loop_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Lorentzian detune ensemble loop checks stop_checkers cooperatively."""

    sim = _SIM.model_copy(update={"T2": 10.0, "T2_star": 5.0})
    soc, soccfg = make_mock_soc(sim=sim)
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=1, rounds=1),
        modules=[_readout(_rf_g_mhz())],
    )
    prog.compile()

    propagator_calls = 0

    def fake_lower(
        self: SimEngine, point: dict[str, int], f_qubit_ghz: float, detune_offset: float
    ) -> LoweredPoint:
        del self, point, f_qubit_ghz
        return LoweredPoint(
            segments=[
                Segment(
                    omega=1.0,
                    delta=detune_offset,
                    phase=0.0,
                    t=0.01,
                    t1=None,
                    t2=None,
                    equilibrium_pop=0.0,
                )
            ],
            readout=ReadoutPlan(f_ro_ghz=7.0, ro_length_us=1.0),
        )

    def fake_sequence_propagator(segments: object) -> NDArray[np.float64]:
        del segments
        nonlocal propagator_calls
        propagator_calls += 1
        return np.eye(4, dtype=np.float64)

    def stop_after_three_detune_nodes() -> bool:
        return propagator_calls >= 3

    monkeypatch.setattr(SimEngine, "_lower", fake_lower)
    monkeypatch.setattr(engine_module, "_sequence_propagator", fake_sequence_propagator)

    engine = SimEngine(prog, sim, stop_checkers=[stop_after_three_detune_nodes])
    lowered = engine._lower({}, 4.0, 0.0)
    with pytest.raises(SimCancelledError, match="cancelled"):
        engine._point_evolution_props({}, 4.0, lowered)

    assert 0 < propagator_calls < 2 * len(engine._detune_nodes)


def test_engine_caches_population_chain_for_readout_only_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sweeping only readout parameters reuses the identical qubit state chain."""

    calls = 0
    real_population_chain = SimEngine._point_population_chain

    def spy_population_chain(
        self: SimEngine,
        model: _PointModel,
        reps: int,
        nreads: int,
        *,
        use_numba: bool = True,
    ) -> NDArray[np.float64]:
        nonlocal calls
        calls += 1
        return real_population_chain(self, model, reps, nreads, use_numba=use_numba)

    monkeypatch.setattr(SimEngine, "_point_population_chain", spy_population_chain)

    _soc, soccfg = make_mock_soc(sim=_SIM)
    f_qubit = _f_qubit_mhz()
    pulse = PulseCfg(
        ch=0,
        nqz=1,
        gain=0.3,
        freq=f_qubit,
        phase=0.0,
        waveform=ConstWaveformCfg(length=0.5),
    ).build("qub")
    sw = SweepCfg(start=_rf_g_mhz() - 5.0, stop=_rf_g_mhz() + 5.0, expts=5, step=2.5)
    ro_param = sweep2param("ro_freq", sw)
    readout = DirectReadoutCfg(ro_ch=0, ro_length=1.0, ro_freq=ro_param).build("ro")
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=7, rounds=1),
        modules=[pulse, readout],
        sweep=[("ro_freq", sw)],
    )
    prog.compile()

    engine = SimEngine(prog, _SIM)
    engine._ensure_signal()

    assert calls == 1


def test_engine_reuses_evolution_lowering_for_readout_only_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Readout-only axes avoid per-detune re-lowering while sweeping readout."""

    sim = _SIM.model_copy(update={"T2": 10.0, "T2_star": 5.0})
    _soc, soccfg = make_mock_soc(sim=sim)
    f_qubit = _f_qubit_mhz()
    pulse = PulseCfg(
        ch=0,
        nqz=1,
        gain=0.3,
        freq=f_qubit,
        phase=0.0,
        waveform=ConstWaveformCfg(length=0.5),
    ).build("qub")
    sw = SweepCfg(start=_rf_g_mhz() - 5.0, stop=_rf_g_mhz() + 5.0, expts=5, step=2.5)
    ro_param = sweep2param("ro_freq", sw)
    readout = DirectReadoutCfg(ro_ch=0, ro_length=1.0, ro_freq=ro_param).build("ro")
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=7, rounds=1),
        modules=[pulse, readout],
        sweep=[("ro_freq", sw)],
    )
    prog.compile()

    lower_calls = 0
    real_lower = SimEngine._lower

    def spy_lower(
        self: SimEngine,
        point: dict[str, int],
        f_qubit_ghz: float,
        detune_offset: float,
    ):
        nonlocal lower_calls
        lower_calls += 1
        return real_lower(self, point, f_qubit_ghz, detune_offset)

    monkeypatch.setattr(SimEngine, "_lower", spy_lower)

    engine = SimEngine(prog, sim)
    engine._ensure_signal()

    assert lower_calls == sw.expts


def test_engine_readout_only_positive_temp_uses_operating_frequency() -> None:
    """White-box contract: engine boundary derives equilibrium population from Temp."""

    sim = _SIM.model_copy(update={"Temp": 0.050})
    _soc, soccfg = make_mock_soc(sim=sim)
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=5, rounds=1),
        modules=[_readout(_rf_g_mhz())],
    )
    prog.compile()
    engine = SimEngine(prog, sim)

    f_qubit_ghz, _rf_g, _rf_e = engine._operating_signal()
    expected = sim.equilibrium_excited_population(f_qubit_ghz)
    _s_g, _s_e, p_e, _signal_scale, _noise_scale, _gain_noise_scale = (
        engine._ensure_signal()
    )

    assert expected > 0.0
    np.testing.assert_allclose(p_e, np.full_like(p_e, expected), rtol=1e-12, atol=1e-12)


def test_cooperative_yield_releases_gil(monkeypatch: pytest.MonkeyPatch) -> None:
    """The mocksim CPU-loop yield hook explicitly releases the process-wide GIL."""

    sleep_calls: list[float] = []
    monkeypatch.setattr(engine_module.time, "sleep", sleep_calls.append)

    yielder = engine_module._CooperativeYield(interval_s=0.0)
    yielder()

    assert sleep_calls == [0]
    assert yielder.count == 1


def test_engine_does_not_reuse_population_chain_for_qubit_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sweeping qubit drive parameters keeps distinct rep-to-rep state chains."""

    calls = 0
    real_population_chain = SimEngine._point_population_chain

    def spy_population_chain(
        self: SimEngine,
        model: _PointModel,
        reps: int,
        nreads: int,
        *,
        use_numba: bool = True,
    ) -> NDArray[np.float64]:
        nonlocal calls
        calls += 1
        return real_population_chain(self, model, reps, nreads, use_numba=use_numba)

    monkeypatch.setattr(SimEngine, "_point_population_chain", spy_population_chain)

    _soc, soccfg = make_mock_soc(sim=_SIM)
    f_qubit = _f_qubit_mhz()
    sw = SweepCfg(start=0.0, stop=0.9, expts=4, step=0.3)
    gain_param = sweep2param("gain", sw)
    pulse = PulseCfg(
        ch=0,
        nqz=1,
        gain=gain_param,
        freq=f_qubit,
        phase=0.0,
        waveform=ConstWaveformCfg(length=1.0),
    ).build("qub")
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=7, rounds=1),
        modules=[pulse, _readout(_rf_g_mhz())],
        sweep=[("gain", sw)],
    )
    prog.compile()

    engine = SimEngine(prog, _SIM)
    engine._ensure_signal()

    assert calls == sw.expts


def test_engine_skips_numba_when_population_work_is_small(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The signal grid avoids numba setup cost for low-work population chains."""

    def fail_numba(*_args: object, **_kwargs: object) -> NDArray[np.float64]:
        raise AssertionError("numba kernel should not be used for this signal grid")

    monkeypatch.setattr(engine_module, "_population_chain_numba", fail_numba)

    sim = _SIM.model_copy(update={"T2": 10.0, "T2_star": 5.0})
    _soc, soccfg = make_mock_soc(sim=sim)
    f_qubit = _f_qubit_mhz()
    pulse = PulseCfg(
        ch=0,
        nqz=1,
        gain=0.3,
        freq=f_qubit,
        phase=0.0,
        waveform=ConstWaveformCfg(length=0.5),
    ).build("qub")
    sw = SweepCfg(start=_rf_g_mhz() - 2.0, stop=_rf_g_mhz() + 2.0, expts=3, step=2.0)
    ro_param = sweep2param("ro_freq", sw)
    readout = DirectReadoutCfg(ro_ch=0, ro_length=1.0, ro_freq=ro_param).build("ro")
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=4, rounds=1),
        modules=[pulse, readout],
        sweep=[("ro_freq", sw)],
    )
    prog.compile()

    engine = SimEngine(prog, sim)
    engine._ensure_signal()


def test_engine_uses_numba_for_large_unique_population_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Large multi-node unique qubit chains are routed through the numba kernel."""

    calls = 0

    def fake_numba(
        _pre_props: NDArray[np.float64],
        _relax_props: NDArray[np.float64],
        _weights: NDArray[np.float64],
        _equilibrium_pop: float,
        reps: int,
        nreads: int,
    ) -> NDArray[np.float64]:
        nonlocal calls
        calls += 1
        return np.zeros((reps, nreads), dtype=np.float64)

    monkeypatch.setattr(engine_module, "_NUMBA_MIN_WORK_UNITS", 1)
    monkeypatch.setattr(engine_module, "_population_chain_numba", fake_numba)

    sim = _SIM.model_copy(update={"T2": 10.0, "T2_star": 5.0})
    _soc, soccfg = make_mock_soc(sim=sim)
    f_qubit = _f_qubit_mhz()
    sw = SweepCfg(start=0.1, stop=0.3, expts=3, step=0.1)
    gain_param = sweep2param("gain", sw)
    pulse = PulseCfg(
        ch=0,
        nqz=1,
        gain=gain_param,
        freq=f_qubit,
        phase=0.0,
        waveform=ConstWaveformCfg(length=0.5),
    ).build("qub")
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=4, rounds=1),
        modules=[pulse, _readout(_rf_g_mhz())],
        sweep=[("gain", sw)],
    )
    prog.compile()

    engine = SimEngine(prog, sim)
    engine._ensure_signal()

    assert calls == sw.expts


def test_engine_batched_population_chain_matches_scalar_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The optimized detune-node recurrence preserves the scalar physics."""

    sim = _SIM.model_copy(update={"T2": 10.0, "T2_star": 5.0})
    _soc, soccfg = make_mock_soc(sim=sim)
    pulse = PulseCfg(
        ch=0,
        nqz=1,
        gain=0.2,
        freq=_f_qubit_mhz(),
        phase=0.0,
        waveform=ConstWaveformCfg(length=0.7),
    ).build("qub")
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=8, rounds=1, relax_delay=0.1),
        modules=[pulse, _readout(_rf_g_mhz())],
    )
    prog.compile()
    engine = SimEngine(prog, sim)
    n_samples, sample_times_us = engine._readout_sample_times_us()
    f_qubit_ghz, rf_g, rf_e = engine._operating_signal()
    lowered = engine._lower({}, f_qubit_ghz, 0.0)
    readout = engine._point_readout_model(
        lowered, f_qubit_ghz, rf_g, rf_e, n_samples, sample_times_us
    )
    evolution = engine._point_evolution_props({}, f_qubit_ghz, lowered)
    equilibrium_pop = sim.equilibrium_excited_population(f_qubit_ghz)
    model = engine._point_model(readout, evolution, equilibrium_pop)

    actual = engine._point_population_chain(model, reps=8, nreads=1)
    monkeypatch.setattr(engine_module, "_population_chain_numba", None)
    fallback = engine._point_population_chain(model, reps=8, nreads=1)

    z0 = 2.0 * model.equilibrium_pop - 1.0
    states = [
        np.array([0.0, 0.0, z0, 1.0], dtype=np.float64) for _ in model.pre_readout_props
    ]
    expected = np.empty((8, 1), dtype=np.float64)
    for rep_idx in range(8):
        p_mean = 0.0
        next_states: list[NDArray[np.float64]] = []
        for state, pre_prop, relax_prop, weight in zip(
            states,
            model.pre_readout_props,
            model.inter_shot_props,
            engine._detune_weights,
        ):
            at_readout = pre_prop @ state
            node_p = 0.5 * (1.0 + float(at_readout[2]))
            p_mean += float(weight) * min(max(node_p, 0.0), 1.0)
            next_states.append(relax_prop @ at_readout)
        expected[rep_idx, :] = p_mean
        states = next_states

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fallback, expected, rtol=1e-12, atol=1e-12)
