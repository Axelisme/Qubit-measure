"""White-box optimization contracts for sim/engine.py.

These tests intentionally spy on private SimEngine helpers and routing decisions.
Public simulator physics and shape behavior stays in ``test_engine.py``.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.program.base import StoppedPartialAcquireError
from zcu_tools.program.v2.base import ProgramV2Cfg
from zcu_tools.program.v2.mocksoc import make_mock_soc
from zcu_tools.program.v2.modular import ModularProgramV2
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import DirectReadoutCfg, PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.sim import engine as engine_module
from zcu_tools.program.v2.sim.bloch import Segment, apply_amplitude_damping_augmented
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


@pytest.fixture(autouse=True)
def _clear_segment_propagator_lru() -> Iterator[None]:
    engine_module._cached_segment_propagator.cache_clear()
    yield
    engine_module._cached_segment_propagator.cache_clear()


class _CancelFlag:
    def __init__(self) -> None:
        self._is_set = False

    def is_set(self) -> bool:
        return self._is_set

    def set(self) -> None:
        self._is_set = True


def _segment(
    *,
    delta: float = 0.0,
    equilibrium_pop: float = 0.0,
    omega: float = 0.0,
    duration: float = 0.2,
) -> Segment:
    return Segment(
        omega=omega,
        delta=delta,
        phase=0.1,
        t=duration,
        t1=20.0,
        t2=10.0,
        equilibrium_pop=equilibrium_pop,
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
    polls round N.  A round hook that sets the stop flag after the first round halts the
    round loop, so compute_round is called exactly once even though 5 rounds were
    configured — proving the unpolled rounds' physics is never computed.
    """

    calls: list[int] = []
    real_compute_round = SimEngine.compute_round

    def spy_compute_round(self: SimEngine, round_idx: int):
        calls.append(round_idx)
        return real_compute_round(self, round_idx)

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

    # The hook fires after the first round has been computed, so this
    # remains a round-boundary early-stop test rather than an intra-round cancel
    # test (covered separately below).
    def stop_after_first_round(_round_count: int, _raw, cancel_flag) -> None:
        cancel_flag.set()

    prog.acquire(
        soc,
        progress=False,
        round_hook=stop_after_first_round,
    )

    assert calls == [0], (
        f"expected exactly one round computed (early stop), got rounds {calls}"
    )


def test_acquire_round_hook_cancel_flag_is_checked_after_mock_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acquire-level stop flags keep hardware-like round-boundary semantics."""

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

    def stop_at_round_boundary(_round_count: int, _raw, cancel_flag) -> None:
        cancel_flag.set()

    prog.acquire(
        soc,
        progress=False,
        round_hook=stop_at_round_boundary,
    )

    assert calls == [0], (
        "acquire-level stop flag should stop after the first completed round, "
        f"not before mock round compute; got rounds {calls}"
    )


def test_acquire_stop_before_first_round_raises_stopped_partial() -> None:
    """Stopping before any completed round never averages an empty rounds buffer."""

    soc, soccfg = make_mock_soc(sim=_SIM)
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=20, rounds=5),
        modules=[_readout(_rf_g_mhz())],
    )
    cancel_flag = _CancelFlag()
    cancel_flag.set()

    with pytest.raises(StoppedPartialAcquireError, match="first round"):
        prog.acquire(soc, progress=False, cancel_flag=cancel_flag)

    assert prog.get_rounds() == []


def test_segment_propagator_lru_reuses_identical_resolved_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Segment propagator LRU keys by resolved Segment content."""

    calls = 0

    def fake_segment_propagator(
        omega: float,
        delta: float,
        phase: float,
        t: float,
        t1: float | None,
        t2: float | None,
        equilibrium_pop: float = 0.0,
    ) -> NDArray[np.float64]:
        del omega, delta, phase, t, t1, t2, equilibrium_pop
        nonlocal calls
        calls += 1
        return np.eye(4, dtype=np.float64)

    monkeypatch.setattr(
        engine_module.bloch, "segment_propagator", fake_segment_propagator
    )

    segment = _segment(delta=0.25, equilibrium_pop=0.1)
    first = engine_module._cached_segment_propagator(segment)
    second = engine_module._cached_segment_propagator(segment)

    assert calls == 1
    assert first is second


def test_segment_propagator_lru_key_includes_shifted_delta_and_equilibrium_pop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shifted detune and thermal equilibrium remain part of the LRU key."""

    calls = 0

    def fake_segment_propagator(
        omega: float,
        delta: float,
        phase: float,
        t: float,
        t1: float | None,
        t2: float | None,
        equilibrium_pop: float = 0.0,
    ) -> NDArray[np.float64]:
        del omega, delta, phase, t, t1, t2, equilibrium_pop
        nonlocal calls
        calls += 1
        return np.full((4, 4), calls, dtype=np.float64)

    monkeypatch.setattr(
        engine_module.bloch, "segment_propagator", fake_segment_propagator
    )

    base = _segment(delta=0.25, equilibrium_pop=0.1)
    same = _segment(delta=0.25, equilibrium_pop=0.1)
    shifted_delta = _segment(delta=0.30, equilibrium_pop=0.1)
    shifted_equilibrium = _segment(delta=0.25, equilibrium_pop=0.2)

    engine_module._cached_segment_propagator(base)
    engine_module._cached_segment_propagator(same)
    engine_module._cached_segment_propagator(shifted_delta)
    engine_module._cached_segment_propagator(shifted_equilibrium)

    assert calls == 3
    assert engine_module._cached_segment_propagator.cache_info().currsize == 3


def test_segment_propagator_lru_returns_readonly_value() -> None:
    """Cached propagators are immutable to prevent cache corruption."""

    prop = engine_module._cached_segment_propagator(_segment(delta=0.1))

    assert prop.flags.writeable is False
    with pytest.raises(ValueError):
        prop[0, 0] = 2.0


def test_sequence_prefix_cache_reuses_common_prefix() -> None:
    """Per-grid sequence cache shares cumulative propagators by prefix."""

    a = _segment(delta=0.0, duration=0.1)
    b = _segment(delta=0.1, duration=0.2)
    c = _segment(delta=0.2, duration=0.3)
    d = _segment(delta=0.3, duration=0.4)
    cache = engine_module._SequencePropagatorCache()

    abc = cache.propagator((a, b, c))
    assert cache.size == 4
    abd = cache.propagator((a, b, d))

    assert cache.size == 5
    assert abc.flags.writeable is False
    assert abd.flags.writeable is False
    np.testing.assert_allclose(abc, engine_module._sequence_propagator((a, b, c)))
    np.testing.assert_allclose(abd, engine_module._sequence_propagator((a, b, d)))


def test_acquire_cancel_flag_does_not_cancel_inside_mock_signal_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acquire-level stop flags discard a mock round only after poll_data returns."""

    readout_calls = 0
    cancel_flag = _CancelFlag()
    stop_tproc_calls = 0

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
        if readout_calls >= 2:
            cancel_flag.set()
        return _PointReadout(
            s_g=np.array([1.0 + 0.0j], dtype=np.complex128),
            s_e=np.array([0.0 + 0.0j], dtype=np.complex128),
            signal_scale=1.0,
            noise_std_scale=1.0,
            gain_noise_std_scale=0.0,
            readout_q_post=1.0,
        )

    monkeypatch.setattr(SimEngine, "_operating_signal", fake_operating_signal)
    monkeypatch.setattr(SimEngine, "_point_readout_model", fake_point_readout_model)

    soc, soccfg = make_mock_soc(sim=_SIM.model_copy(update={"poll_latency": 0.0}))
    real_stop_tproc = soc.stop_tproc

    def spy_stop_tproc(*args, **kwargs) -> None:
        nonlocal stop_tproc_calls
        stop_tproc_calls += 1
        real_stop_tproc(*args, **kwargs)

    soc.stop_tproc = spy_stop_tproc
    sw = SweepCfg(start=7000.0, stop=7010.0, expts=8, step=10.0 / 7)
    ro_param = sweep2param("ro_freq", sw)
    readout = DirectReadoutCfg(ro_ch=0, ro_length=1.0, ro_freq=ro_param).build("ro")
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=1, rounds=5),
        modules=[readout],
        sweep=[("ro_freq", sw)],
    )

    with pytest.raises(StoppedPartialAcquireError, match="first round"):
        prog.acquire(soc, progress=False, cancel_flag=cancel_flag)

    assert readout_calls == sw.expts
    assert prog.get_rounds() == []
    assert prog.stats == []
    assert stop_tproc_calls >= 1


def test_engine_cancel_during_detune_loop_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Lorentzian detune ensemble loop checks cancel_flag cooperatively."""

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
        if propagator_calls >= 3:
            cancel_flag.set()
        return np.eye(4, dtype=np.float64)

    monkeypatch.setattr(SimEngine, "_lower", fake_lower)
    monkeypatch.setattr(engine_module, "_sequence_propagator", fake_sequence_propagator)

    cancel_flag = _CancelFlag()
    engine = SimEngine(prog, sim, cancel_flag=cancel_flag)
    lowered = engine._lower({}, 4.0, 0.0)
    with pytest.raises(SimCancelledError, match="cancelled"):
        engine._point_evolution_props({}, 4.0, lowered)

    assert 0 < propagator_calls < 2 * len(engine._detune_nodes)


def test_engine_caches_population_chain_for_readout_only_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DirectReadout-only sweeps reuse the identical qubit state chain."""

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


def test_engine_population_cache_key_includes_readout_q_post(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PulseReadout gain sweeps cannot reuse population chains when q_post changes."""

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

    sim = _SIM.model_copy(
        update={
            "readout_decay_rate_per_us": 1.0,
            "readout_decay_threshold_ratio": 0.0,
            "readout_decay_exponent": 1.0,
        }
    )
    _soc, soccfg = make_mock_soc(sim=sim)
    sw = SweepCfg(start=0.02, stop=0.08, expts=4, step=0.02)
    gain_param = sweep2param("ro_gain", sw)
    readout = PulseReadoutCfg(
        pulse_cfg=PulseCfg(
            ch=0,
            nqz=1,
            gain=gain_param,
            freq=_rf_g_mhz(),
            phase=0.0,
            waveform=ConstWaveformCfg(length=1.0),
        ),
        ro_cfg=DirectReadoutCfg(
            ro_ch=0,
            ro_length=1.0,
            ro_freq=_rf_g_mhz(),
        ),
    ).build("ro")
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=7, rounds=1),
        modules=[readout],
        sweep=[("ro_gain", sw)],
    )
    prog.compile()

    engine = SimEngine(prog, sim)
    engine._ensure_signal()

    assert calls == sw.expts


def test_engine_scalar_population_chain_applies_readout_damping_before_relax() -> None:
    """Single-node fast path records pre-readout P_e, then damps carry state."""

    q_post = 0.4
    reps = 5
    identity = np.eye(4, dtype=np.float64)
    model = _PointModel(
        s_g=np.array([0.0 + 0.0j], dtype=np.complex128),
        s_e=np.array([1.0 + 0.0j], dtype=np.complex128),
        signal_scale=1.0,
        noise_std_scale=1.0,
        gain_noise_std_scale=0.0,
        readout_q_post=q_post,
        equilibrium_pop=1.0,
        pre_readout_props=(identity,),
        inter_shot_props=(identity,),
    )
    engine = object.__new__(SimEngine)
    engine._cancel_flag = None
    engine._detune_weights = np.ones(1, dtype=np.float64)

    actual = engine._point_population_chain(model, reps=reps, nreads=1, use_numba=False)

    # With identity pre/relax and p0=1, amplitude damping maps p -> q_post * p.
    expected = np.array(
        [[q_post**rep_idx] for rep_idx in range(reps)], dtype=np.float64
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_population_chain_q_post_one_skips_amplitude_damping_scalar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact q_post identity still records pre-readout P_e before relaxing state."""

    identity = np.eye(4, dtype=np.float64)
    relax_to_ground = np.eye(4, dtype=np.float64)
    relax_to_ground[2, 2] = 0.0
    relax_to_ground[2, 3] = -1.0
    model = _PointModel(
        s_g=np.array([0.0 + 0.0j], dtype=np.complex128),
        s_e=np.array([1.0 + 0.0j], dtype=np.complex128),
        signal_scale=1.0,
        noise_std_scale=1.0,
        gain_noise_std_scale=0.0,
        readout_q_post=1.0,
        equilibrium_pop=1.0,
        pre_readout_props=(identity,),
        inter_shot_props=(relax_to_ground,),
    )
    engine = object.__new__(SimEngine)
    engine._cancel_flag = None
    engine._detune_weights = np.ones(1, dtype=np.float64)

    def fail_damping(_state: object, _q_post: object) -> NDArray[np.float64]:
        raise AssertionError("exact q_post=1.0 should not call damping helper")

    monkeypatch.setattr(
        engine_module.bloch, "apply_amplitude_damping_augmented", fail_damping
    )

    actual = engine._point_population_chain(model, reps=3, nreads=1, use_numba=False)

    expected = np.array([[1.0], [0.0], [0.0]], dtype=np.float64)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_population_chain_q_post_one_skips_amplitude_damping_batched_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The batched Python fallback treats exact q_post=1.0 as identity."""

    identity = np.eye(4, dtype=np.float64)
    model = _PointModel(
        s_g=np.array([0.0 + 0.0j], dtype=np.complex128),
        s_e=np.array([1.0 + 0.0j], dtype=np.complex128),
        signal_scale=1.0,
        noise_std_scale=1.0,
        gain_noise_std_scale=0.0,
        readout_q_post=1.0,
        equilibrium_pop=1.0,
        pre_readout_props=(identity, identity),
        inter_shot_props=(identity, identity),
    )
    engine = object.__new__(SimEngine)
    engine._cancel_flag = None
    engine._detune_weights = np.array([0.25, 0.75], dtype=np.float64)

    def fail_damping(_state: object, _q_post: object) -> NDArray[np.float64]:
        raise AssertionError("exact q_post=1.0 should not call damping helper")

    monkeypatch.setattr(
        engine_module.bloch, "apply_amplitude_damping_augmented", fail_damping
    )

    actual = engine._point_population_chain(model, reps=4, nreads=2, use_numba=False)

    np.testing.assert_allclose(actual, np.ones((4, 2)), rtol=1e-12, atol=1e-12)


def test_population_chain_q_post_near_one_still_uses_damping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Near-one q_post remains physical damping, not the identity fast path."""

    calls = 0
    real_damping = engine_module.bloch.apply_amplitude_damping_augmented

    def spy_damping(
        state: NDArray[np.float64],
        q_post: float,
    ) -> NDArray[np.float64]:
        nonlocal calls
        calls += 1
        return real_damping(state, q_post)

    monkeypatch.setattr(
        engine_module.bloch, "apply_amplitude_damping_augmented", spy_damping
    )

    identity = np.eye(4, dtype=np.float64)
    model = _PointModel(
        s_g=np.array([0.0 + 0.0j], dtype=np.complex128),
        s_e=np.array([1.0 + 0.0j], dtype=np.complex128),
        signal_scale=1.0,
        noise_std_scale=1.0,
        gain_noise_std_scale=0.0,
        readout_q_post=float(np.nextafter(1.0, 0.0)),
        equilibrium_pop=1.0,
        pre_readout_props=(identity,),
        inter_shot_props=(identity,),
    )
    engine = object.__new__(SimEngine)
    engine._cancel_flag = None
    engine._detune_weights = np.ones(1, dtype=np.float64)

    engine._point_population_chain(model, reps=3, nreads=1, use_numba=False)

    assert calls == 3


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


def test_engine_skips_numba_for_small_single_node_population_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single-node work still falls back when below the dedicated threshold."""

    def fail_numba(*_args: object, **_kwargs: object) -> NDArray[np.float64]:
        raise AssertionError("small single-node work should not use numba")

    monkeypatch.setattr(engine_module, "_NUMBA_MIN_SINGLE_NODE_WORK_UNITS", 10_000)
    monkeypatch.setattr(engine_module, "_population_chain_numba", fail_numba)

    _soc, soccfg = make_mock_soc(sim=_SIM)
    sw = SweepCfg(start=_rf_g_mhz() - 2.0, stop=_rf_g_mhz() + 2.0, expts=3, step=2.0)
    ro_param = sweep2param("ro_freq", sw)
    readout = DirectReadoutCfg(ro_ch=0, ro_length=1.0, ro_freq=ro_param).build("ro")
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=4, rounds=1),
        modules=[readout],
        sweep=[("ro_freq", sw)],
    )
    prog.compile()

    engine = SimEngine(prog, _SIM)
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
        _readout_q_post: float,
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


def test_engine_uses_numba_for_large_single_node_unique_population_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Large single-node unique qubit chains are routed through numba."""

    calls = 0

    def fake_numba(
        _pre_props: NDArray[np.float64],
        _relax_props: NDArray[np.float64],
        _weights: NDArray[np.float64],
        _equilibrium_pop: float,
        _readout_q_post: float,
        reps: int,
        nreads: int,
    ) -> NDArray[np.float64]:
        nonlocal calls
        calls += 1
        return np.zeros((reps, nreads), dtype=np.float64)

    monkeypatch.setattr(engine_module, "_NUMBA_MIN_SINGLE_NODE_WORK_UNITS", 1)
    monkeypatch.setattr(engine_module, "_population_chain_numba", fake_numba)

    _soc, soccfg = make_mock_soc(sim=_SIM)
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

    engine = SimEngine(prog, _SIM)
    engine._ensure_signal()

    assert calls == sw.expts


def test_engine_skips_numba_when_cancel_flag_is_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct/internal stop flags keep the Python-loop cancellation boundary."""

    def fail_numba(*_args: object, **_kwargs: object) -> NDArray[np.float64]:
        raise AssertionError("stop-flag signal grids should not use numba")

    monkeypatch.setattr(engine_module, "_NUMBA_MIN_SINGLE_NODE_WORK_UNITS", 1)
    monkeypatch.setattr(engine_module, "_population_chain_numba", fail_numba)

    _soc, soccfg = make_mock_soc(sim=_SIM)
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

    engine = SimEngine(prog, _SIM, cancel_flag=_CancelFlag())
    engine._ensure_signal()


def test_engine_batched_population_chain_matches_scalar_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The optimized detune-node recurrence preserves the scalar physics."""

    sim = _SIM.model_copy(
        update={
            "T2": 10.0,
            "T2_star": 5.0,
            "readout_decay_rate_per_us": 1.0,
            "readout_decay_threshold_ratio": 0.0,
            "readout_decay_exponent": 1.0,
        }
    )
    _soc, soccfg = make_mock_soc(sim=sim)
    pulse = PulseCfg(
        ch=0,
        nqz=1,
        gain=0.2,
        freq=_f_qubit_mhz(),
        phase=0.0,
        waveform=ConstWaveformCfg(length=0.7),
    ).build("qub")
    readout = PulseReadoutCfg(
        pulse_cfg=PulseCfg(
            ch=0,
            nqz=1,
            gain=0.08,
            freq=_rf_g_mhz(),
            phase=0.0,
            waveform=ConstWaveformCfg(length=1.0),
        ),
        ro_cfg=DirectReadoutCfg(ro_ch=0, ro_length=1.0, ro_freq=_rf_g_mhz()),
    ).build("ro")
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=8, rounds=1, relax_delay=0.1),
        modules=[pulse, readout],
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
            after_readout = apply_amplitude_damping_augmented(
                at_readout, model.readout_q_post
            )
            next_states.append(relax_prop @ after_readout)
        expected[rep_idx, :] = p_mean
        states = next_states

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fallback, expected, rtol=1e-12, atol=1e-12)
