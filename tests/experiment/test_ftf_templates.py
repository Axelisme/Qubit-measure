"""Template-to-engine composition and per-acquire FTF round semantics."""

from dataclasses import replace
from threading import Event

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from zcu_tools.core.simulation import (
    AcquireRequest,
    EnvironmentSnapshot,
    EventSlot,
    PulseEnvelope,
    ReadoutSlot,
    ReadoutWindow,
    RoundSignals,
    RoundStatistics,
    TimelineTemplate,
)
from zcu_tools.device import DeviceManager
from zcu_tools.resources.ftf import parse_ftf
from zcu_tools.simulate.evolution import Engine

from tests.resources._ftf_data import ftf_document


def template(*, tones: int = 0) -> TimelineTemplate:
    """Two points; controls stop at 4 ns, readout starts at 8/10 ns."""

    def slot(channel: int, start, duration, frequency, phase, gain) -> EventSlot:
        return EventSlot(
            channel,
            "synthetic",
            (),
            np.array(start, dtype=float),
            np.full(2, duration),
            np.full(2, frequency),
            np.full(2, phase),
            np.full(2, gain),
            (None,),
            np.zeros(2, dtype=np.int64),
            np.ones(2, dtype=bool),
            0.001,
        )

    drives = tuple(
        slot(2, [0.0, 0.002], 0.002, 250.0 + i * 100, 90.0, -0.2) for i in range(tones)
    )
    ro = slot(0, [0.008, 0.010], 0.020, 6000.0, 30.0, 0.1)
    ro = replace(ro, envelopes=(PulseEnvelope((0.0, 0.020), (2j, 2j)),))
    readout = ReadoutSlot(
        0,
        4,
        1000.0,
        np.array([[0.008], [0.010]]),
        np.full((2, 1), 6000.0),
        np.full((2, 1), 45.0),
        np.zeros((2, 1), dtype=np.int64),
    )
    return TimelineTemplate(
        np.array([0, 1], dtype=np.int64), (*drives, ro), (readout,), np.full(2, 0.030)
    )


def settings():
    value = parse_ftf(ftf_document())
    return value.model_copy(
        update={"model": value.model.model_copy(update={"retained_levels": 2})}
    )


def snapshot(value: float = 0.2) -> EnvironmentSnapshot:
    return EnvironmentSnapshot("ftf-v1", "synthetic", (("bias", value),))


def request(mode="integrated", *, templates=None) -> AcquireRequest:
    return AcquireRequest(
        mode,
        (3, 2),
        0,
        (ReadoutWindow(0, 0, 4, 1000.0, (0.008,)),),
        (),
        123,
        7,
        f"PCG64/SeedSequence-v2/numpy-{np.__version__}",
        point_dims=(2,),
        point_names=("sweep",),
        reps=3,
        templates=(template(),) if templates is None else templates,
    )


def test_template_units_common_phase_flat_readout_and_dc_model() -> None:
    import zcu_tools.experiment.ftf as ftf

    plan = ftf.plan_template(settings(), snapshot(), template(tones=1))
    track = plan.batch.tracks[0]
    np.testing.assert_array_equal(track.start_ns, [0.0, 2.0])
    np.testing.assert_array_equal(track.duration_ns, [2.0, 2.0])
    np.testing.assert_array_equal(track.frequency_ghz, [0.25, 0.25])
    np.testing.assert_allclose(track.phase_rad, [0.0, 0.0], atol=1e-14)
    np.testing.assert_array_equal(track.amplitude, [-0.2, -0.2])
    np.testing.assert_array_equal(plan.batch.end_ns, [8.0, 10.0])
    pulse = plan.readout.pulses[0][0]
    assert pulse.envelope is None
    assert pulse.amplitude == pytest.approx(0.2j)
    assert pulse.phase_rad == pytest.approx(np.pi / 6 - 12 * np.pi)
    np.testing.assert_array_equal(plan.readout.grids[0][0].times_ns, [8, 9, 10, 11])
    np.testing.assert_array_equal(
        plan.readout.model.receiver.noise_covariance, np.zeros((2, 2))
    )


@pytest.mark.parametrize("tones,path", [(0, "t4"), (3, "dop853")])
def test_template_engine_paths_match_strict_secular_reference(
    tones: int, path: str
) -> None:
    import zcu_tools.experiment.ftf as ftf

    plan = ftf.plan_template(settings(), snapshot(), template(tones=tones))
    engine = Engine(1, 1_000_000)
    try:
        actual = engine.evolve(plan.source, (plan.batch,), cancel=Event())
    finally:
        engine.close()
    assert actual.paths.segments_by_kernel[path] > 0
    model = plan.source.model((0, 0, 0))
    rates = model.rates_per_ns
    loss = rates.sum(axis=0)
    decay = (loss[:, None] + loss[None, :]) / 2 + model.dephasing_per_ns
    for p in range(2):

        def rhs(t, flat):
            rho = flat.reshape(2, 2)
            h = np.diag(model.energies_ghz).astype(complex)
            for track in plan.batch.tracks:
                if track.start_ns[p] <= t < track.start_ns[p] + track.duration_ns[p]:
                    drive = next(d for d in model.drives if d.name == track.port)
                    z = track.amplitude[p] * np.exp(
                        1j
                        * (2 * np.pi * track.frequency_ghz[p] * t + track.phase_rad[p])
                    )
                    h += z.real * drive.in_phase_ghz + z.imag * drive.quadrature_ghz
            out = -2j * np.pi * (h @ rho - rho @ h) - decay * rho
            out[np.diag_indices(2)] += rates @ rho.diagonal()
            return out.ravel()

        state = plan.source.initial_density.ravel()
        boundaries = sorted(
            {
                0.0,
                float(plan.batch.end_ns[p]),
                *(float(t.start_ns[p]) for t in plan.batch.tracks),
                *(float(t.start_ns[p] + t.duration_ns[p]) for t in plan.batch.tracks),
            }
        )
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            if end > start:
                state = solve_ivp(
                    rhs,
                    (start, end),
                    state,
                    method="DOP853",
                    rtol=1e-11,
                    atol=1e-13,
                    max_step=0.05,
                ).y[:, -1]
        np.testing.assert_allclose(
            actual.densities[p], state.reshape(2, 2), atol=1e-5, rtol=0
        )


def test_readout_ignores_controls_after_start_and_rejects_multiple_pulses() -> None:
    import zcu_tools.experiment.ftf as ftf

    base = template()
    # Controls beginning after readout must have no effect, even on another port.
    extra = replace(
        base.events[0], gen_ch=1, start_us=np.full(2, 0.015), gain=np.full(2, 0.5)
    )
    overlap = replace(base, events=(*base.events, extra, replace(extra, gen_ch=2)))
    plans = [ftf.plan_template(settings(), snapshot(), t) for t in (base, overlap)]
    engine = Engine(1, 1_000_000)
    try:
        results = [engine.evolve(p.source, (p.batch,), cancel=Event()) for p in plans]
    finally:
        engine.close()
    np.testing.assert_array_equal(results[0].densities, results[1].densities)
    np.testing.assert_array_equal(
        plans[0].readout.model.frequencies_ghz, plans[1].readout.model.frequencies_ghz
    )
    with pytest.raises(ValueError, match="one.*readout"):
        ftf.plan_template(
            settings(), snapshot(), replace(base, events=(*base.events, base.events[0]))
        )


class CountingEngine(Engine):
    def __init__(self):
        super().__init__(1, 1_000_000)
        self.calls = 0

    def evolve(self, source, batches, *, budget=1e-3, cancel):
        self.calls += 1
        return super().evolve(source, batches, budget=budget, cancel=cancel)


def test_round_statistics_reuse_snapshot_and_engine_across_acquires() -> None:
    from zcu_tools.experiment.ftf import FTFSource

    assert callable(FTFSource.open_acquire)
    engine = CountingEngine()
    source = FTFSource(settings(), DeviceManager(), engine=engine)
    try:
        session = source.open_acquire(request())
        first = session.sample_round(snapshot(), 0)
        second = session.sample_round(snapshot(), 1)
        assert isinstance(first, RoundStatistics)
        assert first.channels[0].sums.shape == (2, 1, 2)
        assert engine.calls == 1
        assert not np.array_equal(first.channels[0].sums, second.channels[0].sums)
        session.sample_round(snapshot(0.4), 2)
        assert engine.calls == 2
        source.open_acquire(request()).sample_round(snapshot(), 0)
        assert engine.calls == 3
        traces = source.open_acquire(request("decimated")).sample_round(snapshot(), 0)
        assert isinstance(traces, RoundSignals)
        assert traces.channels[0].iq.shape == (6, 1, 4, 2)
    finally:
        source.close()
