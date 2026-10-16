"""Template-to-engine composition and per-acquire FTF round semantics."""

from dataclasses import replace
from threading import Event

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from zcu_tools.core.simulation import (
    RoundSignals,
    RoundStatistics,
)
from zcu_tools.device import DeviceManager
from zcu_tools.experiment.ftf import FTFTemplatePlan
from zcu_tools.simulate.evolution import Engine

from tests.experiment._ftf_templates import (
    expect_statistics,
    request,
    settings,
    snapshot,
    template,
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

        def rhs(t, flat, p=p):
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


def continuous_flux_reference(plan: FTFTemplatePlan, point: int):
    """Strict fixed-coordinate integration with independently rotated grid baths."""
    waveform = plan.batch.flux_waveform
    assert waveform is not None, "FTF planning must supply continuous flux samples"
    first, last = waveform.offsets[point : point + 2]
    times = waveform.time_ns[first:last]
    flux = waveform.flux[first:last]
    schedule = plan.batch.flux
    first, last = schedule.offsets[point : point + 2]
    bath_times = schedule.start_ns[first:last]
    levels = schedule.level[first:last]
    expansion = plan.source.flux_expansion()
    dc = plan.source.model((0, 0, 0))
    vectors = dc.eigenvectors
    state = vectors @ plan.source.initial_density @ vectors.conj().T
    end = plan.batch.end_ns[point]
    boundaries = np.unique(
        np.r_[
            0.0,
            end,
            times,
            bath_times,
            [t.start_ns[point] for t in plan.batch.tracks],
            [t.start_ns[point] + t.duration_ns[point] for t in plan.batch.tracks],
        ]
    )
    boundaries = boundaries[(boundaries >= 0) & (boundaries <= end)]

    def rhs(time, flat):
        rho = flat.reshape(state.shape)
        value = [np.interp(time, times, flux[:, loop]) for loop in range(3)]
        h = expansion.static.copy()
        for term in expansion.terms:
            angle = np.pi * value[term.loop] * (2 if "2pi" in term.kind else 1)
            h += (
                np.cos(angle) if term.kind.startswith("cos") else np.sin(angle)
            ) * term.matrix
        for track in plan.batch.tracks:
            if (
                track.start_ns[point]
                <= time
                < track.start_ns[point] + track.duration_ns[point]
            ):
                drive = next(d for d in dc.drives if d.name == track.port)
                z = track.amplitude[point] * np.exp(
                    1j
                    * (
                        2 * np.pi * track.frequency_ghz[point] * time
                        + track.phase_rad[point]
                    )
                )
                h += (
                    vectors
                    @ (z.real * drive.in_phase_ghz + z.imag * drive.quadrature_ghz)
                    @ vectors.conj().T
                )
        bath_rho = bath.eigenvectors.conj().T @ rho @ bath.eigenvectors
        loss = bath.rates_per_ns.sum(axis=0)
        decay = (loss[:, None] + loss[None, :]) / 2 + bath.dephasing_per_ns
        dissipation = -decay * bath_rho
        dissipation[np.diag_indices(len(rho))] += (
            bath.rates_per_ns @ bath_rho.diagonal()
        )
        return (
            -2j * np.pi * (h @ rho - rho @ h)
            + bath.eigenvectors @ dissipation @ bath.eigenvectors.conj().T
        ).ravel()

    for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
        index = np.searchsorted(bath_times, (left + right) / 2, side="right") - 1
        bath = plan.source.model(tuple(int(v) for v in levels[index]))
        solution = solve_ivp(
            rhs,
            (left, right),
            state.ravel(),
            method="DOP853",
            rtol=2e-12,
            atol=2e-14,
            max_step=0.05,
        )
        assert solution.success, solution.message
        state = solution.y[:, -1].reshape(state.shape)
    return vectors.conj().T @ state @ vectors


def test_template_overlapping_flux_drive_and_idle_match_continuous_reference() -> None:
    import zcu_tools.experiment.ftf as ftf

    resource = settings()
    resource = resource.model_copy(
        update={
            "flux_sources": (
                resource.flux_sources[0].model_copy(update={"tau_ns": 4.0}),
            )
        }
    )
    driven = template(tones=1)
    drive = replace(
        driven.events[0], start_us=np.zeros(2), duration_us=np.full(2, 0.006)
    )
    flux = replace(
        drive,
        gen_ch=1,
        duration_us=np.full(2, 0.004),
        frequency_mhz=np.zeros(2),
        phase_deg=np.zeros(2),
        gain=np.full(2, 0.2),
    )
    engine = Engine(1, 1_000_000)
    try:
        for controls, path in (((drive, flux), "flux_dop853"), ((flux,), "t4")):
            plan = ftf.plan_template(
                resource,
                snapshot(),
                replace(driven, events=(*controls, driven.events[-1])),
            )
            assert plan.batch.flux_waveform is not None, (
                "FTF planning must supply continuous flux samples"
            )
            actual = engine.evolve(
                plan.source, (plan.batch,), budget=1e-3, cancel=Event()
            )
            assert actual.paths.segments_by_kernel[path] > 1
            if path == "t4":
                assert actual.paths.segments_by_kernel.get("flux_dop853", 0) == 0
            expected = np.stack([continuous_flux_reference(plan, p) for p in range(2)])
            error = np.abs(actual.densities - expected)
            print(
                f"{path}: population error={np.max(np.abs(np.diagonal(actual.densities - expected, axis1=1, axis2=2))):.6g}, low-four error={np.max(error):.6g}"
            )
            np.testing.assert_allclose(actual.densities, expected, atol=1e-3, rtol=0)
    finally:
        engine.close()


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
        first = expect_statistics(session.sample_round(snapshot(), 0))
        second = expect_statistics(session.sample_round(snapshot(), 1))
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
