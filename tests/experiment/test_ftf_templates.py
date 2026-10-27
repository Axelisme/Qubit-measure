"""Template-to-engine composition and per-acquire FTF round semantics."""

from collections.abc import Sequence
from dataclasses import replace
from threading import Event

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from zcu_tools.core.simulation import (
    CancellationQuery,
    RoundSignals,
    RoundStatistics,
)
from zcu_tools.device import DeviceManager
from zcu_tools.simulate.evolution import (
    Engine,
    EvolutionResult,
    ModelSource,
    TimelineBatch,
)

from tests.experiment._ftf_templates import (
    expect_statistics,
    request,
    settings,
    snapshot,
    template,
)

pytestmark = pytest.mark.sim


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
    pulse = plan.readouts[0].batch.pulses[0][0]
    assert pulse.envelope is None
    assert pulse.amplitude == pytest.approx(0.2j)
    assert pulse.phase_rad == pytest.approx(np.pi / 6 - 12 * np.pi)
    np.testing.assert_array_equal(
        plan.readouts[0].batch.grids[0][0].times_ns, [8, 9, 10, 11]
    )
    np.testing.assert_array_equal(
        plan.readouts[0].batch.model.receiver.noise_covariance, np.zeros((2, 2))
    )


@pytest.mark.parametrize("tones,path", [(0, "t4"), (3, "dop853")])
def test_template_engine_paths_match_strict_secular_reference(
    tones: int, path: str
) -> None:
    import zcu_tools.experiment.ftf as ftf

    base = template(tones=tones)
    # Keep the 2 ns drive support required by FTF's quantization, but shorten
    # the idle before readout. Unit conversion has its own unscaled test above.
    tiny = replace(
        base,
        events=(
            *base.events[:-1],
            replace(base.events[-1], start_us=base.events[-1].start_us - 0.004),
        ),
        readouts=tuple(
            replace(r, trigger_us=r.trigger_us - 0.004) for r in base.readouts
        ),
    )
    plan = ftf.plan_template(settings(), snapshot(), tiny)
    engine = Engine(1)
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


def test_readout_ignores_controls_after_start_and_rejects_multiple_terminal_pulses() -> (
    None
):
    import zcu_tools.experiment.ftf as ftf

    base = template()
    # Charge controls beginning after readout must have no effect.
    extra = replace(
        base.events[0], gen_ch=2, start_us=np.full(2, 0.015), gain=np.full(2, 0.5)
    )
    overlap = replace(base, events=(*base.events, extra))
    plans = [ftf.plan_template(settings(), snapshot(), t) for t in (base, overlap)]
    engine = Engine(1)
    try:
        results = [engine.evolve(p.source, (p.batch,), cancel=Event()) for p in plans]
    finally:
        engine.close()
    np.testing.assert_array_equal(results[0].densities, results[1].densities)
    np.testing.assert_array_equal(
        plans[0].readouts[0].batch.model.frequencies_ghz,
        plans[1].readouts[0].batch.model.frequencies_ghz,
    )
    with pytest.raises(ValueError, match="one.*readout"):
        ftf.plan_template(
            settings(), snapshot(), replace(base, events=(*base.events, base.events[0]))
        )


def test_cavity_drives_are_classified_per_point_with_calibrated_envelopes() -> None:
    import zcu_tools.experiment.ftf as ftf

    base = template()
    terminal = base.events[-1]
    earlier = replace(
        terminal,
        start_us=np.array([0.0, 0.002]),
        duration_us=np.full(2, 0.004),
        frequency_mhz=np.full(2, 5900.0),
        gain=np.array([0.3, -0.4]),
    )
    # Mutually exclusive slots must classify by active point, not slot order.
    alternate = replace(earlier, active=np.array([False, True]))
    earlier = replace(earlier, active=np.array([True, False]))
    try:
        plan = ftf.plan_template(
            settings(), snapshot(), replace(base, events=(terminal, earlier, alternate))
        )
    except ValueError as error:
        pytest.fail(f"nonterminal cavity drives must be accepted: {error}")
    drives = getattr(plan, "cavity_drives", None)
    assert drives is not None, "plan must retain nonterminal readout drives"
    assert tuple(map(len, drives)) == (1, 1)
    first, second = drives[0][0], drives[1][0]
    assert (first.start_ns, first.end_ns, first.frequency_ghz) == (0, 4, 5.9)
    assert (second.start_ns, second.end_ns) == (2, 6)
    assert first.amplitude == pytest.approx(0.6j)
    assert second.amplitude == pytest.approx(-0.8j)
    assert first.envelope is None
    assert second.phase_rad == pytest.approx(np.pi / 6 - 2 * np.pi * 5.9 * 3)
    np.testing.assert_array_equal(plan.batch.end_ns, [8, 10])
    assert all(len(content) == 1 for content in plan.readouts[0].batch.pulses)


@pytest.mark.parametrize("trigger", [0.007999999999, 0.0])
def test_terminal_readout_rejects_first_trigger_before_latest_block(trigger) -> None:
    import zcu_tools.experiment.ftf as ftf

    base = template()
    window = replace(base.readouts[0], trigger_us=np.full((2, 1), trigger))
    with pytest.raises(ValueError, match="trigger"):
        ftf.plan_template(settings(), snapshot(), replace(base, readouts=(window,)))


def test_cavity_drive_must_not_cross_terminal_start_but_may_end_there() -> None:
    import zcu_tools.experiment.ftf as ftf

    base = template()
    terminal = base.events[-1]
    earlier = replace(terminal, start_us=np.zeros(2), duration_us=terminal.start_us)
    try:
        plan = ftf.plan_template(
            settings(), snapshot(), replace(base, events=(earlier, terminal))
        )
    except ValueError as error:
        pytest.fail(f"cavity drives may end exactly at terminal start: {error}")
    drives = getattr(plan, "cavity_drives", None)
    assert drives is not None, "plan must retain earlier readout drives"
    assert tuple(len(p) for p in drives) == (1, 1)
    crossing = replace(earlier, duration_us=earlier.duration_us + 1e-12)
    with pytest.raises(ValueError, match="cross|overlap"):
        ftf.plan_template(
            settings(), snapshot(), replace(base, events=(crossing, terminal))
        )


class CountingEngine(Engine):
    def __init__(self):
        super().__init__(1)
        self.calls = 0

    def evolve(
        self,
        source: ModelSource,
        batches: Sequence[TimelineBatch],
        *,
        budget: float = 1e-3,
        cancel: CancellationQuery | None = None,
        telemetry: bool = False,
        reuse_key: bytes | None = None,
    ) -> EvolutionResult:
        self.calls += 1
        return super().evolve(
            source,
            batches,
            budget=budget,
            cancel=cancel,
            telemetry=telemetry,
            reuse_key=reuse_key,
        )


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


class ResidualEngine(Engine):
    """Configured executor supplying controlled densities at the C3 boundary."""

    def __init__(self, populations):
        super().__init__(1)
        self.populations = populations

    def evolve(
        self,
        source: ModelSource,
        batches: Sequence[TimelineBatch],
        *,
        budget: float = 1e-3,
        cancel: CancellationQuery | None = None,
        telemetry: bool = False,
        reuse_key: bytes | None = None,
    ) -> EvolutionResult:
        from zcu_tools.simulate.evolution import PathDiagnostics

        points = np.concatenate([batch.points for batch in batches])
        densities = np.repeat(np.diag(self.populations)[None], len(points), axis=0)
        return EvolutionResult(points, densities, PathDiagnostics())


@pytest.mark.parametrize(
    "populations, normalized",
    [
        ([0.25 * (1 + 9e-7), 0.75 * (1 + 9e-7)], [0.25, 0.75]),
        ([-1e-6, 1.000001], [0, 1]),
        ([0, 0.999999], [0, 1]),
    ],
)
def test_acquire_explicitly_normalizes_in_budget_evolution(populations, normalized):
    from zcu_tools.experiment.ftf import FTFSource

    source = FTFSource(settings(), DeviceManager(), engine=ResidualEngine(populations))
    acquire = replace(request(), reps=1_000_000_000)
    try:
        try:
            result = expect_statistics(
                source.open_acquire(acquire).sample_round(snapshot(), 0)
            )
        except ValueError as error:
            pytest.fail(f"in-budget evolution must complete acquire: {error}")
        identity = np.random.SeedSequence((acquire.root_seed, acquire.sequence, 0, 0))
        rng = np.random.Generator(np.random.PCG64(identity))
        expected = rng.multinomial(acquire.reps, np.tile(normalized, (2, 1)))
        np.testing.assert_array_equal(result.state_counts, expected)
    finally:
        source.close()


@pytest.mark.parametrize("populations", [[0, 1.000002], [-2e-6, 1.000002]])
def test_acquire_rejects_out_of_budget_evolution_and_retains_diagnostics(populations):
    from zcu_tools.experiment.ftf import FTFSource

    source = FTFSource(settings(), DeviceManager(), engine=ResidualEngine(populations))
    try:
        with pytest.raises(ValueError, match="2 rows") as rejection:
            source.open_acquire(request()).sample_round(snapshot(), 0)
        assert len(source.evaluations) == 1
        assert source.evaluations[0].round_index == 0
        assert getattr(source.evaluations[0], "rejected_rows", None) == 2
        assert getattr(rejection.value, "rejected_rows", None) == 2
    finally:
        source.close()


def test_template_wires_receiver_delay_and_mismatch_without_delaying_evolution() -> (
    None
):
    from zcu_tools.experiment.ftf import plan_template

    base = settings()
    pair = base.readout.resonators
    configured = base.model_copy(
        update={
            "readout": base.readout.model_copy(
                update={
                    "adc": base.readout.receivers[0].model_copy(
                        update={"time_of_flight_ns": 400.0}
                    ),
                    "resonators": pair.model_copy(
                        update={
                            "r1": pair.r1.model_copy(
                                update={"mismatch_phase_rad": 0.6}
                            ),
                            "r2": pair.r2.model_copy(
                                update={"mismatch_phase_rad": -0.4}
                            ),
                        }
                    ),
                }
            ),
        }
    )
    plan = plan_template(configured, snapshot(), template())
    assert (
        getattr(plan.readouts[0].batch.model.receiver, "time_of_flight_ns", None)
        == 400.0
    )
    np.testing.assert_array_equal(
        getattr(plan.readouts[0].batch.model, "mismatch_phase_rad", None), [0.6, -0.4]
    )
    np.testing.assert_array_equal(plan.batch.end_ns, [8.0, 10.0])
    assert plan.readouts[0].batch.pulses[0][0].start_ns == 8.0
    np.testing.assert_array_equal(
        plan.readouts[0].batch.grids[0][0].times_ns, [8, 9, 10, 11]
    )


def test_sampled_readout_template_preserves_linear_envelope_without_field_ode(
    monkeypatch,
):
    from zcu_tools.core.simulation import PulseEnvelope
    from zcu_tools.experiment.ftf import plan_template
    from zcu_tools.simulate import adc
    from zcu_tools.simulate.adc import response

    base = template()
    samples = PulseEnvelope(
        (0.0, 0.003, 0.011, 0.018), (0.1, 0.8 + 0.2j, 1.0 - 0.1j, 0.2)
    )
    shaped = replace(base, events=(replace(base.events[0], envelopes=(samples,)),))
    batch = plan_template(settings(), snapshot(), shaped).readouts[0].batch
    expected = np.stack(
        [
            adc.ConditionalReadout(batch.model, rtol=1e-12, atol=1e-14)
            .response(ps, gs[0])
            .traces.sum(axis=-2)
            for ps, gs in zip(batch.pulses, batch.grids, strict=True)
        ]
    )
    calls = []
    solve = response.solve_ivp

    def observe(*args, **kwargs):
        calls.append(1)
        return solve(*args, **kwargs)

    monkeypatch.setattr(response, "solve_ivp", observe)
    actual = adc.ConditionalReadout(batch.model).responses(batch)
    assert not calls, "FTF sampled readout must expose its numeric envelope to ADC"
    np.testing.assert_allclose(actual.sums[:, 0], expected, rtol=1e-6, atol=1e-9)
