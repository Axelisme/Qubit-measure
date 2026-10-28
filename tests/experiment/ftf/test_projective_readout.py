"""FTF outcome histories connect herald reads to continued terminal populations."""

import json
from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import expm
from zcu_tools.cfg_model import MuxReadoutTone
from zcu_tools.core.simulation import RoundSignals, SamplingIdentity
from zcu_tools.device import DeviceManager
from zcu_tools.experiment.ftf import FTFSource, ftf, plan_template
from zcu_tools.program.v2 import ModularProgramV2
from zcu_tools.program.v2.modules.control import Repeat
from zcu_tools.program.v2.modules.delay import Delay
from zcu_tools.program.v2.modules.readout import MuxReadoutCfg
from zcu_tools.program.v2.simulation import IntegratedRound
from zcu_tools.resources.ftf.schema import MuxReadoutSource
from zcu_tools.simulate import adc
from zcu_tools.simulate.adc import ADC
from zcu_tools.simulate.evolution import Engine

from tests.experiment._ftf_templates import (
    expect_statistics,
    herald_template,
    request,
    settings,
    snapshot,
    template,
)

pytestmark = pytest.mark.sim


class RejectEvolution(Engine):
    """Any solve means request validation missed its pre-evolution boundary."""

    def evolve(self, source, batches, **kwargs):
        raise AssertionError("inconsistent request topology reached evolution")


@pytest.mark.parametrize("difference", ["projective_count", "adc_layout"])
def test_request_rejects_equal_total_reads_with_different_template_event_layout(
    difference,
):
    first, readout = herald_template()
    both = first.readouts[0]
    terminal = replace(
        both,
        trigger_us=np.column_stack(
            (both.trigger_us[:, 1], both.trigger_us[:, 1] + 0.004)
        ),
    )
    other_adc = first.readouts[1]
    events = first.events[-2:]
    if difference == "adc_layout":
        events = first.events
        other_adc = replace(other_adc, trigger_us=both.trigger_us[:, :1])
    second = replace(
        first, points=first.points + 2, events=events, readouts=(terminal, other_adc)
    )
    acquire = replace(
        request(templates=(first, second)), point_dims=(4,), loop_dims=(3, 4)
    )
    source = FTFSource(
        settings().with_telemetry(enabled=False),
        DeviceManager(),
        readout=readout,
        engine=RejectEvolution(1),
    )
    try:
        with pytest.raises(ValueError, match=r"templates 0.*1.*layout"):
            source.open_acquire(acquire).sample_round(snapshot(), 0)
    finally:
        source.close()


def thermal_mux_settings():
    numeric = settings().with_telemetry(enabled=False)
    chip = numeric.chip.model_copy(
        update={
            "temperature_k": 1.0,
            "f1": numeric.chip.f1.model_copy(
                update={"q_cap": None, "q_ind": None, "aphi": 0.0, "xqp": 0.0}
            ),
            "f2": numeric.chip.f2.model_copy(
                update={"q_cap": None, "q_ind": None, "aphi": 0.0, "xqp": 0.0}
            ),
            "coupler": numeric.chip.coupler.model_copy(
                update={"q_cap": None, "aphi": 0.0}
            ),
        }
    )
    receiver = numeric.readout.adc.model_copy(update={"gain": 1e12})
    return numeric.model_copy(
        update={
            "chip": chip,
            "numerics": numeric.numerics.model_copy(
                update={
                    "truncation": numeric.numerics.truncation.model_copy(
                        update={"retained_levels": 4}
                    )
                }
            ),
            "readout": numeric.readout.model_copy(
                update={
                    "mux": MuxReadoutSource(
                        gen_ch=1,
                        input_per_gain=1.0,
                        adc=tuple(
                            receiver.model_copy(update={"ro_ch": ch}) for ch in (4, 1)
                        ),
                    )
                }
            ),
        }
    )


def herald_program(env, *, reads=2, reps=256):
    mux = MuxReadoutCfg(
        gen_ch=1,
        nqz=1,
        mixer_freq=6000.0,
        length=0.020,
        ro_length=0.026,
        tones=[
            MuxReadoutTone(ro_ch=ch, freq=freq, gain=0.01)
            for ch, freq in ((4, 6000.0), (1, 6200.0))
        ],
    )
    prog = ModularProgramV2(
        env.soccfg,
        modules=[
            Repeat("reads", reads).add_content(
                [mux.build("joint"), Delay("wait", 0.020)]
            )
        ],
        reps=reps,
    )
    # QICK counts the trigger macro once; Repeat executes it once per event.
    prog.set_reads_per_shot(reads)
    return prog


@pytest.mark.parametrize("mode", ["integrated", "decimated"])
def test_mux_herald_postselection_retains_joint_ground_and_reports_outcomes(
    tmp_path, mode
):
    numeric = thermal_mux_settings().with_telemetry(enabled=True, directory=tmp_path)
    env = ftf(numeric, time_scale=0)
    try:
        prog = herald_program(env)
        if mode == "integrated":
            prog.acquire(env.soc, progress=False)
        else:
            prog.acquire_decimated(env.soc, progress=False)
        raw = prog.get_raw()
        assert raw is not None
        captured = env.soc.binding.rounds[0]
        plan = plan_template(numeric, captured.snapshot, captured.acquire.templates[0])
        labels = []
        for read in range(2):
            observed = np.concatenate([shots[:, read] for shots in raw], axis=-1)
            centres = np.concatenate(
                [
                    ADC(channel.batch.model).responses(channel.batch).sums[0, read]
                    for channel in plan.readouts
                ],
                axis=-1,
            )
            distances = np.linalg.norm(observed[:, None] - centres[None], axis=-1)
            labels.append(np.argmin(distances, axis=-1))
        ground = labels[0] == 0
        assert 0.1 < ground.mean() < 0.5
        assert np.mean(labels[1][ground] == 0) > 0.97
        assert np.mean(labels[1] == labels[0]) > 0.97
        record = json.loads(
            next(tmp_path.glob("routing-*.jsonl")).read_text().splitlines()[-1]
        )
        assert record["projection_outcomes"] == [4]
        assert record["projection_pruned_weight"] == [0.0]
    finally:
        env.devices.close_all_devices()


def test_herald_expands_joint_count_axis_and_preserves_terminal_only_adc():
    template, readout = herald_template()
    numeric = settings().with_telemetry(enabled=False)
    numeric = numeric.model_copy(
        update={"chip": numeric.chip.model_copy(update={"temperature_k": 1.0})}
    )
    source = FTFSource(numeric, DeviceManager(), readout=readout)
    acquire = replace(request(templates=(template,)), reps=20000)
    try:
        try:
            result = source.open_acquire(acquire).sample_round(snapshot(), 0)
        except ValueError as error:
            pytest.fail(
                f"projective FTF round must prepare aligned outcome histories: {error}"
            )
        statistics = expect_statistics(result)
        assert statistics.state_counts.shape == (2, 4)
        np.testing.assert_array_equal(
            statistics.state_counts.sum(axis=-1), [20000, 20000]
        )
        facts = source.model_facts(snapshot())
        dc = facts.source.model((0, 0, 0))
        rates = dc.rates_per_ns
        generator = rates - np.diag(rates.sum(axis=0))
        expected_joint = (
            facts.source.initial_density.diagonal().real[:, None]
            * expm(generator * 8.0).T
        )
        np.testing.assert_allclose(
            statistics.state_counts.reshape(2, 2, 2) / acquire.reps,
            np.broadcast_to(expected_joint, (2, 2, 2)),
            atol=0.01,
        )
        assert statistics.channels[0].sums.shape[-2] == 2
        assert statistics.channels[1].sums.shape[-2] == 1
        assert source.evaluations[0].projection_outcomes == (2, 2)
    finally:
        source.close()


@pytest.mark.parametrize("floor", [0.0, 0.9])
def test_projection_floor_keeps_zero_weight_histories_or_names_empty_point(floor):
    template, readout = herald_template()
    numeric = settings().with_telemetry(enabled=False)
    numeric = numeric.model_copy(
        update={
            "numerics": numeric.numerics.model_copy(
                update={"projection_weight_floor": floor}
            ),
            "chip": numeric.chip.model_copy(
                update={"temperature_k": 1.0 if floor else 0.0}
            ),
        }
    )
    source = FTFSource(numeric, DeviceManager(), readout=readout)
    try:
        session = source.open_acquire(request(templates=(template,)))
        if floor:
            with pytest.raises(ValueError, match=r"point 0.*floor.*0.9"):
                session.sample_round(snapshot(), 0)
        else:
            statistics = expect_statistics(session.sample_round(snapshot(), 0))
            assert statistics.state_counts.shape == (2, 4)
            assert source.evaluations[0].projection_outcomes == (2, 2)
            assert source.evaluations[0].projection_pruned_weight == (0.0, 0.0)
    finally:
        source.close()


@pytest.mark.parametrize("mode", ["integrated", "decimated"])
def test_unobserved_terminal_still_evolves_and_keeps_joint_final_state(mode):
    base, readout = herald_template()
    first = base.readouts[0]
    first = replace(
        first,
        trigger_us=first.trigger_us[:, :1],
        frequency_mhz=first.frequency_mhz[:, :1],
        phase_deg=first.phase_deg[:, :1],
        generator_channels=first.generator_channels[:, :1],
    )
    base = replace(base, readouts=(first,))
    numeric = settings().with_telemetry(enabled=False)
    numeric = numeric.model_copy(
        update={"chip": numeric.chip.model_copy(update={"temperature_k": 1.0})}
    )
    source = FTFSource(numeric, DeviceManager(), readout=readout)
    try:
        result = source.open_acquire(request(mode, templates=(base,))).sample_round(
            snapshot(), 0
        )
        if mode == "integrated":
            statistics = expect_statistics(result)
            assert statistics.state_counts.shape == (2, 4)
            assert statistics.channels[0].sums.shape == (2, 1, 2)
        else:
            if not isinstance(result, RoundSignals):
                raise AssertionError("decimated round must return traces")
            assert result.channels[0].iq.shape == (6, 1, 4, 2)
        assert source.evaluations[0].projection_outcomes == (2, 2)
    finally:
        source.close()


@pytest.mark.parametrize("mode", ["integrated", "decimated"])
def test_terminal_only_runtime_matches_direct_engine_and_adc_elementwise(mode):
    numeric = settings().with_telemetry(enabled=False)
    source = FTFSource(numeric, DeviceManager())
    engine = Engine(1)
    try:
        acquire = request(mode)
        facts = source.model_facts(snapshot())
        plan = plan_template(numeric, snapshot(), template(), facts=facts)
        evolved = engine.evolve(facts.source, (plan.batch,))
        populations = adc.normalize_populations(
            evolved.densities.diagonal(axis1=-2, axis2=-1).real
        )
        readout = adc.ProjectionReadout.unprojected(populations)
        batch = plan.readouts[0].batch
        result = source.open_acquire(acquire).sample_round(snapshot(), 0)
        if mode == "integrated":
            if not isinstance(result, IntegratedRound):
                raise AssertionError("integrated round must return statistics and raw")
            conditional = adc.ADC(
                batch.model,
                rtol=numeric.numerics.solver.rtol,
                atol=numeric.numerics.solver.atol,
            ).responses(batch)
            expected = adc.round_statistics(
                readout,
                (
                    adc.ChannelReadout(
                        0, conditional, batch.content_index, batch.model.receiver
                    ),
                ),
                reps=acquire.reps,
                seed=acquire.root_seed,
                acquire_sequence=acquire.sequence,
                round_index=0,
            )
            np.testing.assert_array_equal(
                result.statistics.state_counts, expected.state_counts
            )
            np.testing.assert_array_equal(
                result.statistics.channels[0].sums, expected.channels[0].sums
            )
            raw = adc.PreparedRawShots(
                readout,
                {0: batch},
                reps=acquire.reps,
                rtol=numeric.numerics.solver.rtol,
                atol=numeric.numerics.solver.atol,
            ).bind(expected.raw_identity)
            np.testing.assert_array_equal(
                result.raw.materialize(acquire.reps, 0)[0],
                raw.materialize(acquire.reps, 0)[0],
            )
        else:
            if not isinstance(result, RoundSignals):
                raise AssertionError("decimated round must return traces")
            expected = adc.sample_traces(
                readout,
                {0: batch},
                reps=acquire.reps,
                identity=SamplingIdentity(acquire.root_seed, (acquire.sequence, 0, 0)),
                rtol=numeric.numerics.solver.rtol,
                atol=numeric.numerics.solver.atol,
            )
            ordered = np.moveaxis(
                expected.channels[0].iq.reshape(2, 3, 1, 4, 2), 1, 0
            ).reshape(6, 1, 4, 2)
            np.testing.assert_array_equal(result.channels[0].iq, ordered)
    finally:
        engine.close()
        source.close()


def test_configured_floor_prunes_and_redistributes_thermal_outcome_weight():
    template, readout = herald_template()
    numeric = settings().with_telemetry(enabled=False)
    numeric = numeric.model_copy(
        update={
            "chip": numeric.chip.model_copy(update={"temperature_k": 1.0}),
            "numerics": numeric.numerics.model_copy(
                update={"projection_weight_floor": 0.5}
            ),
        }
    )
    source = FTFSource(numeric, DeviceManager(), readout=readout)
    try:
        initial = source.model_facts(snapshot()).source.initial_density.diagonal().real
        result = source.open_acquire(
            replace(request(templates=(template,)), reps=100)
        ).sample_round(snapshot(), 0)
        statistics = expect_statistics(result)
        assert statistics.state_counts.shape == (2, 2)
        assert source.evaluations[0].projection_outcomes == (1, 1)
        np.testing.assert_allclose(
            source.evaluations[0].projection_pruned_weight, initial[1], atol=1e-10
        )
        np.testing.assert_array_equal(statistics.state_counts.sum(axis=-1), [100, 100])
    finally:
        source.close()


def test_two_projections_preserve_both_herald_states_in_conditional_statistics():
    base, readout = herald_template()
    assert readout.mux is not None
    readout = replace(
        readout,
        mux=replace(
            readout.mux,
            receivers=tuple(
                replace(receiver, gain=1e8) for receiver in readout.mux.receivers
            ),
        ),
    )
    last = tuple(
        replace(event, start_us=event.start_us + 0.012) for event in base.events[-2:]
    )
    windows = tuple(
        replace(
            window,
            trigger_us=np.column_stack(
                (window.trigger_us, window.trigger_us[:, -1] + 0.012)
            ),
            frequency_mhz=np.column_stack(
                (window.frequency_mhz, window.frequency_mhz[:, -1])
            ),
            phase_deg=np.column_stack((window.phase_deg, window.phase_deg[:, -1])),
            generator_channels=np.column_stack(
                (window.generator_channels, window.generator_channels[:, -1])
            ),
        )
        for window in base.readouts
    )
    middle = tuple(
        replace(event, duration_us=np.full(2, 0.002)) for event in base.events[-2:]
    )
    template = replace(
        base,
        events=(*base.events[:2], *middle, *last),
        readouts=windows,
        duration_us=base.duration_us + 0.012,
    )
    numeric = settings().with_telemetry(enabled=False)
    chip = numeric.chip.model_copy(
        update={
            "temperature_k": 1.0,
            "f1": numeric.chip.f1.model_copy(update={"q_cap": 10.0}),
            "f2": numeric.chip.f2.model_copy(update={"q_cap": 10.0}),
        }
    )
    source = FTFSource(
        numeric.model_copy(update={"chip": chip}), DeviceManager(), readout=readout
    )
    try:
        result = source.open_acquire(request(templates=(template,))).sample_round(
            snapshot(), 0
        )
        statistics = expect_statistics(result)
        assert source.evaluations[0].projection_outcomes == (4, 4)
        sums = statistics.channels[0].conditional_sums.reshape(2, 3, 4, 2, 2)
        # Histories are 00,01,10,11; the final state is a separate joint axis.
        np.testing.assert_array_equal(sums[:, 0, 0], sums[:, 0, 1])
        np.testing.assert_array_equal(sums[:, 0, 2], sums[:, 0, 3])
        np.testing.assert_array_equal(sums[:, 1, 0], sums[:, 1, 2])
        np.testing.assert_array_equal(sums[:, 1, 1], sums[:, 1, 3])
        assert np.max(np.abs(sums[:, 0, 0] - sums[:, 0, 2])) > 1e-5
        assert np.max(np.abs(sums[:, 1, 0] - sums[:, 1, 1])) > 1e-5
    finally:
        source.close()
