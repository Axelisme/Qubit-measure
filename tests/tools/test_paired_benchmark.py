"""Public paired timing contracts, using synthetic samples without evolution."""

import os
import sys
from contextlib import ExitStack, closing, contextmanager
from io import StringIO
from pathlib import Path

import numpy as np
import pytest
from zcu_tools.simulate import evolution
from zcu_tools.simulate.evolution import (
    EvolutionResult,
    FamilyRouting,
    FrozenRecord,
    PathDiagnostics,
    ReferenceResult,
    physical_input,
    write_frozen_index,
    write_reference,
    write_scene_record,
)

from tests.tools.family_pareto_support import make_scene
from tools import family_pareto
from tools import paired_benchmark as paired
from tools.simulator_measurement import Sample, SampleRunner, Session


def test_block_sum_ratios_median_and_linear_iqr() -> None:
    blocks = [
        [Sample(wall_seconds=w, process_seconds=w * 2, observation=None) for w in times]
        for times in [(1, 3, 3, 9), (2, 2, 2, 2), (1, 2, 2, 1), (1, 4, 4, 1)]
    ]
    summary = paired.summarize_blocks(blocks)
    for clock in (summary.wall, summary.process):
        assert clock.ratios == [0.6, 1.0, 2.0, 4.0]
        assert clock.median == 1.5
        assert clock.iqr == (0.9, 2.5)


@pytest.mark.parametrize("clock", ["wall", "process"])
@pytest.mark.parametrize("zero_side", ["baseline", "candidate"])
def test_zero_side_sums_fail_with_block_clock_and_both_totals(clock, zero_side):
    samples = []
    for side in ("baseline", "candidate", "candidate", "baseline"):
        duration = 0.0 if side == zero_side else 1.0
        samples.append(
            Sample(
                wall_seconds=duration if clock == "wall" else 1.0,
                process_seconds=duration if clock == "process" else 1.0,
                observation=None,
            )
        )
    baseline, candidate = (0.0, 2.0) if zero_side == "baseline" else (2.0, 0.0)
    with pytest.raises(
        ValueError,
        match=rf"block 1: {clock}: baseline={baseline}, candidate={candidate}",
    ):
        paired.summarize_blocks([samples])


def test_abba_waits_for_each_sample_and_preserves_raw_readings():
    events = []

    def sample(side):
        def request(command):
            events.append((side, command.telemetry, "start"))
            result = Sample(wall_seconds=0, process_seconds=2, observation=side)
            events.append((side, command.telemetry, "finish"))
            return result

        return request

    blocks = paired.collect_abba(
        sample("A"), sample("B"), blocks=2, telemetry=(False, True)
    )
    assert events == [
        (side, side == "B", action)
        for side in "ABBAABBA"
        for action in ("start", "finish")
    ]
    assert [[s.observation for s in block] for block in blocks] == [
        ["A", "B", "B", "A"]
    ] * 2
    assert all(
        s.wall_seconds == 0 and s.process_seconds == 2
        for block in blocks
        for s in block
    )


@pytest.mark.parametrize(
    "outside", [None, "A-wall", "A-process", "B-wall", "B-process"]
)
def test_telemetry_requires_all_four_closed_intervals(outside):
    def summary(side):
        def clock(name):
            value = 1.06 if outside == f"{side}-{name}" else 1.03
            return paired.RatioSummary([value] * 3, value, (value, value))

        noise = paired.RatioSummary([0.98, 1.0, 1.03], 1.0, (0.98, 1.03))
        return paired.Calibration(
            on_off=paired.BlockSummary(clock("wall"), clock("process")),
            off_off=paired.BlockSummary(noise, noise),
        )

    assert paired.choose_telemetry((summary("A"), summary("B"))) is (outside is None)


def test_small_calibration_is_not_a_mode_decision():
    ratio = paired.RatioSummary([1.0, 1.0], 1.0, (1.0, 1.0))
    summary = paired.BlockSummary(ratio, ratio)
    calibration = paired.Calibration(on_off=summary, off_off=summary)
    with pytest.raises(ValueError, match="more than two"):
        paired.choose_telemetry((calibration, calibration))


def _ready(**overrides):
    values = {
        "protocol_version": 1,
        "physical_version": 1,
        "cache_version": 1,
        "records": [
            dict(
                name="scene",
                physical_sha256="p",
                oracle_sha256="o",
                metadata_sha256="m",
            )
        ],
        "segment_numeric_version": "numeric-a",
        "provenance": {"head": "commit", "tree": "tree", "dirty": False},
    }
    values.update(overrides)
    return paired.Ready.model_validate(values)


@pytest.mark.parametrize(
    "field", ["protocol_version", "physical_version", "cache_version", "records"]
)
def test_handshake_rejects_each_contract_mismatch_but_not_numeric_revision(field):
    expected = _ready()
    changed = [] if field == "records" else 2
    with pytest.raises(ValueError, match=field):
        paired.validate_ready(expected, _ready(**{field: changed}))
    paired.validate_ready(
        expected, _ready(segment_numeric_version="optimized-numeric-b")
    )


def test_child_stream_handshake_sample_reset_and_cleanup():
    events = []

    @contextmanager
    def factory():
        events.append("open")
        try:
            yield Session(
                lambda telemetry: telemetry,
                lambda result, telemetry: result,
                lambda: events.append("clear"),
            )
        finally:
            events.append("close")

    commands = StringIO(
        '{"telemetry":true}\n'
        '{"action":"reset","telemetry":false}\n'
        '{"telemetry":false,"empty_history":true}\n'
        '{"action":"close","telemetry":false}\n'
    )
    output = StringIO()
    with SampleRunner(factory, mode="warm_retain_history", warmups=0) as runner:
        paired.serve_child(_ready(), runner, commands, output)
    replies = output.getvalue().splitlines()
    assert paired.Ready.model_validate_json(replies[0]) == _ready()
    assert Sample[bool].model_validate_json(replies[1]).observation is True
    assert replies[2] == '{"reset":true}'
    assert Sample[bool].model_validate_json(replies[3]).observation is False
    assert events == ["open", "close", "open", "clear", "close"]


def _config(tmp_path, mode):
    return paired.SideConfig(
        worktree=tmp_path,
        scenario="scene",
        reference_cache=tmp_path / "references",
        mode=mode,
    )


@pytest.mark.parametrize("mode", ["warm_clear_history", "first", "cold"])
def test_controller_owns_lifetimes_prime_and_cold_cache_cleanup(tmp_path, mode):
    launches = []
    requests = []
    closed = []
    cold_parents = []

    @contextmanager
    def opener(launch):
        launches.append(launch.purpose)
        if launch.cache_parent is not None:
            parent = Path(launch.cache_parent)
            assert list(parent.iterdir()) == []
            cold_parents.append(parent)

        def sample(command):
            requests.append((launch.purpose, command.telemetry))
            return Sample(wall_seconds=1, process_seconds=2, observation="density")

        try:
            yield paired.ChildHandle(_ready(), sample, lambda: None)
        finally:
            closed.append(launch.purpose)

    with paired.SideClient(_config(tmp_path, mode), opener) as side:
        side.prime()
        assert side.sample(paired.SampleRequest(False)).observation == "density"
        assert side.sample(paired.SampleRequest(True)).observation == "density"
    if mode == "warm_clear_history":
        assert launches == closed == ["measure"]
        assert requests == [("measure", False), ("measure", True)]
    elif mode == "first":
        assert launches == closed == ["probe", "prime", "measure", "measure"]
        assert requests == [
            ("prime", False),
            ("prime", True),
            ("measure", False),
            ("measure", True),
        ]
    else:
        assert launches == closed == ["probe", "measure", "measure"]
        assert requests == [("measure", False), ("measure", True)]
        assert len(cold_parents) == 2
        assert cold_parents[0] != cold_parents[1]
        assert all(not parent.exists() for parent in cold_parents)


def test_new_child_identity_failure_closes_without_sample_or_retry(tmp_path):
    events = []

    @contextmanager
    def opener(launch):
        events.append(launch.purpose)
        ready = _ready() if launch.purpose == "probe" else _ready(records=[])

        def unexpected_sample(command):
            raise AssertionError("mismatched child must not be sampled")

        try:
            yield paired.ChildHandle(ready, unexpected_sample, lambda: None)
        finally:
            events.append("close")

    with (
        paired.SideClient(_config(tmp_path, "first"), opener) as side,
        pytest.raises(ValueError, match="records"),
    ):
        side.sample(paired.SampleRequest(False))
    assert events == ["probe", "close", "measure", "close"]


def test_comparison_calibrates_both_sides_then_runs_shared_mode_and_separate_diagnostics(
    tmp_path,
):
    requests = []
    resets = []

    @contextmanager
    def opener(launch):
        name = launch.config.worktree.name

        def sample(command):
            requests.append((name, command.telemetry, command.empty_history))
            seconds = 2.0 if command.telemetry else 1.0
            return Sample(
                wall_seconds=seconds, process_seconds=seconds, observation=name
            )

        yield paired.ChildHandle(_ready(), sample, lambda: resets.append(name))

    config_a = _config(tmp_path / "A", "warm_clear_history")
    config_b = _config(tmp_path / "B", "warm_clear_history")
    with (
        paired.SideClient(config_a, opener) as a,
        paired.SideClient(config_b, opener) as b,
    ):
        result = paired.run_comparison(a, b, blocks=3)
    assert result.telemetry is False
    assert result.summary.wall.ratios == [1, 1, 1]
    assert result.summary.process.median == 1
    assert len(result.calibrations) == 2
    assert all(len(c.off_off) == len(c.on_off) == 3 for c in result.calibrations)
    assert requests[-2:] == [("A", True, True), ("B", True, True)]
    assert [name for name, _, empty in requests[-14:-2] if not empty] == list(
        "ABBAABBAABBA"
    )
    assert result.diagnostics is not None
    assert [s.observation for s in result.diagnostics] == ["A", "B"]
    assert resets == ["A", "A", "A", "B", "B", "B"]


@pytest.mark.parametrize("backend", ["cpu", "auto"])
def test_prepared_child_requires_reference_before_evolve_and_inspects_complete_density(
    tmp_path, monkeypatch, backend
):
    engine_type = evolution.Engine
    backends = []

    def record_engine(workers, **kwargs):
        backends.append(kwargs.get("backend", "auto"))
        return engine_type(workers, **kwargs)

    # The configured backend is not observable from a synthetic evolve result.
    monkeypatch.setattr(evolution, "Engine", record_engine)
    scene = make_scene("idle_coherence")
    cache = tmp_path / "scenes"
    index = tmp_path / "index.json"
    references = tmp_path / "references"
    with closing(scene.source):
        record = FrozenRecord(
            "scene",
            physical_input(scene.source, scene.batch),
            {"reason": "synthetic", "expected_families": ["t4"]},
        )
        write_frozen_index(index, (write_scene_record(cache, record),))
        reference = np.repeat(
            scene.source.initial_density[None, :, :], len(scene.batch.points), axis=0
        )
        config = paired.SideConfig(
            worktree=tmp_path,
            scenario="scene",
            reference_cache=references,
            scene_cache=cache,
            index=index,
            mode="first",
            backend=backend,
        )
        calls = []
        unavailable = "OpenCL requires a GPU supporting cl_khr_fp64"

        def invoke(engine, case, options, request):
            calls.append(request.telemetry)
            density = reference.copy()
            density[:, 0, 1] += 0.125j
            routing = (
                {"dop853": FamilyRouting(rows=2, opencl_detection_error=unavailable)}
                if options.backend == "auto" and request.telemetry
                else {}
            )
            return EvolutionResult(
                case.batch.points, density, PathDiagnostics(routing=routing)
            )

        with (
            ExitStack() as stack,
            pytest.raises(FileNotFoundError, match="reference cache miss"),
        ):
            stack.enter_context(paired.prepared_child(config, invoke=invoke))
        assert calls == []
        write_reference(
            references, scene.source, scene.batch, ReferenceResult(reference, 0.0)
        )
    with paired.prepared_child(config, invoke=invoke) as (ready, runner):
        sample = runner.sample(telemetry=backend == "auto")
    selected = family_pareto.select_inputs(index, "scene", scene_cache=cache)
    assert ready.records == [selected[0].identity]
    assert backends == [backend]
    assert calls == [backend == "auto"]
    assert sample.observation.reference_error == 0.125
    assert sample.observation.points == [0, 1]
    assert sample.observation.reference_key
    assert sample.observation.convergence_error == 0.0
    if backend == "auto":
        assert (
            sample.observation.routing["dop853"]["opencl_detection_error"]
            == unavailable
        )
    else:
        assert sample.observation.aggregate is None


def test_main_writes_complete_report_and_preserves_previous_output_on_handshake_failure(
    tmp_path,
):
    output = tmp_path / "report.json"
    argv = [
        "run",
        "--baseline",
        str(tmp_path / "A"),
        "--candidate",
        str(tmp_path / "B"),
        "--baseline-backend",
        "cpu",
        "--candidate-backend",
        "opencl",
        "--scenario",
        "scene",
        "--reference-cache",
        str(tmp_path / "references"),
        "--mode",
        "warm_clear_history",
        "--blocks",
        "3",
        "--output",
        str(output),
        "--exclusive",
    ]

    @contextmanager
    def opener(launch):
        delta = 0.25 if launch.config.backend == "opencl" else 0.0
        observation = paired.Observation(
            points=[0],
            density_real=[[[1, delta], [delta, 0]]],
            density_imag=[[[0, 0], [0, 0]]],
            reference_error=delta,
            reference_key="oracle-key",
            convergence_error=0,
            routing={},
            aggregate=None,
        )
        yield paired.ChildHandle(
            _ready(),
            lambda command: Sample(
                wall_seconds=1, process_seconds=2, observation=observation
            ),
            lambda: None,
        )

    assert paired.main(argv, opener=opener) == 0
    original = output.read_bytes()
    report = paired.BenchmarkReport.model_validate_json(original)
    assert report.schema_version == 1
    assert tuple(side.backend for side in report.coordinates) == ("cpu", "opencl")
    assert report.comparison.summary.wall.median == 1
    assert report.comparison.telemetry is True
    assert report.comparison.diagnostics is None
    assert report.density_errors == [0.25, 0.25, 0.25]
    assert report.comparison.blocks[0][1].observation.reference_key == "oracle-key"
    assert report.exclusive is True

    @contextmanager
    def mismatched(launch):
        ready = _ready(records=[]) if launch.config.worktree.name == "B" else _ready()
        with opener(launch) as child:
            yield paired.ChildHandle(ready, child.sample, child.reset)

    assert paired.main(argv, opener=mismatched) == 1
    assert output.read_bytes() == original


@pytest.mark.parametrize("exit_code", [0, 7])
def test_process_bridge_inherits_group_and_lock_and_reaps_failed_children(
    tmp_path, monkeypatch, exit_code
):
    script = tmp_path / "synthetic_child.py"
    ready = _ready().model_dump_json()
    observation = paired.Observation(
        points=[0],
        density_real=[[[1]]],
        density_imag=[[[0]]],
        reference_error=0,
        reference_key="key",
        convergence_error=0,
        routing={},
        aggregate=None,
    )
    sample = Sample(
        wall_seconds=0, process_seconds=0, observation=observation
    ).model_dump_json()
    script.write_text(
        "import json, os, sys\n"
        f"ready = json.loads({ready!r})\n"
        "os.fstat(int(os.environ['PAIRED_TEST_FD']))\n"
        "ready['provenance'].update(group=str(os.getpgrp()), inherited_fd='yes', cache_parent=os.environ['NUMBA_CACHE_DIR'])\n"
        "print(json.dumps(ready), flush=True)\n"
        "for line in sys.stdin:\n"
        "    request = json.loads(line)\n"
        f"    if request['action'] == 'close': sys.exit({exit_code})\n"
        "    if request['action'] == 'reset': print('{\"reset\":true}', flush=True)\n"
        f"    else: print({sample!r}, flush=True)\n"
    )
    popen = paired.subprocess.Popen

    def synthetic_process(command, **kwargs):
        # Replacing the external executable keeps the real pipe/group/FD bridge;
        # benchmark's opener seam would bypass the transport being verified.
        return popen([sys.executable, str(script)], **kwargs)

    monkeypatch.setattr(paired.subprocess, "Popen", synthetic_process)
    cache = tmp_path / "empty-numba"
    cache.mkdir()
    config = _config(tmp_path, "cold")
    with (tmp_path / "outer-lock").open("w") as lock:
        os.set_inheritable(lock.fileno(), True)
        monkeypatch.setenv("PAIRED_TEST_FD", str(lock.fileno()))
        try:
            with ExitStack() as stack:
                if exit_code:
                    stack.enter_context(pytest.raises(RuntimeError, match="status 7"))
                with paired.launch_child(
                    paired.Launch(config, "measure", cache)
                ) as child:
                    assert child.ready.provenance["group"] == str(os.getpgrp())
                    assert child.ready.provenance["inherited_fd"] == "yes"
                    assert child.ready.provenance["cache_parent"] == str(cache)
                    child.reset()
                    assert child.sample(paired.SampleRequest(False)).wall_seconds == 0
        finally:
            os.set_inheritable(lock.fileno(), False)
