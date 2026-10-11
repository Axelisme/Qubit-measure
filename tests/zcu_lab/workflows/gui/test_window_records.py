"""Window records contracts through public native/GUI seams."""

from base64 import b64decode
from dataclasses import asdict
from threading import get_ident

import pytest
from matplotlib.figure import Figure
from zcu_tools.experiment.workflows import (
    Done,
    InitEnv,
    Next,
    Step,
    WorkflowEnv,
    workflow,
)
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.resources.entry import (
    ResultEntry,
)

from tests.gui.app.workflows._start_fakes import StartHarness
from zcu_lab.workflows.calibration import Calibrated
from zcu_lab.workflows.gui import window as window_module
from zcu_lab.workflows.gui.window import T1WorkflowWindow
from zcu_lab.workflows.t1 import T1Plan, T1Tunables

from ._window_support import RpcRequest


@pytest.mark.parametrize(
    "method", ["workflow.records", "workflow.figures", "workflow.figure"]
)
def test_rpc_read_targets_preserve_pending_and_absent(
    run_attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    rpc: RpcRequest,
    method: str,
) -> None:
    window, harness, entry = run_attachment
    params: dict[str, object] = {"run_id": "reader"}
    if method == "workflow.figure":
        params["name"] = "live"
    absent = rpc(method, params)
    assert absent["ok"] is False
    error = absent["error"]
    assert isinstance(error, dict)
    assert (error["code"], error["reason"], error["data"]) == (
        "precondition_failed",
        "run_mismatch",
        {"expected": "reader", "actual": None},
    )
    token = window.start("reader")
    assert isinstance(token, int)
    pending = rpc(method, params)
    assert pending["ok"] is False
    error = pending["error"]
    assert isinstance(error, dict) and error["code"] == "controller_error"
    assert window.controller.operation_id == token
    assert window.controller.status() is None and harness.gate.snapshot()
    assert not (entry.result_path / "workflows/reader").exists()
    window.stop("reader")
    harness.complete_segment()


@pytest.mark.parametrize(
    "method", ["workflow.records", "workflow.figures", "workflow.figure"]
)
@pytest.mark.parametrize("run_id", [None, "", 3])
def test_rpc_read_requires_target(rpc: RpcRequest, method: str, run_id: object) -> None:
    params: dict[str, object] = {"run_id": run_id, "name": "live"}
    with pytest.raises(RemoteError) as error:
        rpc(method, params)
    assert error.value.code is ErrorCode.INVALID_PARAMS


@pytest.mark.parametrize("since", [True, "1", 1.5])
def test_rpc_records_rejects_noninteger_cursor(rpc: RpcRequest, since: object) -> None:
    with pytest.raises(RemoteError) as error:
        rpc("workflow.records", {"run_id": "absent", "since": since})
    assert error.value.code is ErrorCode.INVALID_PARAMS


def test_rpc_records_retains_commits_and_revision_after_resume(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    paused_tuning: list[tuple[int, float, int]],
    rpc: RpcRequest,
) -> None:
    window, harness, _entry = attachment
    engine = window.controller.engine
    assert engine is not None and len(paused_tuning) == 2
    first = engine.records("tuning")[0]
    expected = {
        "commit_seq": first.commit_seq,
        "call_seq": first.call_seq,
        "revision": first.revision,
        "encoded_record": asdict(first.encoded_record),
        "iteration_dir": first.iteration_dir,
        "run_files": list(first.run_files),
    }
    assert rpc("workflow.records", {"run_id": "tuning"})["result"] == {
        "run_id": "tuning",
        "records": [expected],
    }
    assert rpc("workflow.records", {"run_id": "tuning", "since": first.commit_seq})[
        "result"
    ] == {"run_id": "tuning", "records": []}
    negative = rpc("workflow.records", {"run_id": "tuning", "since": -1})
    assert negative["ok"] is False
    error = negative["error"]
    assert isinstance(error, dict) and error["code"] == "invalid_params"
    assert (
        rpc(
            "workflow.tunables",
            {"run_id": "tuning", "batch": [{"path": ["guess_us"], "value": 41.0}]},
        )["ok"]
        is True
    )
    assert rpc("workflow.resume", {"run_id": "tuning"})["ok"] is True
    harness.complete_segment()
    result = rpc("workflow.records", {"run_id": "tuning", "since": 1})["result"]
    records = engine.records("tuning", since=1)
    assert len(records) == 1 and records[0].revision == 1
    expected = {
        "commit_seq": records[0].commit_seq,
        "call_seq": records[0].call_seq,
        "revision": records[0].revision,
        "encoded_record": asdict(records[0].encoded_record),
        "iteration_dir": records[0].iteration_dir,
        "run_files": list(records[0].run_files),
    }
    assert result == {"run_id": "tuning", "records": [expected]}
    assert engine.records("tuning")[0] == first
    assert window.controller.status() is not None and not harness.gate.snapshot()


def test_rpc_records_running_empty_and_committed_native_summary(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    rpc: RpcRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, harness, _entry = attachment
    observations: list[dict[str, object]] = []

    def init(_env: InitEnv[Calibrated], _plan: T1Plan) -> int:
        return 0

    @workflow("read-running", plan=T1Plan, tunables=T1Tunables, init=init)
    def step(
        env: WorkflowEnv[Calibrated], plan: T1Plan, tun: T1Tunables, state: int
    ) -> Step[int, dict[str, float]]:
        del env, plan, tun
        observations.append(rpc("workflow.records", {"run_id": "reader"}))
        yield from ()
        if state == 0:
            return Next({"measurement": 2.5}, 1)
        return Done()

    monkeypatch.setattr(window_module, "t1_fluxdep", step)
    assert isinstance(window.start("reader"), int)
    harness.complete_segment()
    assert harness.background.error is None
    assert observations[0]["result"] == {"run_id": "reader", "records": []}
    engine = window.controller.engine
    assert engine is not None
    native = engine.records("reader")[0]
    expected = {
        "commit_seq": native.commit_seq,
        "call_seq": native.call_seq,
        "revision": native.revision,
        "encoded_record": asdict(native.encoded_record),
        "iteration_dir": native.iteration_dir,
        "run_files": list(native.run_files),
    }
    assert observations[1]["result"] == {"run_id": "reader", "records": [expected]}
    assert native.encoded_record.value == {"measurement": 2.5}


@pytest.mark.parametrize("ending", ["resume", "stop"])
def test_rpc_figures_latest_owner_render_and_lifetime(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    rpc: RpcRequest,
    monkeypatch: pytest.MonkeyPatch,
    ending: str,
) -> None:
    window, harness, _entry = attachment
    renderer = window_module.render_figure_png
    owner_id = get_ident()
    renders: list[int] = []
    figures: list[Figure] = []
    attempts: list[int] = []

    def render(figure: Figure) -> bytes:
        assert figure.get_facecolor() == (0.0, 0.0, 1.0, 1.0)
        renders.append(get_ident())
        figures.append(figure)
        return renderer(figure)

    monkeypatch.setattr(window_module, "render_figure_png", render)

    def init(_env: InitEnv[Calibrated], _plan: T1Plan) -> int:
        return 0

    @workflow("read-plots", plan=T1Plan, tunables=T1Tunables, init=init)
    def step(
        env: WorkflowEnv[Calibrated], plan: T1Plan, tun: T1Tunables, state: int
    ) -> Step[int, int]:
        del plan, tun
        attempts.append(state)
        if state == 0:
            env.plots.subplots("live")
            env.plots.subplots("reference")
            env.plots.draw("live", lambda figure: figure.set_facecolor("red"))
            yield from ()
            return Next(1, 1)
        env.plots.draw("live", lambda figure: figure.set_facecolor("blue"))
        if len(attempts) == 2:
            window.pause("reader")
            yield from env.wait_until(env.started_at)
        return Done()

    monkeypatch.setattr(window_module, "t1_fluxdep", step)
    assert isinstance(window.start("reader"), int)
    harness.complete_segment()
    assert harness.background.error is None
    assert rpc("workflow.figures", {"run_id": "reader"})["result"] == {
        "run_id": "reader",
        "names": ["live", "reference"],
    }
    assert window.figure_names("reader") == ("live", "reference")
    assert rpc("workflow.status", {})["ok"] is True and renders == []
    reply = rpc("workflow.figure", {"run_id": "reader", "name": "live"})
    result = reply["result"]
    assert isinstance(result, dict) and isinstance(result["png_base64"], str)
    png = b64decode(result["png_base64"], validate=True)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert result["run_id"] == "reader" and result["name"] == "live"
    assert renders == [owner_id]
    assert window.figure_png("reader", "live") == png
    assert figures[0] is figures[1]
    missing = rpc("workflow.figure", {"run_id": "reader", "name": "missing"})
    error = missing["error"]
    assert missing["ok"] is False and isinstance(error, dict)
    assert (error["code"], error["reason"]) == ("invalid_params", "unknown_figure")
    assert len(renders) == 2

    def failed_render(_figure: Figure) -> bytes:
        raise RuntimeError("Renderer failed")

    monkeypatch.setattr(window_module, "render_figure_png", failed_render)
    failed = rpc("workflow.figure", {"run_id": "reader", "name": "live"})
    error = failed["error"]
    assert failed["ok"] is False and isinstance(error, dict)
    assert error["code"] == "controller_error" and error["message"] == "Renderer failed"
    with pytest.raises(RuntimeError, match="Renderer failed"):
        window.figure_png("reader", "live")
    assert harness.gate.snapshot()
    monkeypatch.setattr(window_module, "render_figure_png", render)
    assert rpc("workflow." + ending, {"run_id": "reader"})["ok"] is True
    if ending == "resume":
        harness.complete_segment()
    assert window.figure_png("reader", "live") == png
    assert window.close()
    closed = rpc("workflow.figures", {"run_id": "reader"})
    error = closed["error"]
    assert closed["ok"] is False and isinstance(error, dict)
    assert (error["code"], error["reason"]) == ("precondition_failed", "no_plots")


def test_rpc_empty_figures_and_stale_run_do_not_render(
    attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    paused_tuning: list[tuple[int, float, int]],
    rpc: RpcRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, harness, _entry = attachment
    assert paused_tuning
    renders: list[Figure] = []

    def reject_render(figure: Figure) -> bytes:
        renders.append(figure)
        raise AssertionError("Read must not render")

    monkeypatch.setattr(window_module, "render_figure_png", reject_render)
    assert rpc("workflow.figures", {"run_id": "tuning"})["result"] == {
        "run_id": "tuning",
        "names": [],
    }
    for method in ("workflow.records", "workflow.figures", "workflow.figure"):
        stale = rpc(method, {"run_id": "old", "name": "live"})
        error = stale["error"]
        assert stale["ok"] is False and isinstance(error, dict)
        assert (error["code"], error["reason"], error["data"]) == (
            "precondition_failed",
            "run_mismatch",
            {"expected": "old", "actual": "tuning"},
        )
    assert renders == [] and harness.gate.snapshot()
    assert rpc("workflow.stop", {"run_id": "tuning"})["ok"] is True
    # New Start must not retain even a previous terminal run's figure membership.
    assert isinstance(window.start("replacement"), int)
    assert window.controller.status() is None
    stale = rpc("workflow.figures", {"run_id": "tuning"})
    assert stale["ok"] is False
    window.stop("replacement")
    harness.complete_segment()
