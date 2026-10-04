"""Author analysis/question contracts on the native GUI transport seam."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Event

import pytest
from zcu_tools.mcp.measure.recipe import (
    AnalysisStage,
    RecipeAnalysis,
    RecipeParameter,
    RecipeRun,
    RecipeScalar,
    RecipeSession,
)
from zcu_tools.mcp.measure.session import GuiRpcError

from ._recipe_analysis_support import AnalysisRecipeGui
from ._recipe_support import recipe_client


def completed_run(session: RecipeSession) -> RecipeRun:
    """Obtain one real author Run handle through its completion contract."""
    run, status = session.open_tab("lookback").run().complete_in_current_worker()
    assert isinstance(run, RecipeRun) and status == "completed"
    return run


@pytest.mark.parametrize("unpack", [tuple])
def test_analysis_starts_immediately_and_binds_primary_and_post_sources(
    tmp_path, unpack
):
    gui = AnalysisRecipeGui()
    with recipe_client(tmp_path, gui) as client:
        session = RecipeSession(client.context)
        run = completed_run(session)
        labels: list[RecipeScalar] = ["a", "b"]
        updates: dict[str, RecipeParameter] = {
            "threshold": 0.7,
            "labels": labels,
            "enabled": True,
        }
        operation = run.analyze("primary", params=updates)
        labels.append("later")
        assert (
            "tab.analyze",
            {
                "tab_id": "t",
                "updates": {"threshold": 0.7, "labels": ["a", "b"], "enabled": True},
                "run_operation_id": 71,
            },
        ) in client.transport.sent
        assert not any(
            method == "operation.await" and params["operation_id"] == 93
            for method, params in client.transport.sent
        )
        before = operation.snapshot()
        assert before is not None and before.start.status == "running"
        assert before.params == {"threshold": 0.5}
        assert operation.preview_images() == ()
        with pytest.raises(TypeError):
            unpack(operation)
        analysis, status = operation.complete_in_current_worker()
        assert isinstance(analysis, RecipeAnalysis) and status == "completed"
        capture = analysis.snapshot
        assert capture.result is not None
        assert capture.result.summary == {"frequency": 5.0, "frequency_error": None}
        assert capture.result.invalid == [
            {"path": "summary.frequency_error", "reason": "non_finite"}
        ]
        assert capture.save_status == "saved"
        assert [image.image_path for image in capture.saved_images] == [
            "/actual/fit.png",
            "/actual/residual.png",
        ]
        assert capture.figure is not None and len(operation.preview_images()) == 1
        post = run.analyze("post")
        post_result, status = post.complete_in_current_worker()
        assert isinstance(post_result, RecipeAnalysis) and status == "completed"
        assert (
            "tab.post_analyze",
            {"tab_id": "t", "updates": {}, "run_operation_id": 71, "operation_id": 93},
        ) in client.transport.sent
        captured = run.snapshot()
        assert captured.analyses["primary"] == capture
        assert captured.analyses["post"] == post_result.snapshot
        calls = list(client.transport.sent)
        assert capture.params is not None
        capture.params["threshold"] = -1
        assert session.analysis_snapshot() == post_result.snapshot
        assert run.snapshot().analyses["primary"].params == {"threshold": 0.5}
        assert client.transport.sent == calls
        with pytest.raises(ValueError, match="already started"):
            operation.complete_in_current_worker()


@pytest.mark.parametrize("started", [False, True])
def test_post_requires_this_runs_completed_primary_not_the_current_pane(
    tmp_path, started
):
    gui = AnalysisRecipeGui()
    with recipe_client(tmp_path, gui) as client:
        run = completed_run(RecipeSession(client.context))
        if started:
            run.analyze("primary")
        with pytest.raises(GuiRpcError) as error:
            run.analyze("post")
        assert error.value.reason == "primary_result_unavailable"
        assert not any(
            method == "tab.post_analyze" for method, _ in client.transport.sent
        )


@pytest.mark.parametrize("stage", ["other"])
@pytest.mark.parametrize(
    "params", [{"bad": float("nan")}, {"bad": [[1]]}, {"bad": {"x": 1}}]
)
def test_analysis_rejects_values_outside_the_parameter_contract_before_dispatch(
    tmp_path, params, stage
):
    gui = AnalysisRecipeGui()
    with recipe_client(tmp_path, gui) as client:
        run = completed_run(RecipeSession(client.context))
        with pytest.raises(ValueError, match="bad"):
            run.analyze("primary", params=params)
        with pytest.raises(ValueError, match="stage"):
            run.analyze(stage)
        assert not any(method == "tab.analyze" for method, _ in client.transport.sent)


@pytest.mark.parametrize("at_admission", [False, True])
def test_pending_cancel_suppresses_only_one_analysis_start(tmp_path, at_admission):
    gui = AnalysisRecipeGui()
    pending = Event()
    checks = 0
    armed = False

    def consume_cancel():
        nonlocal checks
        checks += 1
        if armed and at_admission and checks == 2:
            pending.set()
        if pending.is_set():
            pending.clear()
            return True
        return False

    with recipe_client(tmp_path, gui) as client:
        # Construct the Run before introducing the analysis cancellation policy.
        session = RecipeSession(client.context, consume_cancel=consume_cancel)
        run = completed_run(session)
        checks = 0
        armed = True
        if not at_admission:
            pending.set()
        before = list(client.transport.sent)
        suppressed = run.analyze("primary")
        assert suppressed.snapshot() is None
        assert suppressed.complete_in_current_worker() == (None, "cancelled")
        assert suppressed.cancel() is None
        assert session.analysis_snapshot() is None
        assert "primary" not in run.snapshot().analyses
        assert client.transport.sent == before
        if at_admission:
            registrations = client.context.session.executions.snapshots()
            assert registrations[-1].status == "cancelled"
            assert registrations[-1].start.status == "not_started"
            assert registrations[-1].operation_outcome is None
        operation = run.analyze("primary")
        result, status = operation.complete_in_current_worker()
        assert isinstance(result, RecipeAnalysis) and status == "completed"
        assert sum(method == "tab.analyze" for method, _ in client.transport.sent) == 1


@pytest.mark.parametrize("rejected", [False, True])
def test_unyielded_analysis_failure_keeps_unknown_or_rejected_receipt(
    tmp_path, rejected
):
    gui = AnalysisRecipeGui()
    with recipe_client(tmp_path, gui) as client:
        session = RecipeSession(client.context)
        run = completed_run(session)
        client.transport.replies["tab.analyze"] = {
            "ok": False,
            "error": {
                "code": "precondition_failed" if rejected else "timeout",
                "message": "analysis start interrupted",
                "reason": "stale" if rejected else "transport_lost",
            },
        }
        with pytest.raises(GuiRpcError, match="interrupted"):
            run.analyze("primary")
        capture = session.analysis_snapshot()
        assert capture is not None and capture.status == "failed"
        assert capture.start.status == ("not_started" if rejected else "unknown")
        assert capture.start.reason == ("stale" if rejected else None)
        assert capture.op is None and capture.operation_outcome is None
        assert run.snapshot().analyses["primary"] == capture
        assert sum(method == "tab.analyze" for method, _ in client.transport.sent) == 1


@pytest.mark.parametrize("through_owner", [False, True])
def test_analysis_cancellation_delivers_none_without_reading_results(
    tmp_path, through_owner
):
    gui = AnalysisRecipeGui()
    with recipe_client(tmp_path, gui) as client:
        run = completed_run(RecipeSession(client.context))
        operation = run.analyze("primary")
        if through_owner:
            reply = operation.cancel()
            assert (
                reply is not None and reply.data["gui_cancel"]["status"] == "requested"
            )
        else:
            gui.outcomes["primary"] = "cancelled"
        assert operation.complete_in_current_worker() == (None, "cancelled")
        capture = operation.snapshot()
        assert capture is not None and capture.status == "cancelled"
        assert capture.operation_outcome is not None
        assert capture.operation_outcome["status"] == "cancelled"
        assert capture.result is None and capture.writeback is None
        assert not any(
            method
            in ("tab.get_analyze_result", "tab.save_image", "tab.writeback_preview")
            for method, _ in client.transport.sent
        )


def test_two_interactive_stages_join_the_same_completion_owners(tmp_path):
    gui = AnalysisRecipeGui(interactive=True)
    with recipe_client(tmp_path, gui) as client:
        session = RecipeSession(client.context)
        run = completed_run(session)
        stages: tuple[tuple[AnalysisStage, int], ...] = (("primary", 93), ("post", 104))
        with ThreadPoolExecutor(max_workers=1) as pool:
            for stage, wire_op in stages:
                operation = run.analyze(stage)
                capture = operation.snapshot()
                assert capture is not None and capture.status == "interactive"
                assert capture.interaction is not None
                assert capture.interaction["state"] == {"stage": stage}
                assert capture.writeback is None and capture.result is None
                assert len(operation.preview_images()) == 1
                future = pool.submit(operation.complete_in_current_worker)
                try:
                    assert gui.awaited[stage].wait(1.0)
                    assert not future.done()
                    progress = session.analysis_snapshot()
                    assert progress is not None and progress.status == "interactive"
                    reply = client.call(
                        "tab_interact", {"tab": "t", "payload": {"command": "done"}}
                    )
                    assert not reply.is_error
                    result, status = future.result(timeout=2.0)
                    assert isinstance(result, RecipeAnalysis) and status == "completed"
                    assert result.snapshot.writeback == gui.writebacks[stage]
                finally:
                    gui.done[stage].set()
                    operation.wake()
                assert (
                    sum(
                        method == "tab.writeback_preview"
                        and params["operation_id"] == wire_op
                        for method, params in client.transport.sent
                    )
                    == 1
                )
        assert len(client.context.session.executions.snapshots()) == 2


@pytest.mark.parametrize("lifetime", ["recipe", "session"])
def test_lifetime_close_wakes_analysis_without_closing_the_gui_connection(
    tmp_path, lifetime
):
    gui = AnalysisRecipeGui(interactive=True)
    closed = Event()
    with recipe_client(tmp_path, gui) as client:
        run = completed_run(RecipeSession(client.context, closed=closed))
        operation = run.analyze("primary")
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(operation.complete_in_current_worker)
            try:
                assert gui.awaited["primary"].wait(1.0)
            finally:
                if lifetime == "recipe":
                    closed.set()
                    operation.wake()
                else:
                    client.context.session.executions.stop_admission()
                    assert not closed.is_set()
            with pytest.raises(GuiRpcError) as error:
                future.result(timeout=1.0)
            assert error.value.reason == "session_closed"
        capture = operation.snapshot()
        assert capture is not None and capture.status == "failed"
        assert capture.start.status == "running"
        assert capture.op is not None and capture.result is None
        assert client.transport.is_open


@pytest.mark.parametrize("phase", ["result", "save", "writeback"])
def test_analysis_failures_keep_the_original_source_and_confirmed_prefix(
    tmp_path, phase
):
    gui = AnalysisRecipeGui()
    with recipe_client(tmp_path, gui) as client:
        run = completed_run(RecipeSession(client.context))
        raw = run.save_raw()
        operation = run.analyze("primary")
        method = {
            "result": "tab.get_analyze_result",
            "save": "tab.save_image",
            "writeback": "tab.writeback_preview",
        }[phase]
        failure = {
            "ok": False,
            "error": {
                "code": "precondition_failed",
                "message": "source was replaced",
                "reason": "result_superseded",
            },
        }
        if phase == "save":
            client.transport.replies[method] = lambda params: (
                {"ok": True, "result": gui(method, params)}
                if params["figure_name"] == "fit"
                else failure
            )
        else:
            client.transport.replies[method] = failure
        with pytest.raises(GuiRpcError) as error:
            operation.complete_in_current_worker()
        assert error.value.reason == "result_superseded"
        capture = operation.snapshot()
        assert capture is not None and capture.status == "failed"
        assert capture.error is not None
        assert (
            capture.error.phase
            == {
                "result": "result_read",
                "save": "image_save",
                "writeback": "writeback_read",
            }[phase]
        )
        assert [image.image_path for image in capture.saved_images] == {
            "result": [],
            "save": ["/actual/fit.png"],
            "writeback": ["/actual/fit.png", "/actual/residual.png"],
        }[phase]
        assert (capture.result is None) == (phase == "result")
        assert capture.writeback is None
        assert run.snapshot().raw_save == raw
        assert run.snapshot().analyses["primary"] == capture
        assert sum(method == "tab.analyze" for method, _ in client.transport.sent) == 1


@pytest.mark.parametrize("unpack", [tuple])
@pytest.mark.parametrize(
    "items,expected",
    [(None, ("frequency", "linewidth")), ([], ()), (["linewidth"], ("linewidth",))],
)
def test_question_uses_captured_names_without_new_reads_or_writes(
    tmp_path, items, expected, unpack
):
    gui = AnalysisRecipeGui()
    with recipe_client(tmp_path, gui) as client:
        run = completed_run(RecipeSession(client.context))
        run.analyze("primary").complete_in_current_worker()
        run.analyze("post").complete_in_current_worker()
        previews = deepcopy(gui.writebacks)
        before = list(client.transport.sent)
        gui.writebacks["primary"]["items"].clear()
        question = run.propose_writeback(items)
        snapshot = question.snapshot()
        assert snapshot.items == expected
        assert (
            snapshot.preview.primary is not None and snapshot.preview.post is not None
        )
        assert [item["target_name"] for item in snapshot.preview.primary["items"]] == (
            ["frequency"] if "frequency" in expected else []
        )
        assert [item["target_name"] for item in snapshot.preview.post["items"]] == (
            ["linewidth"] if "linewidth" in expected else []
        )
        assert (
            snapshot.preview.primary["destination_context"]
            == previews["primary"]["destination_context"]
        )
        snapshot.preview.post["items"].clear()
        later = question.snapshot().preview.post
        assert later is not None
        assert len(later["items"]) == ("linewidth" in expected)
        with pytest.raises(TypeError):
            unpack(question)
        assert client.transport.sent == before


@pytest.mark.parametrize("mode", ["unknown", "repeated", "ambiguous"])
def test_question_rejects_invalid_names_before_any_prompt_or_write(tmp_path, mode):
    gui = AnalysisRecipeGui()
    if mode == "ambiguous":
        gui.writebacks["post"]["items"][0]["target_name"] = "frequency"
    with recipe_client(tmp_path, gui) as client:
        run = completed_run(RecipeSession(client.context))
        run.analyze("primary").complete_in_current_worker()
        run.analyze("post").complete_in_current_worker()
        before = list(client.transport.sent)
        with pytest.raises(
            ValueError,
            match={
                "unknown": "Unknown",
                "repeated": "Duplicate",
                "ambiguous": "Ambiguous",
            }[mode],
        ):
            run.propose_writeback(
                {
                    "unknown": ["missing"],
                    "repeated": ["frequency", "frequency"],
                    "ambiguous": None,
                }[mode]
            )
        assert client.transport.sent == before


def test_failed_native_analysis_is_not_changed_to_cancelled_by_stop_intent(tmp_path):
    gui = AnalysisRecipeGui()
    with recipe_client(tmp_path, gui) as client:
        run = completed_run(RecipeSession(client.context))
        operation = run.analyze("primary")
        operation.cancel()
        gui.outcomes["primary"] = "failed"
        with pytest.raises(GuiRpcError) as error:
            operation.complete_in_current_worker()
        assert error.value.reason == "analysis_failed"
        capture = operation.snapshot()
        assert capture is not None and capture.cancel_requested
        assert capture.status == "failed"
        assert capture.operation_outcome is not None
        assert capture.operation_outcome["status"] == "failed"
        assert capture.result is None


def test_uncertain_analysis_save_retains_confirmed_image_and_unconfirmed_name(tmp_path):
    gui = AnalysisRecipeGui()
    with recipe_client(tmp_path, gui) as client:
        run = completed_run(RecipeSession(client.context))
        operation = run.analyze("primary")
        client.transport.replies["tab.save_image"] = lambda params: (
            {"ok": True, "result": gui("tab.save_image", params)}
            if params["figure_name"] == "fit"
            else {
                "ok": False,
                "error": {"code": "timeout", "message": "save response lost"},
            }
        )
        with pytest.raises(GuiRpcError, match="save response lost"):
            operation.complete_in_current_worker()
        capture = operation.snapshot()
        assert capture is not None and capture.status == "failed"
        assert capture.save_status == "unknown"
        assert capture.unconfirmed_image == "residual"
        assert capture.remaining_images == ["residual"]
        assert [image.image_path for image in capture.saved_images] == [
            "/actual/fit.png"
        ]
        assert (
            sum(method == "tab.save_image" for method, _ in client.transport.sent) == 2
        )


def test_analysis_without_images_still_captures_writeback_and_empty_question_is_local(
    tmp_path,
):
    gui = AnalysisRecipeGui()
    gui.names["primary"] = []
    with recipe_client(tmp_path, gui) as client:
        run = completed_run(RecipeSession(client.context))
        before = list(client.transport.sent)
        question = run.propose_writeback().snapshot()
        assert question.items == ()
        assert question.preview.primary is None and question.preview.post is None
        assert client.transport.sent == before
        operation = run.analyze("primary")
        result, status = operation.complete_in_current_worker()
        assert isinstance(result, RecipeAnalysis) and status == "completed"
        assert result.snapshot.save_status == "not_available"
        assert result.snapshot.writeback == gui.writebacks["primary"]
        assert operation.preview_images() == ()
        assert not any(
            method in ("tab.save_image", "tab.get_figure")
            for method, _ in client.transport.sent
        )
        assert run.propose_writeback().snapshot().items == ("frequency",)
