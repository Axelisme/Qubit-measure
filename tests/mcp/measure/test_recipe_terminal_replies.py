"""Terminal recipe identification and discovery through the public MCP tools."""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from zcu_tools.mcp.measure.recipe_api import RecipeSession, recipe

from ._script_recipe_support import ConfirmQuestion, Fit, Inputs, Report, SyntheticGui
from ._support import make_client


@pytest.mark.parametrize("terminal", ["completed", "failed", "cancelled"])
def test_terminal_recipe_reply_keeps_queryable_execution_id(
    tmp_path: Path, terminal: str
) -> None:
    @recipe("synthetic", "Return or stop a synthetic recipe")
    async def synthetic(session: RecipeSession, inputs: Inputs) -> Report:
        if terminal == "failed":
            raise ValueError("synthetic failure")
        if terminal == "cancelled":
            await session.ask(ConfirmQuestion(evidence="cancel here"))
        return Report(count=inputs.count)

    client = make_client(tmp_path, recipes=(synthetic,))
    try:
        reply = client.call("synthetic", {"count": "12"})
        if terminal == "cancelled":
            client.call("cancel", {"execution": reply.data["execution"]})
            client.context.session.recipes.get(reply.data["execution"]).join()
            reply = client.call("wait", {"execution": reply.data["execution"]})
        assert reply.data["status"] == terminal
        assert "execution" in reply.data
        assert reply.data["report"] == (
            {"count": 12} if terminal == "completed" else None
        )
        assert reply.data["runs"] == []
        if terminal == "failed":
            assert reply.data["failure"] == {
                "phase": "operation",
                "reason": "synthetic failure",
            }
        execution_id = reply.data["execution"]
        full = client.call("status", {"execution": execution_id, "detail": "full"})
        assert full["execution"] == execution_id
        assert full["status"] == terminal
        assert full["outcome"]["report"] == (
            {"count": 12} if terminal == "completed" else None
        )
        assert client.call("wait", {"execution": execution_id}).data == reply.data
        assert client.call("status", {"execution": execution_id}) == {
            **reply.data,
            "active_path": full["active_path"],
            "live_paths": ["recipe:synthetic"],
        }
    finally:
        client.context.session.close()


def test_global_status_discovers_latest_terminal_recipes_and_retains_older_ids(
    tmp_path: Path,
) -> None:
    @recipe("synthetic", "Complete a recipe or ask a question")
    async def synthetic(session: RecipeSession, inputs: Inputs) -> Report:
        if inputs.count == 0:
            await session.ask(ConfirmQuestion(evidence="still active"))
        return Report(count=inputs.count)

    client = make_client(tmp_path, recipes=(synthetic,))
    client.transport.replies.update(
        {
            "state.has_project": {"ok": True, "result": {"value": False}},
            "state.has_active_context": {"ok": True, "result": {"value": False}},
            "state.has_soc": {"ok": True, "result": {"value": False}},
            "context.active": {"ok": True, "result": {"label": None}},
            "device.list": {"ok": True, "result": {"devices": []}},
            "tab.snapshot": {"ok": True, "result": {"tabs": []}},
            "operation.active": {"ok": True, "result": {"operations": []}},
        }
    )
    try:
        # Discover IDs from global status, independently of recipe summaries.
        completed = []
        for count in range(1, 13):
            client.call("synthetic", {"count": count})
            status = client.call("status", {})
            assert "recent_terminal_executions" in status
            latest = status["recent_terminal_executions"][0]
            full = client.call(
                "status", {"execution": latest["execution"], "detail": "full"}
            )
            assert full["outcome"]["report"] == {"count": count}
            assert datetime.fromisoformat(latest["finished_at"]).tzinfo == timezone.utc
            completed.append(latest)

        pending = client.call("synthetic", {"count": 0}).data
        status = client.call("status", {})
        assert status["recent_terminal_executions"] == list(reversed(completed[-10:]))
        assert all(
            item.keys() == {"execution", "recipe", "status", "finished_at"}
            and item["recipe"] == "synthetic"
            and item["status"] == "completed"
            for item in status["recent_terminal_executions"]
        )
        assert status["terminal_count"] == 12
        assert [item["execution"] for item in status["executions"]] == [
            pending["execution"]
        ]
        assert (
            client.call("status", {})["recent_terminal_executions"]
            == status["recent_terminal_executions"]
        )
        older = client.call("status", {"execution": completed[0]["execution"]})
        assert older["report"] == {"count": 1}
    finally:
        client.context.session.close()


@pytest.mark.parametrize(
    ("failed_method", "phase"),
    [
        ("tab.new", "operation"),
        ("tab.run_start", "operation"),
        ("tab.analyze", "operation"),
        ("tab.get_analyze_result", "result_read"),
        ("tab.save_artifacts", "image_save"),
        ("tab.get_figure", "figure_read"),
        ("tab.writeback_preview", "writeback_read"),
    ],
)
def test_recipe_failure_identifies_the_failed_runtime_action(
    tmp_path: Path, failed_method: str, phase: str
) -> None:
    gui = SyntheticGui()

    @recipe("synthetic", "Fail at a runtime boundary")
    async def synthetic(session: RecipeSession, inputs: Inputs) -> Report:
        tab = session.open_tab("synthetic")
        await tab.run()
        if failed_method == "tab.save_artifacts":
            tab.save()
        elif failed_method == "tab.get_figure":
            tab.preview()
        else:
            await tab.analyze("primary", result=Fit)
            tab.writeback_proposal()
        return Report(count=inputs.count)

    client = make_client(tmp_path, gui.respond, recipes=(synthetic,))
    client.transport.replies[failed_method] = {
        "ok": False,
        "error": {"code": "internal", "message": "synthetic owner failure"},
    }
    try:
        reply = client.call("synthetic", {"count": 9})
        assert reply.data["status"] == "failed"
        assert "failure" in reply.data
        assert reply.data["failure"] == {
            "phase": phase,
            "reason": "GUI Error (internal): synthetic owner failure",
        }
        full = client.call(
            "status", {"execution": reply.data["execution"], "detail": "full"}
        )
        assert full["error"]["message"] == reply.data["failure"]["reason"]
    finally:
        client.context.session.close()


@pytest.mark.parametrize("recover", [False, True])
def test_analysis_decode_failure_survives_cleanup_but_not_author_recovery(
    tmp_path: Path, recover: bool
) -> None:
    gui = SyntheticGui()

    def respond(method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "tab.get_analyze_result":
            return {"summary": {}, "invalid": []}
        return gui.respond(method, params)

    @recipe("synthetic", "Decode an incompatible summary with cleanup")
    async def synthetic(session: RecipeSession, inputs: Inputs) -> Report:
        tab = session.open_tab("synthetic")
        await tab.run()
        try:
            await tab.analyze("primary", result=Fit)
        except ValueError:
            if not recover:
                raise
        finally:
            tab.save()
        raise RuntimeError("author failed after recovery")

    client = make_client(tmp_path, respond, recipes=(synthetic,))
    try:
        reply = client.call("synthetic", {"count": 9})
        assert reply.data["status"] == "failed"
        assert "failure" in reply.data
        failure = reply.data["failure"]
        assert failure["phase"] == ("operation" if recover else "result_read")
        if recover:
            assert failure["reason"] == "author failed after recovery"
        else:
            assert "estimate" in failure["reason"]
        assert reply.data["runs"][0]["saves"][0]["status"] == "completed"
    finally:
        client.context.session.close()
