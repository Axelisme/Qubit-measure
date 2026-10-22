"""Post completion retains a read-only recovery route without replay."""

from pathlib import Path

import pytest
from zcu_tools.mcp.measure.recipe_api import RecipeSession, recipe

from ._analyze_support import analysis_client, analysis_start_reply
from ._script_recipe_support import Fit, Inputs, Report, SyntheticGui
from ._support import make_client

RECOVERY = {
    "rpc": "tab.get_post_analyze_result",
    "params": {"tab_id": "tab-1", "operation_id": 71},
    "stage": "post",
}


@pytest.mark.parametrize("recipe_mode", [False, True])
@pytest.mark.parametrize("seconds", [None, 10.0, 35.0, 60.0])
def test_post_read_timeout_returns_bound_recovery_without_reanalysis(
    tmp_path: Path,
    clients,
    monkeypatch: pytest.MonkeyPatch,
    recipe_mode: bool,
    seconds: float | None,
) -> None:
    gui = SyntheticGui()

    def respond(method, params):
        if method == "tab.post_analyze":
            started = {**analysis_start_reply({}, []), "recovery": RECOVERY}
            if seconds is not None:
                started["result_read_timeout_seconds"] = seconds
            return started
        return gui.respond(method, params)

    @recipe("synthetic", "Read Post once and retain a recovery route")
    async def synthetic(session: RecipeSession, inputs: Inputs) -> Report:
        tab = session.open_tab("synthetic")
        await tab.run()
        await tab.analyze("primary", result=Fit)
        await tab.analyze("post", result=Fit)
        return Report(count=inputs.count)

    client = (
        make_client(tmp_path, respond, recipes=(synthetic,))
        if recipe_mode
        else analysis_client(tmp_path, clients, respond)
    )
    if recipe_mode:
        clients.append(client)
    deadlines = []
    send = client.context.bridge.send_rpc_raw

    def record(method, params, timeout_seconds):
        if method == "tab.get_post_analyze_result":
            deadlines.append(timeout_seconds)
        return send(method, params, timeout_seconds)

    monkeypatch.setattr(client.context.bridge, "send_rpc_raw", record)
    client.transport.replies["tab.get_post_analyze_result"] = {
        "ok": False,
        "error": {"code": "timeout", "message": "synthetic result read timeout"},
    }
    reply = (
        client.call("synthetic", {"count": 1})
        if recipe_mode
        else client.call("tab_analyze", {"tab": "tab-1", "stage": "post"})
    )
    assert reply.data["status"] == "failed"
    assert reply.data.get("recovery") == {
        **RECOVERY,
        "op": client.context.session.expose_operation(71),
    }
    assert deadlines == [5.0 if seconds is None else seconds]
    execution = reply.data["execution"]
    for name in ("status", "wait"):
        queried = client.call(name, {"execution": execution})
        data = queried if name == "status" else queried.data
        assert data["recovery"] == reply.data["recovery"]
    methods = [method for method, _ in client.transport.sent]
    assert methods.count("tab.post_analyze") == 1
    assert methods.count("tab.get_post_analyze_result") == 1
    if recipe_mode:
        assert reply.data["failure"]["phase"] == "result_read"
