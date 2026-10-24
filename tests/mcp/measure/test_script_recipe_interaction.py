"""Recipe-owned done joins the next handoff through public MCP tools."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest
from zcu_tools.mcp.core.reply import ToolReply
from zcu_tools.mcp.measure.recipe_api import RecipeSession, recipe
from zcu_tools.mcp.measure.recipe_host import (
    RecipeHost,
    RecipeHostFactory,
    create_recipe_host,
)
from zcu_tools.mcp.measure.tool_context import MeasureToolContext

from ._script_recipe_support import Fit, Inputs, Report, SyntheticGui
from ._support import make_client


def blocking_host_factory(
    published: Event, release_first: Event, release_next: Event
) -> RecipeHostFactory:
    """Hold each interactive publication until its test-controlled release."""

    def host_factory(
        tools: MeasureToolContext, on_change: Callable[[], None], name: str
    ) -> RecipeHost:
        def publish() -> None:
            on_change()
            pending = host.runtime.pending_analysis
            if pending is not None:
                release = release_first if pending.stage == "primary" else release_next
                published.set()
                # The test releases each published handoff before closing the session.
                assert release.wait(30), "Test did not release the published handoff"

        host = create_recipe_host(tools, publish, name)
        return host

    return host_factory


class InteractiveGui(SyntheticGui):
    """Offer two interactive stages and signal native done acceptance."""

    def __init__(self, done_accepted: Event) -> None:
        super().__init__()
        self.done_accepted = done_accepted

    def respond(self, method: str, params: dict[str, object]) -> dict[str, object]:
        if method in {"tab.analyze", "tab.post_analyze"}:
            return {
                "operation_id": 100 if method == "tab.analyze" else 101,
                "interactive": True,
            }
        if method == "tab.interact":
            self.done_accepted.set()
            return {"operation_id": 100, "figure": None}
        if method == "tab.get_post_analyze_result":
            return {"summary": {"estimate": 4.5}, "invalid": []}
        return super().respond(method, params)


@pytest.mark.parametrize(
    "following", ["question", "terminal", "interactive", "already_terminal"]
)
def test_done_returns_next_recipe_handoff(tmp_path: Path, following: str) -> None:
    published, release_first, release_next = Event(), Event(), Event()
    done_accepted = Event()

    @recipe("synthetic", "Continue after the original interactive analysis")
    async def synthetic(session: RecipeSession, inputs: Inputs) -> Report:
        tab = session.open_tab("synthetic")
        await tab.run()
        await tab.analyze("primary", result=Fit, interactive=True)
        if following == "question":
            await tab.ask_writeback()
        elif following == "interactive":
            await tab.analyze("post", result=Fit, interactive=True)
        return Report(count=inputs.count)

    client = make_client(
        tmp_path,
        InteractiveGui(done_accepted).respond,
        recipes=(synthetic,),
        recipe_host_factory=blocking_host_factory(
            published, release_first, release_next
        ),
    )
    try:
        initial = client.call("synthetic", {"count": 9}).data
        key = initial["execution"]
        execution = client.context.session.recipes.get(key)
        # The first publication must arrive before done; the callback holds it stable.
        assert published.wait(30)
        assert initial["pending"] == {"tab": "tab-1", "stage": "primary", "op": 2}
        if following == "already_terminal":
            release_first.set()
            execution.join()

        def done() -> ToolReply:
            return client.call(
                "tab_interact", {"tab": "tab-1", "payload": {"command": "done"}}
            )

        with ThreadPoolExecutor(max_workers=1) as callers:
            pending_reply = callers.submit(done)
            try:
                # Native done has arrived; releasing the owner guarantees continuation.
                assert done_accepted.wait(30)
                release_first.set()
                reply = pending_reply.result(timeout=30).data
            finally:
                release_first.set()
                release_next.set()
        assert reply["execution"] == key
        if following == "question":
            assert reply["status"] == "paused"
            assert (
                reply["pending_question"]["question"]["proposal"]["items"][0]["name"]
                == "Q1.freq"
            )
        elif following == "interactive":
            assert reply["status"] == "interactive"
            assert reply["pending"] == {"tab": "tab-1", "stage": "post", "op": 3}
        else:
            assert reply["status"] == "completed"
            assert reply["report"] == {"count": 9}
        full = client.call("status", {"execution": key, "detail": "full"})
        assert full["details"][0]["analyses"][0]["summary"] == {"estimate": 3.5}
    finally:
        release_first.set()
        release_next.set()
        client.context.session.close()
