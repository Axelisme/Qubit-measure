"""Captured writeback receipts through registered synthetic recipes and GUI RPC."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.mcp.measure.recipe_api import (
    RecipeSession,
    WritebackReceipt,
    recipe,
)

from tests.resources.entry.fakes import register_fakes, registry_state, restore_registry

from ._recipe_writeback_support import WritebackGui
from ._script_recipe_support import Fit, Inputs, Report, SyntheticGui, start
from ._support import make_client


def test_failed_receipt_raises_with_owner_diagnostic_and_finally_saves(
    tmp_path: Path,
) -> None:
    gui = SyntheticGui()
    receipts: list[WritebackReceipt] = []

    def respond(method: str, params: dict[str, object]) -> dict[str, object]:
        if method in {"tab.writeback_write", "tab.writeback_accept"}:
            raise RemoteError(
                ErrorCode.PRECONDITION_FAILED, "captured destination is read-only"
            )
        return gui.respond(method, params)

    @recipe("synthetic", "Retain write failure and save available data")
    async def synthetic(session: RecipeSession, inputs: Inputs) -> Report:
        tab = session.open_tab("synthetic")
        await tab.run()
        try:
            await tab.analyze("primary", result=Fit)
            answer = await tab.ask_writeback()
            receipt = tab.accept_writeback(answer.items)
            receipts.append(receipt)
            receipt.raise_for_error()
            return Report(count=inputs.count)
        finally:
            tab.save()

    client = make_client(tmp_path, respond, recipes=(synthetic,))
    try:
        execution = start(client, "synthetic")
        execution.wait(3)
        pending = execution.snapshot().pending_question
        assert pending is not None
        execution.answer(
            {
                "request_id": pending.request_id,
                "answer": {
                    "accepted": True,
                    "items": pending.question.model_dump(mode="json")["proposal"][
                        "items"
                    ],
                },
            }
        )
        snapshot = execution.snapshot()
        assert receipts[0].status == "failed"
        assert snapshot.status == "failed"
        assert snapshot.error is not None
        assert "captured destination is read-only" in snapshot.error.message
        assert snapshot.outcome is not None
        assert snapshot.outcome.report is None
        assert len(snapshot.outcome.runs[0].saves) == 1
    finally:
        client.context.session.close()


@pytest.fixture
def writeback_registry() -> Iterator[None]:
    before = registry_state()
    try:
        register_fakes()
        installed = registry_state()
        yield
        assert registry_state() == installed
    finally:
        restore_registry(before)


@pytest.mark.parametrize("edit", ["target", "value", "module", "replacement"])
def test_accept_rejects_changed_post_before_any_primary_write(
    tmp_path: Path,
    edit: str,
    writeback_registry: None,
) -> None:
    gui = WritebackGui(tmp_path, module=edit == "module")
    receipts: list[WritebackReceipt] = []

    @recipe("synthetic", "Bind accepted candidates across both stages")
    async def synthetic(session: RecipeSession, inputs: Inputs) -> Report:
        tab = session.open_tab("synthetic")
        await tab.run()
        try:
            await tab.analyze("primary", result=Fit)
            await tab.analyze("post", result=Fit)
            answer = await tab.ask_writeback()
            receipts.append(tab.accept_writeback(answer.items))
            return Report(count=inputs.count)
        finally:
            tab.save()

    client = make_client(tmp_path, gui.respond, recipes=(synthetic,))
    try:
        execution = start(client, "synthetic")
        execution.wait(3)
        pending = execution.snapshot().pending_question
        assert pending is not None
        items = pending.question.model_dump(mode="json")["proposal"]["items"]
        params: dict[str, object] = {
            "tab_id": "tab-1",
            "subtab_id": "post_analysis",
            "id": "ml-1" if edit == "module" else "param-1",
        }
        if edit == "replacement":
            gui.replace_post()
        else:
            params.update(
                {"edits": [{"path": ["gain"], "value": 0.9}]}
                if edit == "module"
                else {"target_name": "general.ext.wrong"}
                if edit == "target"
                else {"proposed_value": 99.0}
            )
            gui.respond("tab.writeback_set", params)
        execution.answer(
            {
                "request_id": pending.request_id,
                "answer": {"accepted": True, "items": items},
            }
        )
        current = gui.read_context()
        assert not current.lookup("general.ext.second").present
        assert not current.lookup("general.ext.wrong").present
        assert "Q1.pulses.pi" not in current.library.module_keys()
        assert not current.lookup("general.ext.first").present, (
            "stale Post must reject the whole acceptance before Primary writes"
        )
        assert receipts[0].status == "failed"
        assert receipts[0].items == ()
        assert receipts[0].error is not None and "changed" in receipts[0].error
    finally:
        client.context.session.close()
        gui.owner.shutdown()


def test_alias_receipt_confirms_by_item_id_not_authored_target(tmp_path: Path) -> None:
    gui = SyntheticGui()
    receipts: list[WritebackReceipt] = []

    def respond(method: str, params: dict[str, object]) -> dict[str, object]:
        result = gui.respond(method, params)
        if method == "tab.writeback_preview":
            return {
                **result,
                "items": [
                    {
                        "id": "param-0",
                        "target_name": "$qubit.freq",
                        "actual_path": "Q1.freq",
                        "proposed": 3.5,
                    }
                ],
            }
        return result

    @recipe("synthetic", "Confirm an authored alias at its captured target")
    async def synthetic(session: RecipeSession, inputs: Inputs) -> Report:
        tab = session.open_tab("synthetic")
        await tab.run()
        await tab.analyze("primary", result=Fit)
        proposal = tab.writeback_proposal()
        receipts.append(tab.accept_writeback(proposal.items))
        return Report(count=inputs.count)

    client = make_client(tmp_path, respond, recipes=(synthetic,))
    try:
        execution = start(client, "synthetic")
        execution.wait(3)
        assert execution.snapshot().status == "completed"
        assert receipts[0].status == "completed"
        assert len(receipts[0].items) == 1
        assert receipts[0].items[0].name == "$qubit.freq"
        assert receipts[0].items[0].destination.path == "Q1.freq"
    finally:
        client.context.session.close()
