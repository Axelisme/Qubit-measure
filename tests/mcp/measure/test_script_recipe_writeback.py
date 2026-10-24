"""Captured writeback receipts through registered synthetic recipes and GUI RPC."""

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from pydantic import TypeAdapter
from zcu_tools.format_version import YamlValue
from zcu_tools.gui.app.measure.writeback.submissions import WritebackCompleted
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.mcp.measure.recipe_api import (
    RecipeSession,
    WritebackReceipt,
    recipe,
)

from tests.resources.entry.fakes import register_fakes, registry_state, restore_registry

from ._recipe_writeback_support import WritebackGui
from ._script_recipe_support import (
    Fit,
    Inputs,
    MappingWritebackGui,
    Report,
    SyntheticGui,
    answer_writeback,
    start,
)
from ._support import make_client


def test_recipe_timeout_retains_owner_submission_for_receipt_query(
    tmp_path: Path,
) -> None:
    gui = SyntheticGui()
    receipts: list[WritebackReceipt] = []

    def respond(method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "tab.writeback_begin":
            return {"submission_id": "wb-recipe", "status": "pending"}
        if method == "tab.writeback_accept":
            raise RemoteError(ErrorCode.TIMEOUT, "handler budget expired")
        return gui.respond(method, params)

    @recipe("synthetic", "Keep uncertain acceptance identity")
    async def synthetic(session: RecipeSession, inputs: Inputs) -> Report:
        tab = session.open_tab("synthetic")
        await tab.run()
        await tab.analyze("primary", result=Fit)
        proposal = await tab.ask_writeback()
        receipts.append(tab.accept_writeback(proposal.items))
        return Report(count=inputs.count)

    client = make_client(tmp_path, respond, recipes=(synthetic,))
    try:
        execution = start(client, "synthetic")
        answer_writeback(execution)
        assert len(receipts) == 1
        receipt = receipts[0]
        assert getattr(receipt, "submission_id", None) == "wb-recipe"
        assert receipt.status == "unknown"
        assert getattr(receipt, "recovery", None) == {
            "rpc": "tab.writeback_receipt",
            "params": {"submission_id": "wb-recipe"},
        }
        assert receipt.error is not None
        assert "Check status before deciding whether to resubmit" in receipt.error
    finally:
        client.context.session.close()


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


@pytest.fixture
def writeback_gui(
    tmp_path: Path,
    writeback_registry: None,
) -> Iterator[Callable[[bool, bool], WritebackGui]]:
    owners: list[WritebackGui] = []

    def create(module: bool, partial: bool) -> WritebackGui:
        gui = WritebackGui(tmp_path, module=module, partial=partial)
        owners.append(gui)
        return gui

    try:
        yield create
    finally:
        for gui in owners:
            gui.owner.shutdown()


@pytest.mark.parametrize("partial", [False, True])
def test_owner_submission_receipt_retains_prefix_and_rejects_duplicate_commit(
    writeback_gui: Callable[[bool, bool], WritebackGui], partial: bool
) -> None:
    gui = writeback_gui(False, partial)
    try:
        begun = gui.respond("tab.writeback_begin", {"tab_id": "tab-1"})
    except KeyError:
        pytest.fail(
            "Owner must allocate a queryable writeback submission before writes"
        )
    assert begun.get("status") == "pending"
    identity = begun["submission_id"]
    query = {"submission_id": identity}
    assert gui.respond("tab.writeback_receipt", query)["completed"] == []
    params = {
        "tab_id": "tab-1",
        "submission_id": identity,
        "stages": [
            {"subtab_id": "analysis", "ids": ["param-1"]},
            {
                "subtab_id": "post_analysis",
                "ids": ["param-1", "param-2", "param-3"] if partial else ["param-1"],
            },
        ],
    }
    gui.respond("tab.writeback_accept", params)
    receipt = gui.respond("tab.writeback_receipt", query)
    assert receipt["status"] == ("failed" if partial else "completed")
    completed = TypeAdapter(list[WritebackCompleted]).validate_python(
        receipt["completed"]
    )
    expected = [("analysis", "param-1", 20.0)]
    expected_targets = ["general.ext.first"]
    if not partial:
        expected.append(("post_analysis", "param-1", 30.0))
        expected_targets.append("general.ext.second")
    assert [
        (row.subtab_id, row.item.id, row.item.value) for row in completed
    ] == expected
    assert [row.item.target for row in completed] == expected_targets
    if partial:
        assert receipt["error"] == "synthetic setter refused the value"
        current = gui.read_context()
        assert not current.lookup("general.ext.second").present
        assert not current.lookup("general.ext.rejected").present
        assert not current.lookup("general.ext.later").present
    assert all(row.item.destination["point"] == "point" for row in completed)
    gui.respond(
        "tab.writeback_set",
        {
            "tab_id": "tab-1",
            "subtab_id": "analysis",
            "id": "param-1",
            "proposed_value": 999.0,
        },
    )
    assert gui.respond("tab.writeback_receipt", query) == receipt
    with pytest.raises(RemoteError, match="already submitted"):
        gui.respond("tab.writeback_accept", params)
    assert gui.read_context().value("general.ext.first") == 20.0


@pytest.mark.parametrize("edit", ["target", "value", "module", "replacement"])
def test_accept_writes_current_post_draft_after_question(
    tmp_path: Path,
    edit: str,
    writeback_gui: Callable[[bool, bool], WritebackGui],
) -> None:
    gui = writeback_gui(edit == "module", False)
    receipts: list[WritebackReceipt] = []

    @recipe("synthetic", "Accept current draft candidates across both stages")
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
        assert receipts[0].status == "completed"
        current = gui.read_context()
        assert current.value("general.ext.first") == 20.0
        if edit == "module":
            pulse = current.library.entry_value("Q1.pulses.pi", kind="module")
            assert isinstance(pulse, dict) and pulse["gain"] == 0.9
        elif edit == "target":
            assert current.value("general.ext.wrong") == 30.0
            assert not current.lookup("general.ext.second").present
        else:
            assert current.value("general.ext.second") == (
                99.0 if edit == "value" else 30.0
            )
        assert [item.stage for item in receipts[0].items] == ["primary", "post"]
        assert receipts[0].error is None
    finally:
        client.context.session.close()


@pytest.mark.parametrize("mode", ["parameters", "module", "partial"])
def test_captured_owner_writes_in_order_and_retains_confirmed_prefix(
    tmp_path: Path,
    mode: str,
    writeback_gui: Callable[[bool, bool], WritebackGui],
) -> None:
    gui = writeback_gui(mode == "module", mode == "partial")
    receipts: list[WritebackReceipt] = []

    @recipe("synthetic", "Submit captured candidates and preserve confirmed writes")
    async def synthetic(session: RecipeSession, inputs: Inputs) -> Report:
        tab = session.open_tab("synthetic")
        await tab.run()
        try:
            await tab.analyze("primary", result=Fit)
            await tab.analyze("post", result=Fit)
            answer = await tab.ask_writeback()
            receipt = tab.accept_writeback(answer.items)
            receipts.append(receipt)
            receipt.raise_for_error()
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
        execution.answer(
            {
                "request_id": pending.request_id,
                "answer": {"accepted": True, "items": items},
            }
        )
        current = gui.read_context()
        assert current.value("general.ext.first") == 20.0
        if mode == "module":
            pulse = current.library.entry_value("Q1.pulses.pi", kind="module")
            assert isinstance(pulse, dict) and pulse["gain"] == 0.2
        elif mode == "parameters":
            assert current.value("general.ext.second") == 30.0
        receipt = receipts[0]
        assert [item.stage for item in receipt.items] == (
            ["primary"] if mode == "partial" else ["primary", "post"]
        )
        snapshot = execution.snapshot()
        assert snapshot.status == ("failed" if mode == "partial" else "completed")
        assert receipt.status == ("failed" if mode == "partial" else "completed")
        if mode == "partial":
            assert (
                snapshot.error is not None
                and "setter refused" in snapshot.error.message
            )
            assert not current.lookup("general.ext.second").present
            assert not current.lookup("general.ext.rejected").present
            assert not current.lookup("general.ext.later").present
        assert snapshot.details[0].writeback_receipts == (receipt,)
        assert snapshot.outcome is not None and len(snapshot.outcome.runs[0].saves) == 1
    finally:
        client.context.session.close()


def change_gain(value: YamlValue, gain: float) -> None:
    """Edit the nested public module mapping, without replacing its outer object."""
    assert isinstance(value, dict)
    pulse = value["pulse"]
    assert isinstance(pulse, dict)
    pulse["gain"] = gain


def test_mutated_nested_proposal_is_rejected_without_owner_write(
    tmp_path: Path,
) -> None:
    gui = MappingWritebackGui()
    rejected: list[str] = []

    @recipe("synthetic", "Reject author edits to captured module values")
    async def synthetic(session: RecipeSession, inputs: Inputs) -> Report:
        tab = session.open_tab("synthetic")
        await tab.run()
        await tab.analyze("primary", result=Fit)
        proposal = await tab.ask_writeback()
        change_gain(proposal.items[0].value, 0.9)
        try:
            tab.accept_writeback(proposal.items)
        except ValueError as error:
            rejected.append(str(error))
        return Report(count=inputs.count)

    client = make_client(tmp_path, gui.respond, recipes=(synthetic,))
    try:
        execution = start(client, "synthetic")
        answer_writeback(execution)
        assert gui.writes == 0
        assert rejected == ["Unknown, duplicate or modified writeback candidate"]
        snapshot = execution.snapshot()
        assert snapshot.status == "completed"
        proposal = snapshot.details[0].writeback
        assert proposal is not None
        assert proposal.items[0].value == {"pulse": {"gain": 0.2}}
        change_gain(proposal.items[0].value, 0.7)
        fresh = execution.snapshot().details[0].writeback
        assert fresh is not None and fresh.items[0].value == {"pulse": {"gain": 0.2}}
    finally:
        client.context.session.close()


def test_receipt_mapping_cannot_mutate_retained_candidates_or_details(
    tmp_path: Path,
) -> None:
    gui = MappingWritebackGui()
    failures: list[str] = []

    @recipe("synthetic", "Keep public receipt edits separate from captured evidence")
    async def synthetic(session: RecipeSession, inputs: Inputs) -> Report:
        tab = session.open_tab("synthetic")
        await tab.run()
        await tab.analyze("primary", result=Fit)
        proposal = await tab.ask_writeback()
        receipt = tab.accept_writeback(proposal.items)
        change_gain(receipt.items[0].value, 0.9)
        try:
            tab.accept_writeback(proposal.items).raise_for_error()
        except ValueError as error:
            failures.append(str(error))
        return Report(count=inputs.count)

    client = make_client(tmp_path, gui.respond, recipes=(synthetic,))
    try:
        execution = start(client, "synthetic")
        answer_writeback(execution)
        snapshot = execution.snapshot()
        assert snapshot.status == "completed"
        assert failures == []
        proposal = snapshot.details[0].writeback
        assert proposal is not None
        assert proposal.items[0].value == {"pulse": {"gain": 0.2}}
        receipts = snapshot.details[0].writeback_receipts
        assert len(receipts) == 2
        assert all(
            receipt.items[0].value == {"pulse": {"gain": 0.2}} for receipt in receipts
        )
        change_gain(receipts[0].items[0].value, 0.7)
        assert execution.snapshot().details[0].writeback_receipts[0].items[0].value == {
            "pulse": {"gain": 0.2}
        }
    finally:
        client.context.session.close()


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
        proposal = await tab.ask_writeback()
        receipts.append(tab.accept_writeback(proposal.items))
        return Report(count=inputs.count)

    client = make_client(tmp_path, respond, recipes=(synthetic,))
    try:
        execution = start(client, "synthetic")
        answer_writeback(execution)
        assert execution.snapshot().status == "completed"
        assert receipts[0].status == "completed"
        assert len(receipts[0].items) == 1
        assert receipts[0].items[0].name == "$qubit.freq"
        assert receipts[0].items[0].destination.path == "Q1.freq"
    finally:
        client.context.session.close()
