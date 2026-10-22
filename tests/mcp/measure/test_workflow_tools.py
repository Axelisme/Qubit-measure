"""Fixed workflow tools forward to the live measure owner without local state."""

import base64
from io import BytesIO

import pytest
from PIL import Image
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.mcp.core.reply import ToolReply
from zcu_tools.mcp.measure.session import GuiRpcError

from ._support import make_client


@pytest.mark.parametrize(
    ("tool", "arguments", "method"),
    [
        ("workflow_open", {"name": "synthetic"}, "workflow.open"),
        ("workflow_close", {"tab_id": "workflow-2"}, "workflow.close"),
        ("workflow_status", {}, "workflow.status"),
        ("workflow_status", {"run_id": "run"}, "workflow.status"),
        ("workflow_pause", {"run_id": "run"}, "workflow.pause"),
        ("workflow_stop", {"run_id": "run"}, "workflow.stop"),
        (
            "workflow_answer",
            {"run_id": "run", "request_id": "question-1", "answer": {"yes": True}},
            "workflow.answer",
        ),
        (
            "workflow_form",
            {"tab_id": "workflow-2", "form": "plan", "batch": []},
            "workflow.form",
        ),
        (
            "workflow_describe_form",
            {"tab_id": "workflow-2", "form": "tunables", "path": []},
            "workflow.describe_form",
        ),
        (
            "workflow_edit_form",
            {"tab_id": "workflow-2", "form": "tunables", "batch": []},
            "workflow.edit_form",
        ),
        ("workflow_tunables", {"run_id": "run", "batch": []}, "workflow.tunables"),
        ("workflow_records", {"run_id": "run", "since": 7}, "workflow.records"),
        ("workflow_figures", {"run_id": "run"}, "workflow.figures"),
    ],
)
def test_workflow_tools_forward_explicit_targets_and_preserve_native_reply(
    tmp_path, tool, arguments, method
):
    seen = []

    def respond(actual, params):
        seen.append((actual, params))
        return {"tab_id": "workflow-2", "native": {"revision": 4}}

    client = make_client(tmp_path, respond)
    try:
        assert client.call(tool, arguments) == {
            "tab_id": "workflow-2",
            "native": {"revision": 4},
        }
        assert seen == [(method, arguments)]
    finally:
        client.context.session.close()


def test_workflow_start_and_resume_share_existing_session_operation_mapping(tmp_path):
    def respond(method, params):
        if method == "operation.await":
            return {"reason": "completed", "status": "completed"}
        return {"tab_id": "workflow-1", "run_id": "native", "operation_id": 47}

    client = make_client(tmp_path, respond)
    try:
        started = client.call(
            "workflow_start", {"tab_id": "workflow-1", "run_id": "native"}
        )
        resumed = client.call("workflow_resume", {"run_id": "native"})
        assert started["handle"] == resumed["handle"]
        assert "operation_id" not in started
        assert (
            client.call("wait", {"op": started["handle"], "timeout": 0})["status"]
            == "completed"
        )
        assert (
            "operation.await",
            {"operation_id": 47, "timeout": 0.0},
        ) in client.transport.sent
    finally:
        client.context.session.close()


def test_workflow_errors_preserve_owner_envelope_without_retry(tmp_path):
    seen = []

    def respond(method, params):
        seen.append(method)
        raise RemoteError(
            ErrorCode.BUSY,
            "RUN lease held",
            reason="occupied",
            data={"tab_id": "other"},
        )

    client = make_client(tmp_path, respond)
    try:
        with pytest.raises(GuiRpcError) as error:
            client.call("workflow_start", {"tab_id": "workflow-1", "run_id": "native"})
        assert error.value.code == "busy"
        assert error.value.reason == "occupied"
        assert error.value.data == {"tab_id": "other"}
        assert seen == ["workflow.start"]
    finally:
        client.context.session.close()


@pytest.mark.parametrize("valid", [True, False])
def test_workflow_figure_delivers_validated_image_not_base64_text(tmp_path, valid):
    output = BytesIO()
    Image.new("RGB", (2, 2), "blue").save(output, format="PNG")
    png = output.getvalue() if valid else b"not a PNG"
    client = make_client(
        tmp_path,
        lambda method, params: {
            "run_id": "native",
            "tab_id": "workflow-1",
            "name": "trend",
            "png_b64": base64.b64encode(png).decode("ascii"),
        },
    )
    try:
        if not valid:
            with pytest.raises(ValueError, match="PNG"):
                client.call("workflow_figure", {"run_id": "native", "name": "trend"})
        else:
            reply = client.call(
                "workflow_figure", {"run_id": "native", "name": "trend"}
            )
            assert isinstance(reply, ToolReply)
            assert reply.data == {
                "run_id": "native",
                "tab_id": "workflow-1",
                "name": "trend",
            }
            assert reply.images[0].data == png
    finally:
        client.context.session.close()
