"""Original Run preview capture and delivery through public author/tool seams."""

import base64
from pathlib import Path

import pytest
from zcu_tools.mcp.measure.recipe import RecipeGenerator, RecipeRun, RecipeSession
from zcu_tools.mcp.measure.session import GuiRpcError

from ._recipe_execution_support import definition
from ._recipe_support import PNG, LookbackGui, recipe_client
from ._support import full_execution_reply, make_client


def test_run_preview_retains_confirmed_capture_after_a_later_source_rejection(tmp_path):
    gui = LookbackGui()

    def respond(method, params):
        if method == "tab.get_figure":
            assert params == {
                "tab_id": "t",
                "subtab_id": "run",
                "run_operation_id": 71,
            }
            return {"png_b64": base64.b64encode(PNG).decode()}
        return gui(method, params)

    with recipe_client(tmp_path, respond) as client:
        operation = RecipeSession(client.context).open_tab("lookback").run()
        run, status = operation.complete_in_current_worker()
        assert isinstance(run, RecipeRun) and status == "completed"
        run.save_raw()
        run.preview()
        captured = run.snapshot()
        assert captured.preview is not None
        path = Path(captured.preview.path)
        assert path.read_bytes() == PNG
        assert captured.preview_status == "captured"
        assert captured.preview_images[0].data == PNG
        client.transport.replies["tab.get_figure"] = {
            "ok": False,
            "error": {
                "code": "precondition_failed",
                "reason": "result_superseded",
                "message": "Another Run owns the canvas",
            },
        }
        with pytest.raises(GuiRpcError) as error:
            run.preview()
        assert error.value.reason == "result_superseded"
        before = list(client.transport.sent)
        later = operation.snapshot()
        assert later is not None and later.preview_status == "reading"
        assert later.preview == captured.preview
        assert later.preview_images == captured.preview_images
        assert later.raw_save.path == "/actual/raw.h5"
        assert client.transport.sent == before
    assert not path.exists()


@pytest.mark.parametrize("png", ["invalid!", base64.b64encode(b"not-png").decode()])
def test_run_preview_decode_failure_retains_saved_raw_without_a_confirmed_image(
    tmp_path, png
):
    gui = LookbackGui()

    def respond(method, params):
        return {"png_b64": png} if method == "tab.get_figure" else gui(method, params)

    with recipe_client(tmp_path, respond) as client:
        operation = RecipeSession(client.context).open_tab("lookback").run()
        run, _ = operation.complete_in_current_worker()
        assert isinstance(run, RecipeRun)
        run.save_raw()
        with pytest.raises(ValueError, match="base64|PNG"):
            run.preview()
        captured = run.snapshot()
        assert captured.raw_save.path == "/actual/raw.h5"
        assert captured.preview_status == "reading"
        assert captured.preview is None and captured.preview_images == ()


def test_generator_preview_delivers_png_and_local_queries_do_not_repeat_the_read(
    tmp_path,
):
    gui = LookbackGui()

    def sample(session: RecipeSession) -> RecipeGenerator:
        run, status = yield session.open_tab("lookback").run()
        assert isinstance(run, RecipeRun) and status == "completed"
        run.save_raw()
        run.preview()

    def respond(method, params):
        if method == "tab.get_figure":
            return {"png_b64": base64.b64encode(PNG).decode()}
        return gui(method, params)

    client = make_client(tmp_path, respond, recipes=(definition(sample),))
    try:
        reply = client.call("sample", {})
        assert reply.data["status"] == "finished"
        assert reply.images[0].data == PNG
        assert reply.data["previews"]["run"]
        before = list(client.transport.sent)
        full = full_execution_reply(client, reply)
        assert full.data["preview"]["kind"] == "run_preview"
        assert Path(full.data["preview"]["path"]).read_bytes() == PNG
        assert full.data["raw_save"]["path"] == "/actual/raw.h5"
        waited = client.call(
            "wait", {"execution": reply.data["execution"], "timeout": 0}
        )
        assert waited.images == reply.images
        assert client.transport.sent == before
    finally:
        client.context.session.close()
