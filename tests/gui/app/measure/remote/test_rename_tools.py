"""Native missing library rename retains the complete published capture."""

from __future__ import annotations

import pytest
from zcu_tools.cfg_model import ConstWaveformCfg
from zcu_tools.gui.session.context_types import LibraryChange

from tests.gui._context_reads import capture_focus
from tests.gui.app.measure._record_support import require_target

from ._helpers import Fixture, call, open_client


def test_native_missing_rename_preserves_the_published_library(tmp_path, qapp):
    fx = Fixture(project_root=str(tmp_path / "project"))
    context = fx.ctrl.context_control
    context.set_waveform(
        require_target(capture_focus(context)),
        "Q1.waveforms.seed",
        ConstWaveformCfg(length=0.1),
    )
    fx.start()
    sock = open_client(fx.service.port)
    try:
        before = call(sock, "context.snapshot")
        assert before["ok"] is True, before
        parameters = capture_focus(fx.ctrl.context_control)
        library = parameters.library.document()
        version = fx.state.version.get("context")
        with pytest.raises(KeyError, match="a"):
            fx.ctrl.context_control.apply_library_change(
                parameters,
                LibraryChange(
                    "module", "rename", "Q1.pulses.a", new_path="Q1.pulses.b"
                ),
            )
        assert capture_focus(fx.ctrl.context_control).library.document() == library
        assert fx.state.version.get("context") == version
        assert call(sock, "context.snapshot")["result"] == before["result"]
    finally:
        sock.close()
        fx.stop()
