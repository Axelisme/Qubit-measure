"""Complete active-context projection over the GUI control socket."""

from dataclasses import replace

import numpy as np
import pytest
from qick.asm_v2 import QickParam
from zcu_tools.meta_tool import MetaDict, ModuleLibrary
from zcu_tools.program.v2 import ModuleCfgFactory, WaveformCfgFactory

from ._helpers import Fixture, call, open_client, reset_inbox

pytestmark = pytest.mark.uses_wall_clock


@pytest.fixture()
def fx(qapp):
    fixture = Fixture(active_label="ctx001")
    fixture.start()
    yield fixture
    fixture.stop()


def test_context_snapshot_returns_full_active_md_and_ml_over_the_socket(fx):
    md = MetaDict()
    md.update(
        {"r_f": 6000.0, "peaks": [1, 2], "phase": 1 + 2j, "trace": np.array([3.0, 4.0])}
    )
    ml = ModuleLibrary()
    ml.modules["drive"] = ModuleCfgFactory.from_raw(
        {
            "type": "pulse",
            "ch": 0,
            "nqz": 1,
            "freq": 6000.0,
            "gain": 0.25,
            "phase": 0.0,
            "pre_delay": 0.0,
            "post_delay": 0.0,
            "waveform": {"style": "const", "length": 0.1},
        }
    )
    ml.waveforms["square"] = WaveformCfgFactory.from_raw(
        {"style": "const", "length": 0.1}
    )
    fx.state.set_context(replace(fx.state.exp_context, md=md, ml=ml))
    sock = open_client(fx.service.port)
    try:
        reset_inbox(sock)
        reply = call(sock, "context.snapshot", rid="full")
        assert reply["ok"] is True, reply
        assert reply["result"] == {
            "label": "ctx001",
            "md": {
                "peaks": [1, 2],
                "phase": {"__complex__": [1.0, 2.0]},
                "r_f": 6000.0,
                "trace": [3.0, 4.0],
            },
            "ml": {
                "modules": {"drive": ml.modules["drive"].to_dict()},
                "waveforms": {"square": ml.waveforms["square"].to_dict()},
            },
        }
    finally:
        reset_inbox(sock)
        sock.close()


@pytest.mark.parametrize(
    "value",
    [{"1": "first", 1: "second"}, QickParam(start=1.0, spans={"pulse": 0.1})],
    ids=["nested-key-collision", "qick-param"],
)
def test_context_snapshot_rejects_lossy_md_projection_over_socket(fx, value):
    md = MetaDict()
    md.update({"nested": value})
    fx.state.set_context(replace(fx.state.exp_context, md=md, ml=ModuleLibrary()))
    sock = open_client(fx.service.port)
    try:
        reset_inbox(sock)
        reply = call(sock, "context.snapshot", rid="lossy")
        assert reply["ok"] is False, reply
        assert reply["error"]["code"] == "precondition_failed"
        assert reply["error"]["reason"] == "unserializable_context"
    finally:
        reset_inbox(sock)
        sock.close()


def test_context_snapshot_rejects_opaque_values_instead_of_claiming_full_read(fx):
    md = MetaDict()
    md.update({"opaque": object()})
    fx.state.set_context(replace(fx.state.exp_context, md=md, ml=ModuleLibrary()))
    sock = open_client(fx.service.port)
    try:
        reset_inbox(sock)
        reply = call(sock, "context.snapshot", rid="full")
        assert reply["ok"] is False
        assert reply["error"]["code"] == "precondition_failed"
        assert reply["error"]["reason"] == "unserializable_context"
    finally:
        reset_inbox(sock)
        sock.close()
