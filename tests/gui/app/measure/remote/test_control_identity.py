"""Measure connects shared control authority to its real application owner."""

import socket

from ._helpers import recv_response, send


def test_measure_control_owner_projection_and_mixed_interact_permission(application):
    assert application.result("control.status", {})["holder"] == "rpc:agent"
    owner = application.controller.control_identity
    with socket.create_connection(("127.0.0.1", application.remote.port)) as reader:
        def rpc(method, **params):
            send(reader, {"id": "control", "method": method, "params": params})
            return recv_response(reader, "control")

        reply = rpc("control.connect", label="observer", mode="readonly")
        assert reply["ok"] and reply["result"]["me"] is False
        assert rpc("rpc.catalog")["ok"]
        assert rpc("tab.list_all")["ok"]
        denied = rpc("tab.interact", tab_id="never-created")
        assert not denied["ok"]
        assert denied["error"]["reason"] == "not_holder"
        owner.reclaim_local()
        projection = rpc("control.status")["result"]
        assert projection == {"holder": "gui", "locked": True, "online": True, "me": False}
        assert owner.status().holder == "gui"
        assert owner.status().locked is True
