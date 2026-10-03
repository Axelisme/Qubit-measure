"""Tab closing delegates admission to the GUI."""

from ._support import make_client


def test_tab_close_is_one_explicit_command(tmp_path):
    def respond(method, params):
        assert method == "tab.close"
        assert params == {"tab_id": "t1", "discard_unsaved": False}
        return {"ok": True}

    client = make_client(tmp_path, respond)
    assert client.call("tab_close", {"tab": "t1"}) == {"closed": "t1"}
    methods = [name for name, _ in client.transport.sent]
    assert methods.count("tab.close") == 1
    assert "tab.snapshot" not in methods
