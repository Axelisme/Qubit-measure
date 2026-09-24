"""Keep behavior tests from opening process-global MCP call-log files."""

import pytest


@pytest.fixture(autouse=True)
def disable_call_log(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZCU_MCP_CALL_LOG", "0")
