"""Keep behavior tests from opening process-global MCP call-log files."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from ._support import MeasureClient


@pytest.fixture(autouse=True)
def disable_call_log(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZCU_MCP_CALL_LOG", "0")


@pytest.fixture
def clients(tmp_path: Path) -> Iterator[list[MeasureClient]]:
    created: list[MeasureClient] = []
    yield created
    for client in created:
        client.context.session.close()
