from unittest.mock import MagicMock

import pytest


def _make_mock_prog(pmem_size: int = 512) -> MagicMock:
    prog = MagicMock()
    prog._get_reg.side_effect = lambda name: name
    prog.tproccfg = {"pmem_size": pmem_size}
    return prog


@pytest.fixture
def mock_prog():
    return _make_mock_prog(pmem_size=512)


@pytest.fixture
def large_pmem_prog():
    return _make_mock_prog(pmem_size=4096)
