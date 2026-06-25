import pytest

from tests.program.v2.support import ProgramTrace


def _make_mock_prog(pmem_size: int = 512) -> ProgramTrace:
    return ProgramTrace(pmem_size=pmem_size)


@pytest.fixture
def mock_prog() -> ProgramTrace:
    return _make_mock_prog(pmem_size=512)


@pytest.fixture
def large_pmem_prog() -> ProgramTrace:
    return _make_mock_prog(pmem_size=4096)
