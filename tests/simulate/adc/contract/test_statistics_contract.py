"""Run the reusable statistics promises against a test-only reference sampler."""

import pytest

from ._reference import reference_statistics
from ._statistics_contract import RoundStatisticsContract, StatisticsImplementation


class TestReferenceStatistics(RoundStatisticsContract):
    @pytest.fixture
    def round_statistics_impl(self) -> StatisticsImplementation:
        return reference_statistics
