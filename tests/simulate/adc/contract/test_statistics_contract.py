"""Run the reusable statistics promises against a test-only reference sampler."""

import pytest
from zcu_tools.simulate.adc import round_statistics

from ._reference import reference_statistics
from ._statistics_contract import RoundStatisticsContract, StatisticsImplementation


class TestReferenceStatistics(RoundStatisticsContract):
    @pytest.fixture
    def round_statistics_impl(self) -> StatisticsImplementation:
        return reference_statistics


class TestBatchedStatistics(RoundStatisticsContract):
    @pytest.fixture
    def round_statistics_impl(self) -> StatisticsImplementation:
        return round_statistics
