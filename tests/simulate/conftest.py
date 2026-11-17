"""Mark numerical tests and summarize transported Engine accuracy observations."""

import pytest
from _pytest.terminal import TerminalReporter

from tests.simulate.evolution.contract.accuracy import (
    ALL_FAMILIES,
    accuracy_table,
    reported_measurements,
)


def pytest_itemcollected(item: pytest.Item) -> None:
    """Mark each item through its path-scoped hook before marker selection."""
    item.add_marker(pytest.mark.sim)


def pytest_terminal_summary(terminalreporter: TerminalReporter) -> None:
    """Summarize call reports, including numerical failures and strict xfails."""
    measurements = []
    for reports in terminalreporter.stats.values():
        for report in reports:
            if isinstance(report, pytest.TestReport) and report.when == "call":
                measurements.extend(reported_measurements(report.user_properties))
    if measurements:
        terminalreporter.section("Evolution density accuracy (limit 2e-3)")
        for line in accuracy_table(measurements, ALL_FAMILIES):
            terminalreporter.write_line(line)
