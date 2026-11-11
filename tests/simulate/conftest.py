"""Mark numerical tests and summarize transported Engine accuracy observations."""

from pathlib import Path

import pytest
from _pytest.terminal import TerminalReporter

from tests.simulate.evolution.contract.accuracy import (
    ALL_FAMILIES,
    accuracy_table,
    reported_measurements,
)


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark this subtree before pytest applies command-line marker selection."""
    simulation_tests = Path(__file__).parent
    for item in items:
        if item.path.is_relative_to(simulation_tests):
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
