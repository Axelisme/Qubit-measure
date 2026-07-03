"""Terminal markdown report tests."""

from __future__ import annotations

from zcu_tools.gui.app.autofluxdep.services.run_report import render_markdown_report


def test_run_report_contains_terminal_status_paths_and_counts():
    manifest = {
        "run_id": "run-1",
        "created_at": "2026-07-04T00:00:00Z",
        "project": {"chip_name": "chip", "qub_name": "q1"},
        "workflow": {
            "workflow_hash": "sha256:abc",
            "nodes": [{"name": "qubit_freq"}, {"name": "t1"}],
            "flux": {"values": [0.0, 0.5], "device_name": "fake_flux"},
        },
        "files": {
            "journal": "journal.jsonl",
            "nodes": [{"name": "qubit_freq", "path": "nodes/000-qubit_freq.hdf5"}],
        },
        "exports": {"fluxdep_spectrum": "exports/fluxdep/qubit_freq.hdf5"},
        "reports": {"markdown": "report.md"},
        "terminal": {
            "status": "failed",
            "finalized_at": "2026-07-04T00:10:00Z",
            "error": "boom",
        },
    }
    events = [
        {"type": "node_row_written", "node": "qubit_freq"},
        {"type": "node_skipped", "node": "t1"},
        {"type": "node_failed", "node": "t1"},
        {"type": "run_failed"},
        {"type": "run_finalized"},
    ]

    report = render_markdown_report(manifest, events)

    assert "# Autofluxdep Run run-1" in report
    assert "- Status: failed" in report
    assert "- Journal: journal.jsonl" in report
    assert "- Export fluxdep_spectrum: exports/fluxdep/qubit_freq.hdf5" in report
    assert "| qubit_freq | 1 | 0 | 0 |" in report
    assert "| t1 | 0 | 1 | 1 |" in report
    assert "- Run failed events: 1" in report
    assert "boom" in report
