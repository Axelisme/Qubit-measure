"""Result-only projection of the application Load outcome."""

from zcu_tools.gui.app.measure.services.load import LoadTabResultOutcome


def project_load_outcome(outcome: LoadTabResultOutcome) -> dict[str, object]:
    """Report committed result and analysis facts, not Form backfill details."""
    return {
        "tab_id": outcome.tab_id,
        "data_path": outcome.data_path,
        "result_type": outcome.result_type,
        "has_cfg_snapshot": outcome.has_cfg_snapshot,
        "has_analyze_params": outcome.has_analyze_params,
        "source_kind": outcome.source_kind,
        "analysis_error": outcome.analysis_error,
    }
