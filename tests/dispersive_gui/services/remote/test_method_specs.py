"""Tests for dispersive method_specs + the MCP tool-name derivation."""

from __future__ import annotations

from zcu_tools.gui.app.dispersive.services.remote.method_specs import METHOD_SPECS


def test_all_specs_are_read_only_no_params():
    # The read-only surface takes no parameters (pure queries).
    for spec in METHOD_SPECS.values():
        assert spec.params == ()
        assert spec.off_main_thread is False
        assert spec.timeout_seconds > 0


def test_expected_method_set():
    assert set(METHOD_SPECS) == {
        "project.info",
        "fit_inputs.info",
        "preprocess.status",
        "fit.result",
        "resources.versions",
        "state.check",
    }


def test_tool_name_derivation():
    # The mcp_server derives ``dispersive_<method_with_dots_to_underscores>``.
    def derived(method: str) -> str:
        spec = METHOD_SPECS[method]
        return spec.tool_name or "dispersive_" + method.replace(".", "_")

    assert derived("state.check") == "dispersive_state_check"
    assert derived("fit_inputs.info") == "dispersive_fit_inputs_info"
    assert derived("fit.result") == "dispersive_fit_result"
