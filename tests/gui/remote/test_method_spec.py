"""MCP policy construction rejects contradictory exposure declarations."""

import pytest
from zcu_tools.gui.remote.method_spec import McpExposure, McpMethodPolicy


@pytest.mark.parametrize("exposure", [McpExposure.GENERATED, McpExposure.INTERNAL])
def test_non_override_policy_rejects_override_names(exposure: McpExposure) -> None:
    with pytest.raises(ValueError, match="cannot declare override tools"):
        McpMethodPolicy(
            exposure=exposure, override_tool_names=("gui_bad",), reason="wire-only"
        )


def test_override_policy_requires_a_tool_name() -> None:
    with pytest.raises(ValueError, match="requires at least one tool name"):
        McpMethodPolicy.override(reason="manual shape")


def test_internal_policy_requires_a_reason() -> None:
    with pytest.raises(ValueError, match="requires a reason"):
        McpMethodPolicy.internal("")
