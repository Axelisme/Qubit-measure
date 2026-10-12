"""Window RPC call types and explicit synthetic Start batches."""

from collections.abc import Callable

from zcu_tools.gui.app.workflows.remote import FormWrite

RpcRequest = Callable[[str, dict[str, object]], dict[str, object]]


def start_plan_batch() -> list[FormWrite]:
    """Supply a branch and expression, leaving points to fresh Start defaults."""
    return [
        {"path": ["samples", "kind"], "value": "list"},
        {"path": ["samples", "values"], "value": [1.0, 1.1]},
        {"path": ["scale"], "value": {"expr": "0.06 * 2"}},
    ]
