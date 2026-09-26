"""Hardware gate presence is projected through the registered wire handler."""

from zcu_tools.gui.session.events import GatePresence

from ._helpers import dispatch_handler


class GateController:
    def get_hardware_gate_presence(self) -> tuple[GatePresence, ...]:
        return (
            GatePresence(
                kind="run",
                origin_kind="agent",
                note="run T1 (tab t1)",
                active_for_seconds=1.25,
            ),
        )


def test_hardware_gate_rpc_projects_presence_without_monotonic_epoch() -> None:
    assert dispatch_handler(GateController(), "state.hardware_gate", {}) == {
        "active": [
            {
                "kind": "run",
                "origin_kind": "agent",
                "note": "run T1 (tab t1)",
                "active_for_seconds": 1.25,
            }
        ]
    }
