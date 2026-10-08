"""Entry/point RPC contracts; the session owner controls selection."""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import MethodSpec
from zcu_tools.gui.remote.param_spec import JsonType, ParamSpec

from ._params import optional_string, required_string
from ._registry import AgentMethodPolicy, RemoteMethodEntry, method_entry

METHODS: tuple[RemoteMethodEntry, ...] = (
    method_entry(
        "point.list",
        "point:h_point_list",
        MethodSpec(
            5.0,
            "Return {entry, points}: entry is null or entry_id/name/result_path/"
            "database_path; points contains label/description. Refresh persisted "
            "metadata; read failures propagate. Does not reveal context content.",
        ),
    ),
    method_entry(
        "point.new",
        "point:h_point_new",
        MethodSpec(
            10.0,
            "Create and select label in the opened entry UUID. Copies setup when "
            "clone_from is omitted, otherwise a same-entry point; never DRAFT. "
            "description defaults to empty and replaces seed/clone description. "
            "Returns {kind: 'saved_point', entry_id, point}. Binding failure retains "
            "the created point and old selection: point_binding_failed with "
            "{entry_id, point, created: true}; recover with point.use, not New.",
            (
                required_string("entry_id", "UUID obtained from point.list"),
                required_string("label", "New single-segment point label"),
                ParamSpec(
                    "description",
                    JsonType.STRING,
                    required=False,
                    default="",
                    description="Replacement description, including empty text",
                ),
                optional_string(
                    "clone_from", "Same-entry source label; omit for setup"
                ),
            ),
        ),
        agent=AgentMethodPolicy(refresh_after_write=True),
    ),
    method_entry(
        "point.use",
        "point:h_point_use",
        MethodSpec(
            5.0,
            "Select label in the opened entry UUID and return "
            "{kind: 'saved_point', entry_id, point}. Failure preserves the old "
            "binding. Existing tabs retain their capture; new tabs use this point.",
            (
                required_string("entry_id", "UUID obtained from point.list"),
                required_string("label", "Existing single-segment point label"),
            ),
        ),
        agent=AgentMethodPolicy(refresh_after_write=True),
    ),
)
