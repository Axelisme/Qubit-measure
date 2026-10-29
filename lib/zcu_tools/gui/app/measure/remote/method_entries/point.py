"""Entry/point RPC contracts; the session owner controls selection."""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import MethodSpec
from zcu_tools.gui.remote.param_spec import JsonType, ParamSpec

from ._params import optional_string, required_string
from ._registry import RemoteMethodEntry, method_entry

METHODS: tuple[RemoteMethodEntry, ...] = (
    method_entry(
        "entry.list",
        "point:h_entry_list",
        MethodSpec(
            5.0,
            "Refresh and return {entries: [name, ...]} from the result entry owner. "
            "Does not open an entry or reveal context content. Read errors propagate.",
            readonly=True,
        ),
    ),
    method_entry(
        "entry.info",
        "point:h_entry_info",
        MethodSpec(
            5.0,
            "Return {entry, files}: entry is null before opening, otherwise "
            "name/result_path/database_path from the cached owner identity. "
            "files lists installed point {document, path, revision}, ordered "
            "parameters/library/aliases; empty before point selection. Paths are "
            "absolute and revisions are raw-byte SHA-256 (null for missing library or aliases). "
            "No disk read or implicit refresh.",
            readonly=True,
        ),
    ),
    method_entry(
        "entry.new",
        "point:h_entry_new",
        MethodSpec(
            10.0,
            "Create a named entry without selecting a point "
            "or touching hardware. Return name/result_path/database_path. "
            "Name is one arbitrary resource-validated segment, not chip/qubit. "
            "Existing or invalid names fail without changing the binding. "
            "Installation failure retains the created entry; recover with entry.open, "
            "not another New. No retry or rollback.",
            (required_string("name", "New single-segment result entry name"),),
        ),
    ),
    method_entry(
        "entry.open",
        "point:h_entry_open",
        MethodSpec(
            10.0,
            "Open an existing named entry without selecting "
            "a point or touching hardware. Return name/result_path/"
            "database_path. Invalid/missing entry or installation failure preserves "
            "the old binding; never creates an entry or retries.",
            (required_string("name", "Existing single-segment result entry name"),),
        ),
    ),
    method_entry(
        "point.list",
        "point:h_point_list",
        MethodSpec(
            5.0,
            "Return {entry, points}: entry is null or name/result_path/"
            "database_path; points contains label/description/available/error. "
            "Refresh parameter availability independently per point without reading "
            "library/aliases. Point errors stay in their row; discovery failures "
            "propagate. Does not reveal context content.",
            readonly=True,
        ),
    ),
    method_entry(
        "point.new",
        "point:h_point_new",
        MethodSpec(
            10.0,
            "Create and select label in the currently opened entry; no_entry if none. "
            "mode defaults to inherit (copy global parameters/library/aliases); empty creates "
            "minimal parameters and no library/aliases; clone requires a same-entry clone_from. "
            "Other modes reject clone_from. Never uses unsaved session parameters. "
            "description defaults to empty and replaces seed/clone description. "
            "Returns {entry_name, point}. Binding failure retains "
            "the created point and old selection: point_binding_failed with "
            "{entry_name, point, created: true}; recover with point.use, not New.",
            (
                required_string("label", "New single-segment point label"),
                ParamSpec(
                    "description",
                    JsonType.STRING,
                    required=False,
                    default="",
                    description="Replacement description, including empty text",
                ),
                ParamSpec(
                    "mode",
                    JsonType.STRING,
                    required=False,
                    default="inherit",
                    enum=("inherit", "empty", "clone"),
                    description="inherit Global (default), empty, or clone another point",
                ),
                optional_string(
                    "clone_from", "Same-entry source label, required only in clone mode"
                ),
            ),
        ),
    ),
    method_entry(
        "point.use",
        "point:h_point_use",
        MethodSpec(
            5.0,
            "Select required label in the currently opened entry, or null for Global; "
            "omitting label is invalid_params; no_entry if no opened entry. Return "
            "{entry_name, point}, with point null for Global. Failure preserves the old "
            "binding. Existing tabs retain their capture; new tabs use this selection.",
            (
                ParamSpec(
                    "label",
                    JsonType.STRING,
                    nullable=True,
                    description="Existing single-segment point label, or null for Global",
                ),
            ),
        ),
    ),
)
