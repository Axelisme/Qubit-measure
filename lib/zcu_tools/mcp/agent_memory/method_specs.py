"""Qt-free wire-method contract table for the agent-memory MCP server — the single
source of truth for each tool's parameter schema, timeout and description.

``generate_tools`` builds one MCP tool per entry; ``server.build_dispatch`` binds a
``MemoryStore`` method to each. The two must stay in lockstep (a test asserts the
key sets match).

List-shaped params (``exp_type``, ``figure_paths``, checklist ``items``) use
``J.ARRAY`` — NOT ``J.JSON`` — so the MCP client sends a real array and never
char-splits a string into a list.
"""

from __future__ import annotations

# The wire-spec primitives belong to the shared remote layer; mcp consumes them from
# zcu_tools.gui.remote (no Qt is pulled in), it does not duplicate them under mcp/.
from zcu_tools.gui.remote.method_spec import MethodSpec
from zcu_tools.gui.remote.param_spec import JsonType as J
from zcu_tools.gui.remote.param_spec import ParamSpec as P

METHOD_SPECS = {
    # -- reads (pure queries) ------------------------------------------------
    "recall": MethodSpec(
        5.0,
        "Open the notebook to a qubit/experiment. Returns three buckets: "
        "'checklist' (acceptance items for exp_type), 'gotchas' (solutions for "
        "exp_type), 'recent' (recent records for this chip/qub). Call BEFORE an "
        "experiment, and re-read the checklist at the acceptance gate.",
        params=(
            P("chip", J.STRING, required=False, description="chip name, e.g. Q5_2D"),
            P("qub", J.STRING, required=False, description="qubit name, e.g. Q1"),
            P(
                "exp_type",
                J.STRING,
                required=False,
                description="experiment type, e.g. 'reset/bath'",
            ),
            P(
                "limit",
                J.INTEGER,
                required=False,
                default=10,
                description="max items per bucket (default 10)",
            ),
        ),
    ),
    "search": MethodSpec(
        5.0,
        "Keyword search over symptom + body (mainly hits troubleshooting "
        "solutions). Use it to find a prior fix for a symptom, and to check for "
        "duplicates before adding one. Each result carries a 'score' (total "
        "query-term hit count over symptom+body+exp_type+reason); results are "
        "ranked highest-score first.",
        params=(
            P("query", J.STRING, description="symptom / keywords"),
            P(
                "kind",
                J.STRING,
                required=False,
                description="filter by entry kind: 'record' or 'solution'",
            ),
            P("exp_type", J.STRING, required=False),
            P(
                "category",
                J.STRING,
                required=False,
                description="solution category, e.g. failure-fix / analysis-heuristic",
            ),
            P(
                "limit",
                J.INTEGER,
                required=False,
                default=10,
                description="max results (default 10)",
            ),
        ),
    ),
    "get": MethodSpec(
        5.0,
        "Read one entry in full (frontmatter + body) by its id.",
        params=(P("entry_id", J.STRING, description="namespace-relative id, no .md"),),
    ),
    "checklist_get": MethodSpec(
        5.0,
        "Get the acceptance checklist for an experiment type as a list of items "
        "(parsed from the file's markdown bullet body). Empty list if none exists.",
        params=(P("exp_type", J.STRING, description="experiment type or 'general'"),),
    ),
    # -- writes --------------------------------------------------------------
    "record_measurement": MethodSpec(
        10.0,
        "Record one measurement as a FOLDER (record.md + copied figures). Records "
        "are immutable: a same-day same-experiment collision auto-suffixes rather "
        "than overwriting. On disk, record.md frontmatter carries "
        "type/chip/qub/date/exp_type/decision/figures; its body opens with the "
        "one-line reason verdict, then the per-item pass/fail with evidence and "
        "numbers (reason lives in the body, not a frontmatter field).",
        params=(
            P("chip", J.STRING),
            P("qub", J.STRING),
            P("date", J.STRING, description="ISO date, e.g. 2026-06-08"),
            P(
                "exp_type",
                J.ARRAY,
                description="list of experiment types, e.g. ['reset/bath']",
            ),
            P("decision", J.STRING, description="accept | reject"),
            P("reason", J.STRING, description="one-line accept/reject verdict"),
            P(
                "body",
                J.STRING,
                description="per-item pass/fail + evidence + the numbers",
            ),
            P(
                "figure_paths",
                J.ARRAY,
                required=False,
                description="absolute paths to figures to copy into the record folder",
            ),
            P(
                "data_ref",
                J.STRING,
                required=False,
                description="pointer to the saved data",
            ),
        ),
    ),
    "checklist_set": MethodSpec(
        10.0,
        "Replace the acceptance checklist for an experiment type wholesale (NOT "
        "append — pass the full list; checklist_get first if extending). Items "
        "are stored as a hand-editable markdown bullet list.",
        params=(
            P("exp_type", J.STRING, description="experiment type or 'general'"),
            P("items", J.ARRAY, description="full list of checklist item strings"),
        ),
    ),
    "add_solution": MethodSpec(
        10.0,
        "Add a context-free, reusable troubleshooting solution (starts "
        "'provisional'). Search first; if one already exists, update it instead. "
        "The body must be self-contained — no source-file references; cfg knob "
        "names are fine.",
        params=(
            P("exp_type", J.STRING, description="experiment type or 'general'"),
            P("symptom", J.STRING, description="the searchable problem statement"),
            P(
                "category",
                J.STRING,
                description="failure-fix | analysis-heuristic | gotcha | workflow-pref",
            ),
            P("body", J.STRING, description="現象 / 原因 / 怎麼做 — the reusable rule"),
            P(
                "seen_in",
                J.ARRAY,
                required=False,
                description="list of record ids evidencing it",
            ),
        ),
    ),
    "update_solution": MethodSpec(
        10.0,
        "Refine an existing solution: replace body/symptom, extend seen_in (which "
        "promotes to 'confirmed' at >=2 records), or set confidence explicitly.",
        params=(
            P("entry_id", J.STRING),
            P("body", J.STRING, required=False),
            P(
                "confidence",
                J.STRING,
                required=False,
                description="provisional | confirmed",
            ),
            P(
                "add_seen_in",
                J.ARRAY,
                required=False,
                description="record ids to append to seen_in",
            ),
            P("symptom", J.STRING, required=False),
        ),
    ),
    "delete": MethodSpec(
        10.0,
        "Delete a troubleshooting solution or a checklist (a record cannot be "
        "deleted — records are immutable history).",
        params=(P("entry_id", J.STRING),),
    ),
}
