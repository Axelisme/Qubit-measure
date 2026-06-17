"""Wire-method contract table for the taskboard MCP server.

``generate_tools`` builds one MCP tool per entry; ``server.build_dispatch`` binds a
``TaskboardStore`` method to each.  The two must stay in lockstep (a test asserts
the key sets match).
"""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import MethodSpec
from zcu_tools.gui.remote.param_spec import JsonType as J
from zcu_tools.gui.remote.param_spec import ParamSpec as P

_PATHS_DOC = (
    "List of paths to claim.  Each entry may be: "
    "(1) a repo-relative file or directory path (e.g. 'lib/zcu_tools/mcp/'); "
    "(2) a glob pattern (e.g. 'lib/zcu_tools/gui/**/*.py'); "
    "(3) an @-prefixed resource token for non-file singletons "
    "(e.g. '@hw/zcu216', '@gui/measure', '@port/8767').  "
    "Conflict is detected by exact match, ancestor-directory prefix, or glob overlap."
)

METHOD_SPECS: dict[str, MethodSpec] = {
    # -- mutating tools ---------------------------------------------------
    "claim": MethodSpec(
        10.0,
        "Claim a set of paths/resources before writing them.  Returns "
        "{status: 'granted'|'pending', claim_id, conflicts}.  "
        "If status is 'pending' the claim is queued behind conflicting grants; "
        "use taskboard_wait(claim_id) or schedule a ScheduleWakeup to poll "
        "taskboard_list until your claim appears in 'active'.  "
        "Only claims from a DIFFERENT Claude Code session contend: an orchestrator "
        "and the sub-agents it spawns share one session, so their claims never "
        "block each other, and re-claiming an already-held scope returns the same "
        "claim_id (no duplicate).  "
        "Always release (taskboard_release) AFTER your changes are committed.",
        params=(
            P(
                "owner",
                J.STRING,
                description=(
                    "human-readable label shown on the board, e.g. 'impl-162a'.  "
                    "NOT the conflict identity — the server derives that from the "
                    "Claude Code session, so claims from one session (an "
                    "orchestrator + its sub-agents) never block each other and a "
                    "re-claim of an already-held scope is ignored."
                ),
            ),
            P("paths", J.ARRAY, description=_PATHS_DOC),
            P(
                "task",
                J.STRING,
                description="short description of what this claim covers",
            ),
            P(
                "mode",
                J.STRING,
                required=False,
                default="write",
                description="'read' or 'write' (default: 'write'); read+read claims never conflict",
            ),
        ),
    ),
    "release": MethodSpec(
        10.0,
        "Release a claim and auto-promote queued pending claims whose blockers "
        "are gone.  Returns {released_id, promoted:[{claim_id, owner, paths}]}.  "
        "Only call AFTER your changes are committed to git.",
        params=(
            P(
                "claim_id",
                J.STRING,
                description="the claim_id returned by taskboard_claim",
            ),
        ),
    ),
    "touch": MethodSpec(
        5.0,
        "Heartbeat — update the claim's 'touched' timestamp to prevent TTL-based "
        "stale reclaim.  Call periodically for long-running operations.  "
        "Returns {claim_id, touched}.",
        params=(P("claim_id", J.STRING),),
    ),
    "force_release": MethodSpec(
        10.0,
        "Forcibly release any claim, including stale ones not yet TTL-reclaimed.  "
        "Use for manual recovery when an agent crashed without releasing.  "
        "Returns {released_id, promoted:[...]}.",
        params=(P("claim_id", J.STRING),),
    ),
    # -- read-only tools --------------------------------------------------
    "check": MethodSpec(
        5.0,
        "Dry-run conflict check — zero side effects, shared lock.  "
        "Returns {conflicts:[{owner, paths, mode}]}.  "
        "Use before taskboard_claim to decide whether to proceed or wait.",
        params=(
            P("paths", J.ARRAY, description=_PATHS_DOC),
            P(
                "mode",
                J.STRING,
                required=False,
                default="write",
                description="'read' or 'write' (default: 'write')",
            ),
        ),
    ),
    "list": MethodSpec(
        5.0,
        "List active (granted) claims, the pending queue, and recent released claims.  "
        "Zero side effects, shared lock.  "
        "Returns {active:[...], pending:[...], recent_released:[...]}.",
        params=(),
    ),
    "wait": MethodSpec(
        35.0,  # > MAX_WAIT_TIMEOUT=30 so the server-side loop always owns the timeout
        "Block-poll until the pending claim becomes granted or timeout_s elapses "
        "(server-side poll every 0.5 s; capped at 30 s regardless of timeout_s).  "
        "Returns {status: 'granted'|'timeout'|'released'}.  "
        "For longer waits use a ScheduleWakeup and poll taskboard_list instead.",
        params=(
            P("claim_id", J.STRING),
            P(
                "timeout_s",
                J.NUMBER,
                required=False,
                default=5.0,
                description="poll duration in seconds (max 30)",
            ),
        ),
    ),
}
