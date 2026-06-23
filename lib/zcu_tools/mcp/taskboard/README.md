# AI Note for `lib/zcu_tools/mcp/taskboard/`

**Last updated:** 2026-06-23 — session identity fallback

Taskboard is the MCP-backed coordination layer for parallel agents sharing one checkout. It stores claims in `task_plans/taskboard.json`, renders the human view at `task_plans/taskboard.md`, and uses file locks so claim/check/release operations are atomic.

The conflict key is a top-level agent session identity, not the user-facing `owner` label. The server reads `CLAUDE_CODE_SESSION_ID`, `CODEX_THREAD_ID`, or `AGENT_SESSION_ID`; if the MCP subprocess env lacks them, Linux builds fall back to reading only those allowlisted names from ancestor `/proc/*/environ` entries. If no session identity is available, coordination degrades to the older per-owner behavior.

Claim conflict rules live in `store.py`: overlapping paths conflict only when at least one side is `write` and the identities differ. Same-identity overlaps are granted and returned as warnings; re-claiming a scope already covered by the same identity is idempotent and returns the existing claim.

Path matching supports repo-relative files, directory prefixes, glob patterns, and `@` resource tokens for singleton live resources such as boards, GUIs, and ports.
