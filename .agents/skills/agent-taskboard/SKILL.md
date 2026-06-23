---
name: agent-taskboard
description: Multi-agent path-coordination protocol using the taskboard MCP server. Use when starting work that edits files or holds a singleton resource shared with other agents.
skill_version: 2
---

# agent-taskboard

The taskboard MCP server (`lib/zcu_tools/mcp/taskboard/`) provides atomic, file-backed coordination for parallel agents on the same checkout.  Seven `taskboard_*` tools implement read/write path locking, pending queues, TTL heartbeats, and a human-readable markdown view auto-rendered after each mutation.

Spec: ADR-0022.  JSON store: `task_plans/taskboard.json` (gitignored).  Human view: `task_plans/taskboard.md` (auto-generated, do not edit).

---

## When to claim

Claim before any `Edit` or `Write` operation whose scope could overlap with **another Claude Code session**.  Pure reads, queries, and isolated one-file fixes that have no plausible contention do not require a claim.

---

## Session identity & same-session claims

The conflict identity is **the Claude Code session**, not the `owner` string you pass.  The server reads it from `CLAUDE_CODE_SESSION_ID`, which is the *same* value for a top-level session and every sub-agent it spawns, and *different* across top-level sessions.

Consequences:

- **An orchestrator and the sub-agents it launches share one session, so their claims never block each other.**  You do **not** need to tell a sub-agent to skip claiming or to claim under the orchestrator — let each agent claim normally; same-session overlap is auto-granted.
- **Re-claiming a scope you already hold is ignored** — it returns the same `claim_id`, still granted, and adds no duplicate.  A held `write` covers a re-claimed `read`/`write` of the same or a narrower path; a held `read` covers a re-claimed `read`.
- **Only a *different* session contends.**  Cross-session overlapping writes still queue as `pending`, which is exactly what coordination is for.
- `owner` is just a human-readable label on the board; pick something descriptive (e.g. `'impl-162a'`).  It has no effect on conflict resolution unless `CLAUDE_CODE_SESSION_ID` is unset, in which case the server falls back to `owner` as the identity.

---

## Standard workflow

```
1.  taskboard_check(paths, mode='write')        # dry-run — zero side effects
2.  taskboard_claim(owner, paths, task)         # reserve scope
    → status='granted': proceed with edits
    → status='pending': wait (see below)
3.  ... edits ... git commit ...
4.  taskboard_release(claim_id)                 # AFTER commit only
```

**Always release AFTER the relevant changes are committed to git.**  An unreleased claim means "uncommitted work exists here" — releasing early risks another agent's commit overtaking yours.

---

## Handling `pending` status

When `taskboard_claim` returns `status='pending'` your claim is queued behind the conflicting grants listed in `conflicts`.

- **Short wait (≤30 s)**: `taskboard_wait(claim_id, timeout_s=N)` blocks server-side; returns `{status: 'granted'|'timeout'}`.
- **Longer wait**: use a `ScheduleWakeup` to return later and poll `taskboard_list` until your `claim_id` appears in `active`.  Do not spin — one probe per wakeup is enough.

---

## Heartbeat

For operations expected to take more than a few minutes call `taskboard_touch(claim_id)` periodically.  The default TTL is 2 hours; stale claims are auto-reclaimed on the next mutating call.

---

## Path syntax

| Form | Example | Notes |
|---|---|---|
| Repo-relative file | `lib/zcu_tools/mcp/taskboard/store.py` | exact |
| Repo-relative directory | `lib/zcu_tools/gui/` | any descendant overlaps |
| Glob | `tests/gui/**/*.py` | `*` and `**` supported |
| Resource token | `@hw/zcu216`, `@gui/measure`, `@port/8767` | non-file singletons; same overlap rules |

Read claims (`mode='read'`) never conflict with each other — only a write claim conflicts with any overlapping read or write.

---

## Reference tool signatures

| Tool | Key params | Returns |
|---|---|---|
| `taskboard_check` | `paths`, `mode='write'` | `{conflicts:[{owner,paths,mode}]}` |
| `taskboard_claim` | `owner`, `paths`, `task`, `mode='write'` | `{status, claim_id, conflicts}` |
| `taskboard_release` | `claim_id` | `{released_id, promoted:[...]}` |
| `taskboard_wait` | `claim_id`, `timeout_s=5` | `{status}` |
| `taskboard_touch` | `claim_id` | `{claim_id, touched}` |
| `taskboard_force_release` | `claim_id` | `{released_id, promoted:[...]}` |
| `taskboard_list` | — | `{active, pending, recent_released}` |
