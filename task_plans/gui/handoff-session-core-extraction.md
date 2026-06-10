# Handoff — measure-gui ↔ autofluxdep session-core extraction

**For a fresh agent continuing GUI shared-layer work in the Qubit-measure repo.**
Repo: `/home/axel/Documents/VSCode/Python/Qubit-measure` · Branch: `gui2`.

## Mission
Implement the **session-core extraction** (S1–S5): lift measure-gui's measurement
session core (context system + SoC connection + multi-device + setup/device
dialogs) into a new shared `gui/session/` layer, then reshape the in-development
`autofluxdep` app to reuse it. Start at **S1** (lowest risk, pure relocation).

## READ FIRST (the authoritative plan — do not re-derive)
**`task_plans/gui/session_core_extraction.md`** — full plan: locked decisions, the
2026-06-09 boundary investigation findings, responsibility-corrections (P-a/P-b/P-c),
the `gui/session/` layout, and S1–S5 with scope/seam/risk. This is the meat; this
handoff only orients.

Then, before touching code:
- **`CLAUDE.md`** (repo root) — the binding project rules. Non-negotiable.
- **`<memory>/MEMORY.md`** + `project_gui_shared_layer_batch.md` — the #1–#6 shared-layer
  batch this extraction builds on. (memory dir:
  `~/.claude/projects/-home-axel-Documents-VSCode-Python-Qubit-measure/memory/`)
- **`task_plans/gui/task_plan.md`** Phase 133 步驟 B+ — the umbrella shared-layer plan.
- The relevant **`AI_NOTE.md`** before editing any module (they sit beside the code;
  e.g. `gui/app/main/AI_NOTE.md`, `gui/app/main/services/remote/AI_NOTE.md`,
  `gui/app/autofluxdep/CONTEXT.md`). All gitignored.

## Repo state at handoff
- Latest commit `79da2f8e` (flux_dep magnitude-only hardcode — separate, done).
- This session landed the #1–#6 shared-layer batch (commits `96f52a61` `5eabf5f4`
  `754efa69` `7ee123ea` `b26a129d` `ba5d6703`) + `79da2f8e`. User has pushed `gui2`.
- **Baseline is green**: `pyright lib/zcu_tools/gui/` = 0/0/0; tests gui **1034** /
  fluxdep **199** / dispersive **99**; ruff clean.
- ⚠️ Working tree has a stray **`.claude/settings.json`** modification that is NOT
  ours (the user's environment) — never stage/commit it.
- No in-flight code changes; the plan is written but **S1 not started**.

## Operating conventions / gotchas (learned this session — load-bearing)
- **Respond in 中文**; code/comments/identifiers in English (CLAUDE.md rule).
- Python: always `.venv/bin/python`. After any task: **pyright → pytest → ruff**
  (`ruff check --select I --fix <files>` THEN `ruff format <files>` — import-sort
  before format). **Commit only when the user asks.** Then update the touched
  `AI_NOTE.md` and bump its header `**Commit:** <hash>` to the new commit.
- **command-line `pyright` is authoritative** (binary `~/.local/bin/pyright`;
  `python -m pyright` is NOT installed). The harness's **inline diagnostics show
  mid-edit STALE ✘** during heavy agent edits — always confirm final state with
  command-line `pyright` + `grep`, never trust an inline ✘ at face value.
- Run test suites **separately** (`tests/gui`, `tests/fluxdep_gui`,
  `tests/dispersive_gui`) — a combined run Qt-segfaults (harness limitation, not a bug).
- **zsh does NOT word-split unquoted vars** — pass explicit file paths to
  ruff/pyright, not `$FILES`.
- **gitignored** (never in commits, don't force-add): all `AI_NOTE.md`,
  `task_plans/`, `docs/adr/`, `CONTEXT.md`, `CLAUDE.md`.
- **GUI scope** = `lib/zcu_tools/gui/` + `tests/{gui,fluxdep_gui,dispersive_gui}` +
  (approved) `experiment/v2_gui/adapters`. Get user approval before touching outside
  (e.g. `notebook/`, `experiment/v2/`). Note S1 moves types OUT of
  `gui/app/main/adapter/types.py` — still inside `gui/`, in scope.
- **Versions**: WIRE_VERSION (21) only bumps on an mcp-interpreted wire-contract
  change — this extraction is internal, WIRE stays 21. Bump GUI_VERSION for any
  observable change. Live MCP verification needs the user to **restart the MCP
  server** first (then `gui_launch`, check the `wire vN (mcp==gui)` banner).
- **Effective workflow this session**: per phase, spawn an **opus sub-agent** to do
  the mechanical edits + run gates, then the **parent independently re-verifies**
  (pyright/pytest/ruff + targeted greps) before committing. The user commits
  per-phase ("逐項 commit"). Worked well for #1–#6.

## Open items to re-confirm with the user before S1 (方向已認同, 細節未最終點頭)
The user said "這4件事基本同意" + delegated 2 calls to the agent; they then ran
`/handoff` ("我打算從乾淨的 context 中實作"). The plan's "鎖定決策" reflect the agreed
direction. A quick re-confirm of decisions 3/4/5 (OperationGate per-app + shared
Handles; ml-edit measure-only for now; dialogs under `gui/session/ui/`) is prudent
before committing to S1's structure, but they are already user-aligned.

## Suggested skills
- **run-measure-gui** — after S1–S3 (measure-gui must be behaviorally unchanged),
  live-smoke the MCP path to confirm: launch → connect(mock) → context → run
  fake/freq → analyze → save → device connect/setup → shutdown. The user restarts
  the MCP server; check the banner. (S1–S3 are "zero behavior change" — this proves it.)
- **code-review** — run on each phase's diff before/after committing (correctness +
  reuse/simplification).
- **grill-with-docs** — if the user wants to further stress-test the S1–S5 plan
  against the domain language / ADRs before diving in.

## First moves
1. Read `task_plans/gui/session_core_extraction.md` + `CLAUDE.md` + `MEMORY.md`.
2. Briefly re-confirm decisions 3/4/5 with the user.
3. Begin **S1**: relocate `ExpContext`/`SocHandle`/`SocCfgHandle`/`ContextReadiness`
   out of `gui/app/main/adapter/types.py` into `gui/session/types.py`; move the
   session-core event payloads + a `SessionEvent` enum into `gui/session/events.py`;
   move `OperationHandles` to the shared layer; redirect measure-gui's imports.
   Keep it behavior-neutral (pure relocation, like the #6 payload move). Verify
   pyright 0 + tests/gui green + ruff clean, then hand the diff to the user to commit.
