#!/usr/bin/env sh
set -eu

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <task-id> [goal]" >&2
  exit 2
fi

root=${PROJECT_ROOT:-$(pwd)}
plan_id=$1
shift || true
goal=${*:-"<填入任務目標。>"}
slug_re='^[A-Za-z0-9_][A-Za-z0-9._-]*$'

if ! printf '%s' "$plan_id" | grep -Eq "$slug_re"; then
  echo "invalid task id: $plan_id" >&2
  exit 2
fi

plan_dir="$root/.agent_state/plans/$plan_id"
date_stamp=$(date +%F)

mkdir -p "$plan_dir"

if [ -f "$plan_dir/task_plan.md" ]; then
  printf '%s\n' "$plan_id" > "$root/.agent_state/active_plan"
  echo "plan already exists: $plan_dir"
  echo "active plan: $plan_id"
  exit 0
fi

cat > "$plan_dir/task_plan.md" <<EOF
# $plan_id 任務計劃

**Last updated:** $date_stamp

## Goal

$goal

## Current State

- <目前已知狀態。>

## Architecture Baseline

- <相關模組、ADR、README 或設計約束。>

## Phase Status

| Phase | Status | Scope | Acceptance |
|---|---|---|---|
| Phase 1 | pending | <範圍> | <驗收條件> |

## Decisions

- <決策與理由。>

## Errors Encountered

| Error | Attempt | Resolution |
|---|---|---|

## Historical Phase Summary

| Phase | Topic | Conclusion / Commit |
|---|---|---|

## Active Notes

- <仍在詳細保留的 Phase note；超過規則時移到 archive.md。>
EOF

cat > "$plan_dir/findings.md" <<EOF
# $plan_id findings

**Last updated:** $date_stamp

## Discoveries

| Date | Area | Finding | Evidence |
|---|---|---|---|

## Design Notes

- <非顯而易見、未來 session 需要知道的設計事實。>

## Risks

- <風險與觸發條件。>

## Open Questions

- <需要用戶或後續研究回答的問題。>
EOF

cat > "$plan_dir/progress.md" <<EOF
# $plan_id progress

**Last updated:** $date_stamp

## Timeline

| Time | Actor | Action | Result | Next |
|---|---|---|---|---|
| $date_stamp | Codex | 建立 task plan | 三件套初始化 | 開始 Phase 1 |

## Verification Log

| Date | Command | Result |
|---|---|---|

## Handoff Notes

- <context compaction 或下一位 agent 需要知道的狀態。>
EOF

cat > "$plan_dir/archive.md" <<EOF
# $plan_id archive

**Last updated:** $date_stamp

## Archived Phase Details

EOF

printf '%s\n' "$plan_id" > "$root/.agent_state/active_plan"
echo "created plan: $plan_dir"
echo "active plan: $plan_id"
