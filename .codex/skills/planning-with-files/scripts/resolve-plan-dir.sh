#!/usr/bin/env sh
set -eu

root=${PROJECT_ROOT:-$(pwd)}
plan_id=${1:-${PLAN_ID:-}}
slug_re='^[A-Za-z0-9_][A-Za-z0-9._-]*$'

is_slug() {
  printf '%s' "$1" | grep -Eq "$slug_re"
}

if [ -z "$plan_id" ] && [ -f "$root/.agent_state/active_plan" ]; then
  plan_id=$(tr -d '\r\n[:space:]' < "$root/.agent_state/active_plan")
fi

if [ -n "$plan_id" ]; then
  if ! is_slug "$plan_id"; then
    echo "invalid task id: $plan_id" >&2
    exit 2
  fi
  plan_dir="$root/.agent_state/plans/$plan_id"
  if [ -d "$plan_dir" ]; then
    printf '%s\n' "$plan_dir"
    exit 0
  fi
  echo "plan not found: $plan_dir" >&2
  exit 1
fi

newest=''
newest_mtime=0
if [ -d "$root/.agent_state/plans" ]; then
  for dir in "$root"/.agent_state/plans/*; do
    [ -d "$dir" ] || continue
    [ -f "$dir/task_plan.md" ] || continue
    mtime=$(stat -c '%Y' "$dir" 2>/dev/null || stat -f '%m' "$dir" 2>/dev/null || echo 0)
    if [ "$mtime" -gt "$newest_mtime" ] 2>/dev/null; then
      newest_mtime=$mtime
      newest=$dir
    fi
  done
fi

if [ -n "$newest" ]; then
  printf '%s\n' "$newest"
  exit 0
fi

echo "no active plan found under .agent_state/plans" >&2
exit 1
