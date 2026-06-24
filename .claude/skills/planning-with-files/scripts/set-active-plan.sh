#!/usr/bin/env sh
set -eu

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <task-id>" >&2
  exit 2
fi

root=${PROJECT_ROOT:-$(pwd)}
plan_id=$1
slug_re='^[A-Za-z0-9_][A-Za-z0-9._-]*$'

if ! printf '%s' "$plan_id" | grep -Eq "$slug_re"; then
  echo "invalid task id: $plan_id" >&2
  exit 2
fi

plan_dir="$root/.agent_state/plans/$plan_id"
if [ ! -d "$plan_dir" ]; then
  echo "plan not found: $plan_dir" >&2
  exit 1
fi

mkdir -p "$root/.agent_state"
printf '%s\n' "$plan_id" > "$root/.agent_state/active_plan"
echo "active plan: $plan_id"
