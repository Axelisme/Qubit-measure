#!/usr/bin/env sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
plan_dir=$("$script_dir/resolve-plan-dir.sh" "${1:-}")

missing=0
for file in task_plan.md findings.md progress.md archive.md; do
  if [ ! -f "$plan_dir/$file" ]; then
    echo "missing: $plan_dir/$file" >&2
    missing=1
  fi
done

if [ "$missing" -ne 0 ]; then
  exit 1
fi

if grep -En '\|[^|]+\|[[:space:]]*(pending|in_progress|blocked)[[:space:]]*\|' "$plan_dir/task_plan.md"; then
  echo "plan has unfinished phases: $plan_dir/task_plan.md" >&2
  exit 1
fi

echo "planning files look complete: $plan_dir"
