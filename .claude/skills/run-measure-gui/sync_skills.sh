#!/usr/bin/env bash
# Sync the skill directory that contains this script into the matching skill
# directory under .agents, .codex, and .claude. The three copies are independent
# files, not hard-linked, so run this after editing a skill copy.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SKILL_NAME="$(basename "$SCRIPT_DIR")"
AGENT_DIRS=(.agents .codex .claude)

for agent_dir in "${AGENT_DIRS[@]}"; do
    dst_dir="$REPO_ROOT/$agent_dir/skills/$SKILL_NAME"
    mkdir -p "$dst_dir"
    if [ "$dst_dir" = "$SCRIPT_DIR" ]; then
        continue
    fi
    cp -a "$SCRIPT_DIR"/. "$dst_dir"/
done

for agent_dir in "${AGENT_DIRS[@]}"; do
    dst_dir="$REPO_ROOT/$agent_dir/skills/$SKILL_NAME"
    if ! diff -qr "$SCRIPT_DIR" "$dst_dir" >/dev/null; then
        echo "FATAL: $agent_dir/skills/$SKILL_NAME differs from source after copy" >&2
        exit 1
    fi
done

echo "synced $SKILL_NAME to: ${AGENT_DIRS[*]}"
