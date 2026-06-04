#!/usr/bin/env bash
# Sync this skill (.claude is the source of truth) into the other agents' skill
# dirs (.agent, .codex). The three are independent files — NOT hard-linked (the
# editor's write-then-rename breaks links) — so run this after editing SKILL.md.
# .gemini has no skill copy, so it is left alone.
#
# Usage:  bash .claude/skills/run-dispersive-gui/sync_skills.sh
set -euo pipefail

SKILL_REL="skills/run-dispersive-gui"
FILES=(SKILL.md)
DESTS=(.agent .codex)

# Repo root = three levels up from this script (.claude/skills/run-dispersive-gui).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SRC_DIR="$REPO_ROOT/.claude/$SKILL_REL"

for f in "${FILES[@]}"; do
    [ -f "$SRC_DIR/$f" ] || { echo "FATAL: source missing: $SRC_DIR/$f" >&2; exit 1; }
done

for dst in "${DESTS[@]}"; do
    dst_dir="$REPO_ROOT/$dst/$SKILL_REL"
    mkdir -p "$dst_dir"
    for f in "${FILES[@]}"; do
        cp "$SRC_DIR/$f" "$dst_dir/$f"
    done
done

# Verify: every dest TRACKED file is byte-identical to the source.
for dst in "${DESTS[@]}"; do
    for f in "${FILES[@]}"; do
        if ! diff -q "$SRC_DIR/$f" "$REPO_ROOT/$dst/$SKILL_REL/$f" >/dev/null; then
            echo "FATAL: $dst/$SKILL_REL/$f differs from source after copy" >&2
            exit 1
        fi
    done
done

echo "synced run-dispersive-gui to: ${DESTS[*]}"
