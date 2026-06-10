#!/usr/bin/env bash
# Sync this skill (.claude is the source of truth) into the other agents' skill
# dirs (.agent, .codex). The three are independent files — NOT hard-linked (the
# editor's write-then-rename breaks links) — so run this after editing SKILL.md
# or smoke.py here. .gemini has no skill copy, so it is left alone.
#
# Usage:  bash .claude/skills/run-measure-gui/sync_skills.sh
set -euo pipefail

SKILL_REL="skills/run-measure-gui"
FILES=(SKILL.md smoke.py)
DESTS=(.agent .codex)

# Repo root = three levels up from this script (.claude/skills/run-measure-gui).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SRC_DIR="$REPO_ROOT/.claude/$SKILL_REL"

for f in "${FILES[@]}"; do
    [ -f "$SRC_DIR/$f" ] || { echo "FATAL: source missing: $SRC_DIR/$f" >&2; exit 1; }
done

for dst in "${DESTS[@]}"; do
    dst_dir="$REPO_ROOT/$dst/$SKILL_REL"
    [ -d "$dst_dir" ] || { echo "FATAL: dest dir missing: $dst_dir" >&2; exit 1; }
    for f in "${FILES[@]}"; do
        cp "$SRC_DIR/$f" "$dst_dir/$f"
    done
done

# Verify: every dest is byte-identical to the source (fail-fast if not).
for dst in "${DESTS[@]}"; do
    for f in "${FILES[@]}"; do
        if ! diff -q "$SRC_DIR/$f" "$REPO_ROOT/$dst/$SKILL_REL/$f" >/dev/null; then
            echo "FATAL: $dst/$SKILL_REL/$f differs from source after copy" >&2
            exit 1
        fi
    done
done

version="$(grep -m1 '^skill_version:' "$SRC_DIR/SKILL.md" || true)"
echo "Synced ${FILES[*]} from .claude -> ${DESTS[*]} (${version:-no skill_version})"
