"""Tools-owned benchmark/scene locations and cross-index selection policy."""

import subprocess
from collections.abc import Sequence
from pathlib import Path

from zcu_tools.simulate.evolution import (
    SCENE_CACHE_VERSION,
    FrozenIndexEntry,
    read_frozen_index,
)


def local_data_directory() -> Path:
    """Create shared benchmark storage located through Git's absolute common dir.

    All worktrees share the main repository's .agent_state/benchmark/family_pareto.
    Git or filesystem errors propagate; no references are solved here.
    """
    common = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            text=True,
        ).strip()
    )
    directory = common.parent / ".agent_state" / "benchmark" / "family_pareto"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def scene_cache_directory(override: Path | None = None) -> Path:
    """Resolve the current versioned shared cache or an absolute explicit override.

    Git is queried only without an override, from this tool's worktree. Invalid
    relative overrides raise ValueError; Git lookup errors propagate. No cache
    directory is created or cleared. Cache lifetime is independent of tasks.
    """
    if override is not None:
        if not override.is_absolute():
            raise ValueError("--scene-cache must be an absolute path")
        return override
    common = subprocess.check_output(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=Path(__file__).parent,
        text=True,
    ).strip()
    return (
        Path(common).parent
        / ".agent_state/benchmark/family_pareto/scenes"
        / f"v{SCENE_CACHE_VERSION}"
    )


def scene_index_paths(index: Path | None = None) -> tuple[Path, ...]:
    """Select one explicit index or the two tracked synthetic/real indexes."""
    if index is not None:
        return (index,)
    directory = Path(__file__).parent / "data/family_pareto"
    return tuple(
        directory / f"frozen-{category}.index.json"
        for category in ("synthetic", "real")
    )


def indexed_scenes(paths: Sequence[Path]) -> tuple[tuple[Path, FrozenIndexEntry], ...]:
    """Read all identities before cache access, rejecting cross-index duplicates.

    Preserve index/record order. ValueError names ambiguous scenes and all their
    index sources even when no scenario selector was supplied. Codec failures
    propagate. No numeric source, policy or Engine is created.
    """
    entries = tuple(
        (path, entry) for path in paths for entry in read_frozen_index(path)
    )
    sources: dict[str, list[Path]] = {}
    for path, entry in entries:
        sources.setdefault(entry.name, []).append(path)
    duplicates = [
        f"Ambiguous scenario: {name}. Sources: {', '.join(map(str, values))}"
        for name, values in sources.items()
        if len(values) > 1
    ]
    if duplicates:
        raise ValueError("\n".join(duplicates))
    return entries


def missing_scene_command(error: FileNotFoundError, cache: Path) -> FileNotFoundError:
    """Add the tools producer instruction to a codec cache-miss diagnostic."""
    return FileNotFoundError(
        f"{error}\nPrepare scenes with: python -m tools.family_pareto_scenarios "
        f"--scene-cache {cache} (use --scenario NAME for one indexed scene)"
    )
