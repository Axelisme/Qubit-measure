"""External, collision-resistant backup destinations for explicit file upgrades."""

from pathlib import Path
from uuid import uuid4


def upgrade_backup_path(source: Path, *, backup_root: Path | None = None) -> Path:
    """Resolve a unique .bak destination without writing files or directories.

    Find the nearest result/Database ancestor of the resolved source and mirror
    its project-relative path under <project>-backup. backup_root overrides that
    root; without a project ancestor it is required and receives the basename.
    Relative paths use cwd. Resolve symlinks before rejecting destinations inside
    any result/Database tree, or inside a standalone source parent. Raise
    ValueError for an unidentified project without an override or unsafe output;
    filesystem resolution errors propagate. Call before acquiring write locks.
    The caller creates parents and opens the destination exclusively.
    """
    source = source.resolve()
    data_root = next(
        (parent for parent in source.parents if parent.name in {"result", "Database"}),
        None,
    )
    if data_root is None:
        if backup_root is None:
            raise ValueError(
                f"{source}: cannot locate result/Database; specify --backup-root"
            )
        relative = Path(source.name)
        root = backup_root.resolve()
    else:
        project = data_root.parent
        relative = source.relative_to(project)
        root = (backup_root or project.with_name(f"{project.name}-backup")).resolve()
    destination = (
        (root / relative).with_name(f"{source.name}.{uuid4().hex}.bak").resolve()
    )
    if any(parent.name in {"result", "Database"} for parent in (root, *root.parents)):
        raise ValueError(f"{root}: backup cannot be inside a result/Database tree")
    if not destination.is_relative_to(root):
        raise ValueError(f"{destination}: backup mirror symlink escapes backup root")
    if data_root is None and destination.is_relative_to(source.parent):
        raise ValueError(
            f"{destination}: backup cannot be inside standalone source parent"
        )
    return destination
