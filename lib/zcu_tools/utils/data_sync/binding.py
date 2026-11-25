"""Bind checkout data directories to the physical Labdata tree."""

from __future__ import annotations

import stat
import sys
from contextlib import suppress
from pathlib import Path

from .settings import DATA_DIRS, SyncSettings

if sys.platform == "win32":
    import _winapi


def bind_checkout(checkout: Path, selection: SyncSettings) -> None:
    """Create missing data links under checkout, leaving valid links in place.

    Check both link locations and physical targets before creating anything.
    Create missing target directories, then verify both links. Raise on an
    invalid binding or filesystem failure. Leave any directories or links
    created before the failure in place; never move data or save settings.
    """
    issues = _binding_issues(checkout, selection, allow_missing=True)
    if issues:
        raise ValueError("\n".join(issues))
    root = selection.local_root(selection.prefix)
    for name in DATA_DIRS:
        target = root / name
        target.mkdir(parents=True, exist_ok=True)
        link = checkout / name
        if not link.exists(follow_symlinks=False):
            _create_link(link, target)
    issues = binding_issues(checkout, selection)
    if issues:
        raise ValueError("\n".join(issues))


def binding_issues(checkout: Path, selection: SyncSettings) -> list[str]:
    """Return diagnostics for invalid checkout links or targets, or an empty list."""
    return _binding_issues(checkout, selection, allow_missing=False)


def require_binding(checkout: Path, selection: SyncSettings) -> None:
    """Raise ValueError unless both checkout data links reach the saved Labdata targets."""
    issues = binding_issues(checkout, selection)
    if issues:
        raise ValueError("\n".join([*issues, "Run 'sync_data.py setup' again."]))


def _binding_issues(
    checkout: Path, selection: SyncSettings, *, allow_missing: bool
) -> list[str]:
    root = selection.local_root(selection.prefix)
    links = [checkout / name for name in DATA_DIRS]
    issues: list[str] = []
    for link in links:
        target = root / link.name
        if problem := _target_problem(target, links):
            issues.append(
                f"{link}: {problem}; expected a physical directory at {target}."
            )
        problem = _link_problem(link, target)
        if problem and not (allow_missing and problem == "missing"):
            issues.append(f"{link}: {problem}; expected a link to {target}.")
    return issues


def _target_problem(target: Path, links: list[Path]) -> str | None:
    try:
        physical = target.resolve()
    except (OSError, RuntimeError) as exc:
        return f"cannot resolve target ({exc})"
    for link in links:
        if any(
            path.is_relative_to(link) or link.is_relative_to(path)
            for path in (target, physical)
        ):
            return f"target overlaps checkout data location {link}"
    if physical != target:
        return f"target aliases a different location ({physical})"
    for path in (target, *target.parents):
        try:
            mode = path.stat().st_mode
        except FileNotFoundError:
            continue
        except OSError as exc:
            return f"cannot inspect target ({exc})"
        return None if stat.S_ISDIR(mode) else f"target path contains a file ({path})"
    return None


def _link_problem(link: Path, target: Path) -> str | None:
    try:
        mode = link.lstat().st_mode
    except FileNotFoundError:
        return "missing"
    if not (stat.S_ISLNK(mode) or link.is_junction()):
        return "existing directory" if stat.S_ISDIR(mode) else "existing file"
    if not link.is_dir():
        return "broken link or non-directory target"
    actual = link.resolve(strict=True)
    if actual != target:
        return f"wrong link (points to {actual})"
    return None


def _create_link(link: Path, target: Path) -> None:
    try:
        if sys.platform == "win32":
            _winapi.CreateJunction(str(target), str(link))
        else:
            link.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        message = f"Could not create checkout link {link} to {target}: {exc}"
        # CreateJunction can create a directory before setting its reparse tag.
        # If inspection also fails, report the original error.
        with suppress(OSError):
            if stat.S_ISDIR(link.lstat().st_mode) and not link.is_junction():
                message += (
                    "\nA plain directory is present at the link path. "
                    "Check it before retrying."
                )
        raise OSError(message) from exc
