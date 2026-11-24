"""Copy result and Database trees between a checkout and named cloud sources."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

from . import rclone, setup
from .discovery import list_sources
from .settings import (
    DATA_DIRS,
    SyncSettings,
    load_settings,
    save_settings,
    validate_source,
)


def _parser() -> argparse.ArgumentParser:
    config_dir, cache_dir = rclone.user_directories()
    parser = argparse.ArgumentParser(prog="sync_data.py", description=__doc__)
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=config_dir,
        help="Per-user rclone account directory (rclone.conf), shared by checkouts",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=cache_dir,
        help="Verified rclone download cache",
    )
    actions = parser.add_subparsers(dest="action", required=True)
    configure = actions.add_parser("setup", help="Authorize and select the cloud root")
    configure.add_argument(
        "--source",
        required=True,
        help="Unique source prefix (suggestion: short-hostname/username)",
    )
    configure.add_argument(
        "--labdata",
        help="Local collection root; relative paths use the working directory",
    )
    configure.add_argument(
        "--no-browser", action="store_true", help="Print a private setup link"
    )
    status = actions.add_parser(
        "status", help="Read this checkout's local settings only"
    )
    status.add_argument(
        "--json", action="store_true", help="Emit machine-readable status"
    )
    push = actions.add_parser(
        "push", help="Copy checkout data to its configured source"
    )
    _add_copy_options(push)
    pull = actions.add_parser("pull", help="Copy named sources to the local collection")
    pull.add_argument("prefixes", nargs="*", help="Source prefixes, e.g. alice/pc1/sim")
    pull.add_argument(
        "--all",
        action="store_true",
        help="Pull all discovered sources except this checkout's",
    )
    _add_copy_options(pull)
    actions.add_parser("sources", help="List discovered cloud source prefixes")
    return parser


def _add_copy_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dry-run", action="store_true", help="Report changes without copying"
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=8, help="Parallel transfers (default: 8)"
    )


def _status(
    selection: SyncSettings | None, checkout: Path, *, configured: bool, as_json: bool
) -> int:
    if as_json:
        print(
            json.dumps(
                {
                    "configured": configured,
                    "remote_root": selection.remote_root if selection else None,
                    "source": selection.source if selection else None,
                    "labdata": str(selection.labdata) if selection else None,
                    "project_root": str(checkout),
                },
                ensure_ascii=False,
            )
        )
    elif configured and selection:
        print(
            f"Cloud root: {selection.remote_root}\nSource: {selection.source}\nLabdata: {selection.labdata}"
        )
    else:
        print("Not configured. Run 'sync_data.py setup' first.")
    return 0 if configured else 1


def _copy_options(args: argparse.Namespace) -> list[str]:
    command = [
        "--update",
        "--fast-list",
        "--checkers",
        "16",
        "--transfers",
        str(args.jobs),
        "--drive-pacer-min-sleep",
        "10ms",
        "--drive-pacer-burst",
        "200",
        "-v",
        "--filter",
        "- *.lock",
    ]
    command += (
        ["--progress"]
        if sys.stdout.isatty()
        else ["--stats", "10s", "--stats-one-line"]
    )
    if args.dry_run:
        command.append("--dry-run")
    return command


def _push(
    checkout: Path,
    selection: SyncSettings,
    binary: Path,
    config_dir: Path,
    args: argparse.Namespace,
) -> int:
    roots = [checkout / directory for directory in DATA_DIRS]
    for root in roots:
        if (root.exists() or root.is_symlink()) and not root.is_dir():
            raise ValueError(f"Not a data directory: {root}")
    for root in roots:
        if not root.is_dir():
            continue
        target = selection.remote_path(f"{selection.source}/{root.name}")
        print(f"Push: {root} -> {target}", flush=True)
        result = rclone.run_cli(
            binary,
            config_dir,
            ["copy", str(root.resolve()), target, *_copy_options(args)],
        )
        if result.returncode:
            return result.returncode
    return 0


def _pull_destination(labdata: Path, prefix: str, checkout: Path) -> Path:
    destination = (labdata / prefix).resolve()
    owned_trees = [(checkout / directory).resolve() for directory in DATA_DIRS]
    incoming_trees = [(destination / directory).resolve() for directory in DATA_DIRS]
    if (
        destination.is_relative_to(checkout)
        or any(tree.is_relative_to(checkout) for tree in incoming_trees)
        or any(
            incoming.is_relative_to(owned) or owned.is_relative_to(incoming)
            for incoming in incoming_trees
            for owned in owned_trees
        )
    ):
        raise ValueError(
            f"Pull destination overlaps the checkout or its data: {destination}"
        )
    return destination


def _pull(
    checkout: Path,
    selection: SyncSettings,
    binary: Path,
    config_dir: Path,
    args: argparse.Namespace,
) -> int:
    if bool(args.prefixes) == args.all:
        raise ValueError("Specify source prefixes or --all, but not both.")
    if args.all:
        prefixes = [
            prefix
            for prefix in list_sources(binary, config_dir, selection.remote_root)
            if prefix != selection.source
        ]
    else:
        prefixes = [validate_source(prefix) for prefix in args.prefixes]
    targets = [
        (prefix, _pull_destination(selection.labdata, prefix, checkout))
        for prefix in prefixes
    ]
    options = _copy_options(args)
    for directory in DATA_DIRS:
        options += ["--filter", f"+ /{directory}/**"]
    options += ["--filter", "- **"]
    for prefix, destination in targets:
        origin = selection.remote_path(prefix)
        print(f"Pull: {origin} -> {destination}", flush=True)
        result = rclone.run_cli(
            binary, config_dir, ["copy", origin, str(destination), *options]
        )
        if result.returncode:
            return result.returncode
    return 0


def _run_setup(
    settings_dir: Path, config_dir: Path, cache_dir: Path, args: argparse.Namespace
) -> int:
    source = validate_source(args.source)
    config_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    remote_root = setup.select_remote_root(
        rclone.prepare_rclone(cache_dir),
        config_dir,
        cache_dir,
        no_browser=args.no_browser,
    )
    labdata = _confirm_labdata(remote_root, source, args.labdata)
    settings_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    save_settings(settings_dir, SyncSettings(remote_root, source, labdata))
    print(f"Saved settings for {source}.")
    return 0


def _confirm_labdata(remote_root: str, source: str, path: str | None) -> Path:
    try:
        if path is None:
            remote, _, folder = remote_root.partition(":")
            default = Path.home() / (folder.rsplit("/", 1)[-1] or remote)
            path = input(f"Local collection root [{default}]: ") or str(default)
        labdata = Path(path).expanduser().resolve()
        print(f"Cloud root: {remote_root}\nSource: {source}\nLabdata: {labdata}")
        input("Verify these settings, then press Enter to save (Ctrl+C cancels): ")
    except EOFError as exc:
        raise RuntimeError("Setup cancelled: no settings were saved.") from exc
    return labdata


def _run(args: argparse.Namespace, checkout: Path) -> int:
    config_dir = args.config_dir.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    settings_dir = checkout / ".data-sync"
    if args.action == "setup":
        return _run_setup(settings_dir, config_dir, cache_dir, args)
    selection = load_settings(settings_dir)
    has_account = (config_dir / "rclone.conf").is_file()
    if args.action == "status":
        return _status(
            selection,
            checkout,
            configured=selection is not None and has_account,
            as_json=args.json,
        )
    if selection is None or not has_account:
        print("Not configured. Run 'sync_data.py setup' first.", file=sys.stderr)
        return 2
    if args.action == "sources":
        for source in list_sources(
            rclone.prepare_rclone(cache_dir), config_dir, selection.remote_root
        ):
            print(source)
        return 0
    if args.jobs < 1:
        raise ValueError("--jobs must be positive.")
    binary = rclone.prepare_rclone(cache_dir)
    if args.action == "pull":
        return _pull(checkout, selection, binary, config_dir, args)
    return _push(checkout, selection, binary, config_dir, args)


def main(argv: list[str] | None = None, *, checkout: Path) -> int:
    """Run argv (None reads sys.argv) using checkout for data and .data-sync.

    Return a shell exit status; argparse raises SystemExit for help or invalid
    syntax. Only setup opens the browser. A configured push reads result/ and
    Database/ in checkout, including directories linked from those two paths.
    """
    parser = _parser()
    try:
        return _run(parser.parse_args(argv), checkout.resolve())
    except KeyboardInterrupt:
        print("\nData sync cancelled.", file=sys.stderr)
        return 130
    except subprocess.CalledProcessError as exc:
        print(
            f"Could not list cloud sources (rclone exit {exc.returncode}).",
            file=sys.stderr,
        )
        return exc.returncode
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        print(f"Data sync failed: {exc}", file=sys.stderr)
        return 1
