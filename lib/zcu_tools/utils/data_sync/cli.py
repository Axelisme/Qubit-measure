"""Copy result and Database trees between Labdata and the cloud."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

from . import rclone, setup
from .binding import bind_checkout, binding_issues, require_binding
from .discovery import list_sources
from .settings import (
    DATA_DIRS,
    SyncSettings,
    load_settings,
    save_settings,
    validate_prefix,
    validate_remote_root,
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
        "--prefix",
        required=True,
        help="Labdata prefix to bind to this checkout, e.g. alice/pc1/sim",
    )
    configure.add_argument(
        "--labdata",
        help="Local Labdata root; relative paths use the working directory",
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
    push = actions.add_parser("push", help="Copy a Labdata prefix to the cloud")
    push.set_defaults(all=False, prefixes=())
    push.add_argument(
        "prefix", nargs="?", help="Labdata prefix (default: bound prefix)"
    )
    _add_copy_options(push)
    pull = actions.add_parser("pull", help="Copy cloud data to Labdata")
    pull.set_defaults(prefix=None)
    pull.add_argument(
        "prefixes", nargs="*", help="Labdata prefixes (default: bound prefix)"
    )
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
        "--force", action="store_true", help="Overwrite from the source for one prefix"
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=8, help="Parallel transfers (default: 8)"
    )


def _print_roots(selection: SyncSettings) -> None:
    print(f"Labdata: {selection.labdata}")
    print(f"Labdata prefix: {selection.prefix}")
    print(f"Local data root: {selection.local_root(selection.prefix)}")
    print(f"Cloud data root: {selection.remote_path(selection.prefix)}")


def _status(
    selection: SyncSettings | None, checkout: Path, *, has_account: bool, as_json: bool
) -> int:
    issues = binding_issues(checkout, selection) if selection is not None else []
    bound = selection is not None and not issues
    configured = bound and has_account
    if as_json:
        print(
            json.dumps(
                {
                    "configured": configured,
                    "bound": bound,
                    "binding_errors": issues,
                    "remote_root": selection.remote_root if selection else None,
                    "prefix": selection.prefix if selection else None,
                    "labdata": str(selection.labdata) if selection else None,
                    "local_root": (
                        str(selection.local_root(selection.prefix))
                        if selection
                        else None
                    ),
                    "remote_data_root": (
                        selection.remote_path(selection.prefix) if selection else None
                    ),
                    "project_root": str(checkout),
                },
                ensure_ascii=False,
            )
        )
    else:
        if selection is not None:
            _print_roots(selection)
            print("Checkout binding: valid." if bound else "\n".join(issues))
        if not configured:
            print("Not configured. Run 'sync_data.py setup' first.")
    return 0 if configured else 1


def _copy_options(args: argparse.Namespace) -> list[str]:
    command = [
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
    # Dropping --update alone would still skip files with equal size and mtime.
    command.append("--ignore-times" if args.force else "--update")
    if args.dry_run:
        command.append("--dry-run")
    for directory in DATA_DIRS:
        command += ["--filter", f"+ /{directory}/**"]
    command += ["--filter", "- **"]
    return command


def _transfer_root(checkout: Path, selection: SyncSettings, prefix: str) -> Path:
    local = selection.local_root(prefix)
    if prefix == selection.prefix:
        return local

    destination = local.resolve()
    owned_trees = [
        (root / directory).resolve()
        for root in (checkout, selection.local_root(selection.prefix))
        for directory in DATA_DIRS
    ]
    trees = [destination / directory for directory in DATA_DIRS]
    resolved_trees = [(tree, tree.resolve()) for tree in trees]
    # Labdata may be inside the checkout, but a different prefix must not
    # alias a checkout path or overlap any of its bound data trees.
    if (
        destination == checkout
        or any(checkout.is_relative_to(tree) for _, tree in resolved_trees)
        or any(
            actual != expected and actual.is_relative_to(checkout)
            for expected, actual in [(local, destination), *resolved_trees]
        )
        or any(
            tree.is_relative_to(owned) or owned.is_relative_to(tree)
            for _, tree in resolved_trees
            for owned in owned_trees
        )
    ):
        raise ValueError(
            f"Local data root overlaps the checkout or its data: {destination}"
        )
    return destination


def _transfer(
    checkout: Path,
    selection: SyncSettings,
    cache_dir: Path,
    config_dir: Path,
    args: argparse.Namespace,
) -> int:
    named = args.prefixes if args.prefix is None else [args.prefix]
    if args.force and (args.all or len(named) > 1):
        raise ValueError("--force requires one prefix; omit --all and extra prefixes.")
    if args.all and named:
        raise ValueError("Specify prefixes or --all, but not both.")

    prefixes = [validate_prefix(prefix) for prefix in named]
    if not args.all:
        prefixes = prefixes or [selection.prefix]
        if selection.prefix in prefixes:
            require_binding(checkout, selection)

    binary = rclone.prepare_rclone(cache_dir)
    if args.all:
        prefixes = [
            prefix
            for prefix in list_sources(binary, config_dir, selection.remote_root)
            if prefix != selection.prefix
        ]
    targets = [
        (
            str(_transfer_root(checkout, selection, prefix)),
            selection.remote_path(prefix),
        )
        for prefix in prefixes
    ]
    options = _copy_options(args)
    for local, remote in targets:
        origin, destination = (
            (local, remote) if args.action == "push" else (remote, local)
        )
        print(f"{args.action.capitalize()}: {origin} -> {destination}", flush=True)
        result = rclone.run_cli(
            binary, config_dir, ["copy", origin, destination, *options]
        )
        if result.returncode:
            return result.returncode
    return 0


def _run_setup(
    checkout: Path, config_dir: Path, cache_dir: Path, args: argparse.Namespace
) -> int:
    prefix = validate_prefix(args.prefix)
    config_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    remote_root = validate_remote_root(
        setup.select_remote_root(
            rclone.prepare_rclone(cache_dir),
            config_dir,
            cache_dir,
            no_browser=args.no_browser,
        )
    )
    selection = _confirm_settings(remote_root, prefix, args.labdata)
    bind_checkout(checkout, selection)
    settings_dir = checkout / ".data-sync"
    settings_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    save_settings(settings_dir, selection)
    print(f"Saved settings for {prefix}.")
    return 0


def _confirm_settings(remote_root: str, prefix: str, path: str | None) -> SyncSettings:
    try:
        if path is None:
            remote, _, folder = remote_root.partition(":")
            default = Path.home() / (folder.rsplit("/", 1)[-1] or remote)
            path = input(f"Local Labdata root [{default}]: ") or str(default)
        selection = SyncSettings(remote_root, prefix, Path(path).expanduser().resolve())
        _print_roots(selection)
        input("Verify these settings, then press Enter to bind (Ctrl+C cancels): ")
    except EOFError as exc:
        raise RuntimeError("Setup cancelled: no settings were saved.") from exc
    return selection


def _run(args: argparse.Namespace, checkout: Path) -> int:
    config_dir = args.config_dir.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    settings_dir = checkout / ".data-sync"
    if args.action == "setup":
        return _run_setup(checkout, config_dir, cache_dir, args)
    selection = load_settings(settings_dir)
    has_account = (config_dir / "rclone.conf").is_file()
    if args.action == "status":
        return _status(
            selection,
            checkout,
            has_account=has_account,
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
    return _transfer(checkout, selection, cache_dir, config_dir, args)


def main(argv: list[str] | None = None, *, checkout: Path) -> int:
    """Run argv (None reads sys.argv) using checkout for data and .data-sync.

    Return a shell exit status; argparse raises SystemExit for help or invalid
    syntax. Only setup opens the browser. Transfers copy result/ and Database/
    under the chosen Labdata prefix, defaulting to the checkout's bound prefix.
    Transferring the bound prefix requires both checkout links to be valid.
    --force overwrites from the source for one prefix only. Neither direction
    deletes destination-only files.
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
