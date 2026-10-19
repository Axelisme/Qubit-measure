"""Configure once in the official rclone GUI, then upload/download from Bash."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import zipfile
from pathlib import Path

from .rclone import prepare_rclone, user_directories
from .settings import (
    TransferSettings,
    load_settings,
    validate_entry_name,
)
from .setup import run_setup


def _parser(
    default_local_root: Path, default_settings_dir: Path
) -> argparse.ArgumentParser:
    config_dir, cache_dir = user_directories()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=config_dir,
        help="Per-user rclone account directory (rclone.conf), shared by checkouts",
    )
    parser.add_argument(
        "--settings-dir",
        type=Path,
        default=default_settings_dir,
        help="This checkout's transfer.json directory (default: this checkout/.result-transfer)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=cache_dir,
        help="Verified rclone download cache",
    )
    parser.add_argument(
        "--local-root",
        type=Path,
        default=default_local_root,
        help="Local parent of entry folders (default: this checkout/result)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Print a private GUI setup link rather than opening a browser",
    )
    parser.set_defaults(action="setup")
    actions = parser.add_subparsers(dest="action")
    actions.add_parser(
        "setup", help="Authorize and select a cloud parent with the official GUI"
    )
    actions.add_parser("prepare", help="Download/verify rclone without starting a GUI")
    status = actions.add_parser(
        "status", help="Show saved non-secret location without starting rclone"
    )
    status.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable status; exit 1 if unconfigured",
    )
    for action in ("upload", "download"):
        transfer = actions.add_parser(
            action, help=f"{action.capitalize()} named entry folders without a GUI"
        )
        transfer.add_argument(
            "entries", nargs="*", help="Entry folder names, e.g. Si001 Si002"
        )
        transfer.add_argument(
            "--all",
            action="store_true",
            help="Transfer all immediate source entry folders instead of named entries",
        )
        transfer.add_argument(
            "--dry-run",
            action="store_true",
            help="Report changes without transferring files",
        )
        transfer.add_argument(
            "-j",
            "--jobs",
            type=int,
            default=8,
            help="Parallel file transfers (default: 8)",
        )
    return parser


def _status(
    config_dir: Path, settings_dir: Path, local_root: Path, *, as_json: bool
) -> int:
    selection = load_settings(settings_dir)
    configured = selection is not None and (config_dir / "rclone.conf").is_file()
    value = {
        "configured": configured,
        "remote_parent": selection.remote_parent if selection else None,
        "local_root": str(local_root),
        "config_dir": str(config_dir),
        "settings_dir": str(settings_dir),
    }
    if as_json:
        print(json.dumps(value, ensure_ascii=False))
    elif configured:
        print(f"Cloud parent: {value['remote_parent']}\nLocal results: {local_root}")
    else:
        print("Not configured. Run this command with 'setup' on your desktop first.")
    return 0 if configured else 1


def build_copy_arguments(
    source: str,
    destination: str,
    entries: list[str],
    *,
    jobs: int,
    dry_run: bool,
    live: bool,
) -> list[str]:
    """Return rclone copy arguments (after the binary) for all named entries.

    The caller prepends the rclone binary and --config.

    source and destination are the entry parents: the local root and the remote
    parent, in transfer direction. entries are already-validated folder names; a filter list selects them, so
    one process lists the tree once and reuses its folder-ID cache across
    entries. --fast-list and a relaxed Drive pacer cut per-directory API
    latency; rclone backs off on rate-limit replies. Empty source directories
    are not created. Every copied, updated or skipped file is logged (-v).
    live=True (a terminal) adds rclone's continuously refreshed progress view;
    otherwise a one-line summary is printed every 10 s.
    """
    command = [
        "copy",
        source,
        destination,
        "--ask-password=false",
        "--update",
        "--fast-list",
        "--checkers",
        "16",
        "--transfers",
        str(jobs),
        "--drive-pacer-min-sleep",
        "10ms",
        "--drive-pacer-burst",
        "200",
        "-v",
        "--filter",
        "- *.lock",
    ]
    for entry in entries:
        command += ["--filter", f"+ /{entry}/**"]
    command += ["--filter", "- **"]
    command += ["--progress"] if live else ["--stats", "10s", "--stats-one-line"]
    if dry_run:
        command.append("--dry-run")
    return command


def _transfer(
    binary: Path,
    config_dir: Path,
    selection: TransferSettings,
    local_root: Path,
    args: argparse.Namespace,
) -> int:
    local, remote = str(local_root), selection.remote_parent
    source, destination = (
        (local, remote) if args.action == "upload" else (remote, local)
    )
    command = [str(binary), "--config", str(config_dir / "rclone.conf")]
    command += build_copy_arguments(
        source,
        destination,
        args.entries,
        jobs=args.jobs,
        dry_run=args.dry_run,
        live=sys.stdout.isatty(),
    )
    print(f"{args.action}: {' '.join(args.entries)}", flush=True)
    with subprocess.Popen(command, stdin=subprocess.DEVNULL) as process:
        try:
            return process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def _interrupt(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt


def _validate_entries(entries: list[str], local_root: Path, action: str) -> None:
    for entry in entries:
        validate_entry_name(entry)
        if (local_root / entry).is_symlink():
            raise ValueError(f"Entry is a symlink: {entry}")
        if action == "upload" and not (local_root / entry).is_dir():
            raise ValueError(f"Local entry folder does not exist: {local_root / entry}")


def _all_entries(
    binary: Path,
    config_dir: Path,
    selection: TransferSettings,
    local_root: Path,
    action: str,
) -> list[str]:
    if action == "upload":
        if not local_root.is_dir():
            raise ValueError(f"Local results directory does not exist: {local_root}")
        return sorted(
            child.name
            for child in local_root.iterdir()
            if child.is_dir() and not child.is_symlink() and not child.match("*.lock")
        )
    result = subprocess.run(
        [
            str(binary),
            "lsjson",
            selection.remote_parent,
            "--dirs-only",
            "--max-depth",
            "1",
            "--exclude",
            "*.lock",
            "--config",
            str(config_dir / "rclone.conf"),
            "--ask-password=false",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        check=True,
    )
    value: object = json.loads(result.stdout)
    if not isinstance(value, list):
        raise ValueError("rclone returned a non-list directory listing.")
    entries: list[str] = []
    for item in value:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("Path"), str)
            or item.get("IsDir") is not True
        ):
            raise ValueError("rclone returned an invalid entry directory.")
        entries.append(item["Path"])
    return sorted(entries)


def _run(default_local_root: Path, default_settings_dir: Path) -> int:
    """Run setup or CLI copy; return rclone failures and never prompt an unattended agent.

    Common path overrides precede the subcommand. Without a subcommand, opens
    setup. Missing settings trigger GUI setup only for an interactive transfer;
    noninteractive transfers fail immediately with exit 2 and setup instructions.
    """
    parser = _parser(default_local_root, default_settings_dir)
    args = parser.parse_args()
    action = args.action or "setup"
    config_dir = args.config_dir.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    settings_dir = args.settings_dir.expanduser().resolve()
    local_root = args.local_root.expanduser().resolve()
    if action == "status":
        return _status(config_dir, settings_dir, local_root, as_json=args.json)
    if action in ("upload", "download"):
        if bool(args.entries) == args.all:
            parser.error("Specify entry names or --all, but not both.")
        if args.jobs < 1:
            raise ValueError("--jobs must be positive.")
        _validate_entries(args.entries, local_root, action)
        selection = load_settings(settings_dir)
        if selection is None and not sys.stdin.isatty():
            print(
                "Not configured. Run 'transfer_results.py setup' on your desktop first; CLI transfer did not start a GUI.",
                file=sys.stderr,
            )
            return 2
    else:
        selection = None
    config_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    settings_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    binary = prepare_rclone(cache_dir)
    if action == "prepare":
        print(f"Verified executable: {binary}")
        return 0
    if action == "setup" or selection is None:
        selection = run_setup(
            binary, config_dir, cache_dir, settings_dir, no_browser=args.no_browser
        )
        if action == "setup":
            return 0
    if args.all:
        args.entries = _all_entries(binary, config_dir, selection, local_root, action)
        _validate_entries(args.entries, local_root, action)
        if not args.entries:
            print("No entry folders found; nothing to transfer.")
            return 0
    return _transfer(binary, config_dir, selection, local_root, args)


def main(default_local_root: Path, default_settings_dir: Path) -> int:
    """Run the transfer CLI with caller-supplied checkout defaults.

    default_local_root is the local entry parent (the checkout's result/);
    default_settings_dir holds that checkout's transfer.json, so different
    checkouts keep separate cloud parents while sharing the per-user rclone
    account. Parses process arguments, installs signal handling, and returns a
    shell exit status. Library installation paths never determine where result
    data or settings are written.
    """
    signal.signal(signal.SIGTERM, _interrupt)
    try:
        return _run(default_local_root, default_settings_dir)
    except KeyboardInterrupt:
        print("\nResult transfer cancelled.", file=sys.stderr)
        return 130
    except subprocess.CalledProcessError as exc:
        print(
            f"Could not list source entry folders (rclone exit {exc.returncode}).",
            file=sys.stderr,
        )
        return exc.returncode
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        print(f"Result transfer failed: {exc}", file=sys.stderr)
        return 1
