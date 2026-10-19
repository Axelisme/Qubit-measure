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


def _parser(default_local_root: Path) -> argparse.ArgumentParser:
    config_dir, cache_dir = user_directories()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=config_dir,
        help="Dedicated account and transfer settings directory",
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
            default=4,
            help="Parallel file transfers (default: 4)",
        )
    return parser


def _status(config_dir: Path, local_root: Path, *, as_json: bool) -> int:
    selection = load_settings(config_dir)
    configured = selection is not None and (config_dir / "rclone.conf").is_file()
    value = {
        "configured": configured,
        "remote_parent": selection.remote_parent if selection else None,
        "local_root": str(local_root),
        "config_dir": str(config_dir),
    }
    if as_json:
        print(json.dumps(value, ensure_ascii=False))
    elif configured:
        print(f"Cloud parent: {value['remote_parent']}\nLocal results: {local_root}")
    else:
        print("Not configured. Run this command with 'setup' on your desktop first.")
    return 0 if configured else 1


def _transfer(
    binary: Path,
    config_dir: Path,
    selection: TransferSettings,
    local_root: Path,
    args: argparse.Namespace,
) -> int:
    parent = selection.remote_parent
    separator = "" if parent.endswith(":") else "/"
    for entry in args.entries:
        local = str(local_root / entry)
        remote = parent + separator + entry
        source, destination = (
            (local, remote) if args.action == "upload" else (remote, local)
        )
        command = [
            str(binary),
            "copy",
            source,
            destination,
            "--config",
            str(config_dir / "rclone.conf"),
            "--ask-password=false",
            "--update",
            "--exclude",
            "*.lock",
            "--create-empty-src-dirs",
            "--transfers",
            str(args.jobs),
        ]
        if args.dry_run:
            command.append("--dry-run")
        print(f"{args.action}: {entry}", flush=True)
        with subprocess.Popen(command, stdin=subprocess.DEVNULL) as process:
            try:
                result = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        if result != 0:
            return result
    return 0


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


def _run(default_local_root: Path) -> int:
    """Run setup or CLI copy; return rclone failures and never prompt an unattended agent.

    Common path overrides precede the subcommand. Without a subcommand, opens
    setup. Missing settings trigger GUI setup only for an interactive transfer;
    noninteractive transfers fail immediately with exit 2 and setup instructions.
    """
    parser = _parser(default_local_root)
    args = parser.parse_args()
    action = args.action or "setup"
    config_dir = args.config_dir.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    local_root = args.local_root.expanduser().resolve()
    if action == "status":
        return _status(config_dir, local_root, as_json=args.json)
    if action in ("upload", "download"):
        if bool(args.entries) == args.all:
            parser.error("Specify entry names or --all, but not both.")
        if args.jobs < 1:
            raise ValueError("--jobs must be positive.")
        _validate_entries(args.entries, local_root, action)
        selection = load_settings(config_dir)
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
    binary = prepare_rclone(cache_dir)
    if action == "prepare":
        print(f"Verified executable: {binary}")
        return 0
    if action == "setup" or selection is None:
        selection = run_setup(binary, config_dir, cache_dir, no_browser=args.no_browser)
        if action == "setup":
            return 0
    if args.all:
        args.entries = _all_entries(binary, config_dir, selection, local_root, action)
        _validate_entries(args.entries, local_root, action)
        if not args.entries:
            print("No entry folders found; nothing to transfer.")
            return 0
    return _transfer(binary, config_dir, selection, local_root, args)


def main(default_local_root: Path) -> int:
    """Run the transfer CLI with a caller-supplied default local results parent.

    Parses process arguments, installs signal handling, and returns a shell exit
    status. The script composition root supplies its checkout's results path;
    library installation paths never determine where result data is written.
    """
    signal.signal(signal.SIGTERM, _interrupt)
    try:
        return _run(default_local_root)
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
