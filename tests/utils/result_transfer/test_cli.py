from zcu_tools.utils.result_transfer.cli import build_copy_arguments


def _build(
    entries: list[str], *, dry_run: bool = False, live: bool = False
) -> list[str]:
    return build_copy_arguments(
        "/repo/result",
        "lab:results",
        entries,
        jobs=8,
        dry_run=dry_run,
        live=live,
    )


def _filters(command: list[str]) -> list[str]:
    return [command[i + 1] for i, arg in enumerate(command) if arg == "--filter"]


def test_all_entries_go_in_one_command():
    command = _build(["Si001", "Si002"])

    assert command[:3] == ["copy", "/repo/result", "lab:results"]
    assert _filters(command) == ["- *.lock", "+ /Si001/**", "+ /Si002/**", "- **"]
    assert "--fast-list" in command
    assert "--create-empty-src-dirs" not in command
    assert command[command.index("--transfers") + 1] == "8"
    assert "--dry-run" not in command


def test_dry_run_flag_is_last():
    assert _build(["Si001"], dry_run=True)[-1] == "--dry-run"


def test_live_terminal_gets_progress_view_and_file_log():
    live = _build(["Si001"], live=True)
    batch = _build(["Si001"])

    assert "--progress" in live and "-v" in live
    assert "--progress" not in batch and "-v" in batch
    assert batch[batch.index("--stats") + 1] == "10s"
