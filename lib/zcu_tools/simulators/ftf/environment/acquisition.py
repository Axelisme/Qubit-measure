"""Acquisition-scoped clock and private persistence options for FTF callers."""

from dataclasses import dataclass
from pathlib import Path

from zcu_tools.program.v2.simulation import SimulationClock


@dataclass(frozen=True)
class FTFAcquisitionOptions:
    """Select an environment clock and author-only acquisition output paths.

    clock is a caller-owned SimulationClock using monotonic seconds, or None to
    create one per builder/source. Source and binding share it; completed rounds
    add unscaled physical-duration credit. The clock must not be shared between
    independent environments. No hardware clock is read or modified.
    truth_log selects the private drift JSONL file; relative paths use the cwd.
    None uses ~/.local/state/zcu-tools/simulation/ftf/<session>/truth.jsonl.
    Enabled drift creates parents and appends three loop rows at acquisition begin;
    disabled drift creates no file. Log serialization and I/O errors propagate.
    private_replay selects an explicit absolute archive root, or None to disable
    archives. Relative roots raise ValueError at builder/source construction.
    Acquisition appends checked private inputs before returning numeric results;
    archive errors propagate. Callers own both paths and their retention policy.
    This immutable value creates no clock, directories, files or other resources.
    """

    clock: SimulationClock | None = None
    truth_log: Path | None = None
    private_replay: Path | None = None
