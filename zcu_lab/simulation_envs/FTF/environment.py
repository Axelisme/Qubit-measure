"""Trusted FTF environment shared by startup scripts and Notebook callers."""

import os
from pathlib import Path

from zcu_tools.experiment.ftf import ftf
from zcu_tools.experiment.simulation import SimulationEnvironment


def build_environment(
    *,
    settings_path: Path | None = None,
    workers: int | None = None,
    warmup: bool = True,
) -> SimulationEnvironment:
    """Read FTF YAML once and return caller-owned offline resources.

    settings_path selects a YAML file; None uses adjacent ftf.yaml. Relative
    paths resolve from cwd. Missing files and YAML/binding errors propagate
    without falling back to another file.
    workers is a positive parallel width; None reserves two host CPUs and uses
    at least one worker. warmup=True starts process-shared background numeric
    preparation; False avoids startup compilation. Acquire retains its existing
    ready handling. Invalid widths, YAML and binding errors propagate.
    Call from a script worker or Notebook thread. No hardware is accessed. The
    caller owns returned resources and closes the device manager after work has
    drained, or transfers drivers to a host with an explicit cleanup handoff.
    """
    path = (
        Path(__file__).with_name("ftf.yaml") if settings_path is None else settings_path
    )
    # ftf.yaml is the user's ignored working copy of the tracked default.yaml.
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; copy default.yaml to ftf.yaml")
    width = max(1, (os.cpu_count() or 1) - 2) if workers is None else workers
    return ftf(path, workers=width, warmup=warmup)
