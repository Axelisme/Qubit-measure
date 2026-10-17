"""Trusted FTF environment shared by GUI --simulate and Notebook callers."""

import os
from pathlib import Path

from zcu_tools.experiment.ftf import ftf
from zcu_tools.experiment.simulation import SimulationEnvironment


def build_environment() -> SimulationEnvironment:
    """Read adjacent FTF YAML once and return caller-owned offline resources.

    The host closes the returned device manager after acquisition has drained.
    YAML/binding errors abort startup without selecting another environment.
    """
    path = Path(__file__).with_name("ftf.yaml")
    # ftf.yaml is the user's ignored working copy of the tracked default.yaml.
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; copy default.yaml to ftf.yaml")
    count = os.process_cpu_count() or 1
    return ftf(path, workers=max(1, count - 2))
