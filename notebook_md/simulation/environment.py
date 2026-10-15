"""Trusted FTF environment shared by GUI --simulate and Notebook callers."""

from pathlib import Path

from zcu_tools.experiment.ftf import ftf
from zcu_tools.experiment.simulation import SimulationEnvironment


def build_environment() -> SimulationEnvironment:
    """Read adjacent FTF YAML once and return caller-owned offline resources.

    The host closes the returned device manager after acquisition has drained.
    YAML/binding errors abort startup without selecting another environment.
    """
    return ftf(Path(__file__).with_name("ftf.yaml"))
