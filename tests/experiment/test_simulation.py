"""Trusted environment authors can use ordinary Python dataclasses."""

from zcu_tools.experiment.simulation import load_environment


def test_environment_builder_can_resolve_its_own_postponed_annotations(tmp_path):
    path = tmp_path / "authored.py"
    path.write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass, replace\n"
        "from zcu_tools.experiment.simulation import white_noise\n"
        "@dataclass\n"
        "class Settings:\n"
        "    label: str = 'authored environment'\n"
        "def build_environment():\n"
        "    return replace(white_noise(), description=Settings().label)\n"
    )
    environment = load_environment(path)
    try:
        assert environment.description == "authored environment"
    finally:
        environment.devices.close_all_devices()
