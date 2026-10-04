"""Observable entry behavior with explicitly registered lab definitions."""

from collections.abc import Generator
from pathlib import Path

import pytest
from pydantic import ValidationError
from zcu_tools.resources.entry import (
    MissingReferenceError,
    ResultEntry,
    RoleResolutionError,
    component_registry,
)
from zcu_tools.resources.entry.builtin_kinds import register_all
from zcu_tools.resources.entry.views import FieldView

from .fakes import registry_state, restore_registry


@pytest.fixture(scope="module", autouse=True)
def entry_registry_models() -> Generator[None]:
    before = registry_state()
    try:
        register_all(component_registry)
        yield
    finally:
        restore_registry(before)


@pytest.mark.parametrize("kind", ["qubit/transmon", "qubit/fluxonium"])
def test_resonator_role_follows_bound_qubit_reference(
    tmp_path: Path, kind: str
) -> None:
    entry = ResultEntry.create(
        "lab", result_root=tmp_path / "results", database_root=tmp_path / "database"
    )
    entry.setup.add_component("R1", kind="resonator", freq=6500.0)
    entry.setup.add_component("Q1", kind=kind, freq=4300.0, resonator="R1")
    point = entry.new_point("working")
    entry.setup.add_component("Q2", kind=kind, freq=5100.0)
    roles = point.resolve()
    assert roles.components == {"qubit": "Q1", "resonator": "R1"}
    assert roles.qubit.freq == 4300.0
    roles.resonator.freq = 6550.0
    assert point.R1.freq == 6550.0
    assert entry.setup.R1.freq == 6500.0
    point.refresh()
    assert point.resolve().resonator.freq == 6550.0


def test_invalid_resonator_reference_preserves_point_and_source(tmp_path: Path) -> None:
    entry = ResultEntry.create(
        "lab", result_root=tmp_path / "results", database_root=tmp_path / "database"
    )
    entry.setup.add_component("R1", kind="resonator", freq=6500.0)
    entry.setup.add_component("Q1", kind="qubit/transmon", resonator="R1")
    point = entry.new_point("working")
    source = point.meta("Q1.resonator")
    with pytest.raises(MissingReferenceError):
        point.Q1.resonator = "missing"
    assert point.Q1.resonator == "R1"
    assert point.meta("Q1.resonator") == source
    point.refresh()
    assert point.resolve().resonator.freq == 6500.0


def test_resonator_role_rejects_reference_to_another_kind(tmp_path: Path) -> None:
    entry = ResultEntry.create(
        "lab", result_root=tmp_path / "results", database_root=tmp_path / "database"
    )
    entry.setup.add_component("J1", kind="amplifier/jpa")
    entry.setup.add_component("Q1", kind="qubit/fluxonium", resonator="J1")
    point = entry.new_point("working")
    with pytest.raises(RoleResolutionError) as error:
        point.resolve()
    assert error.value.role == "resonator"
    assert error.value.required_kind == "resonator"
    assert "J1" in error.value.reason


@pytest.mark.parametrize("kind", ["qubit/transmon", "qubit/fluxonium"])
def test_global_flux_values_do_not_follow_local_unit_or_setup_general(
    tmp_path: Path, kind: str
) -> None:
    entry = ResultEntry.create(
        "lab", result_root=tmp_path / "results", database_root=tmp_path / "database"
    )
    entry.setup.add_component(
        "Q1", kind=kind, flux_unit="V", flux_half=-0.0108, t2r=12.0, t2e=20.0
    )
    entry.setup.general.flux_unit = "A"
    entry.setup.general.flux_value = 0.003
    point = entry.new_point("working")
    with pytest.raises(AttributeError):
        _ = point.general.flux_value
    with point.edit() as draft:
        draft.set("general.flux_unit", "A")
        draft.set("general.flux_value", -0.001)
        draft.set("Q1.flux_int", 0.002)
    point.refresh()
    assert point.general.flux_unit == "A"
    assert point.general.flux_value == -0.001
    assert point.Q1.flux_unit == "V"
    assert point.Q1.flux_half == -0.0108
    assert point.Q1.flux_int == 0.002
    assert point.Q1.t2r == 12.0
    assert point.Q1.t2e == 20.0
    assert entry.setup.general.flux_value == 0.003
    clone = entry.new_point("clone", clone_from="working")
    assert clone.general.flux_unit == "A"
    assert clone.general.flux_value == -0.001
    assert clone.Q1.flux_unit == "V"
    with pytest.raises(ValidationError):
        point.general.flux_unit = "mA"
    assert point.general.flux_unit == "A"


def test_jpa_native_pump_and_flux_values_survive_seed_and_reload(
    tmp_path: Path,
) -> None:
    entry = ResultEntry.create(
        "lab", result_root=tmp_path / "results", database_root=tmp_path / "database"
    )
    entry.setup.add_component(
        "J1",
        kind="amplifier/jpa",
        pump_freq=8000.0,
        pump_power=-12.5,
        flux=0.01,
        flux_unit="A",
    )
    point = entry.new_point("working")
    point.J1.flux_unit = "V"
    point.J1.pump_power = -10.0
    point.refresh()
    assert point.J1.pump_freq == 8000.0
    assert point.J1.pump_power == -10.0
    assert point.J1.flux == 0.01
    assert entry.setup.J1.pump_power == -12.5
    assert entry.setup.J1.flux_unit == "A"
    with pytest.raises(ValidationError):
        point.J1.pump_power = float("inf")
    assert point.J1.pump_power == -10.0


def test_readout_module_slot_is_an_unresolved_path_not_a_component_ref(
    tmp_path: Path,
) -> None:
    entry = ResultEntry.create(
        "lab", result_root=tmp_path / "results", database_root=tmp_path / "database"
    )
    entry.setup.add_component(
        "Q1", kind="qubit/transmon", module={"readout": "R1.readout.dpm"}
    )
    point = entry.new_point("working")
    slots = point.Q1.module
    assert isinstance(slots, FieldView)
    assert slots.readout == "R1.readout.dpm"
    slots.readout = "unresolved.module"
    with point.edit() as draft:
        draft.set("Q1.module.x180", "Q1.pulse.pi")
    with pytest.raises(ValidationError):
        slots.readout = 3
    point.refresh()
    assert slots.readout == "unresolved.module"
    assert slots.x180 == "Q1.pulse.pi"
    setup_slots = entry.setup.Q1.module
    assert isinstance(setup_slots, FieldView)
    assert setup_slots.readout == "R1.readout.dpm"


def test_nested_qubit_wiring_reads_isolated_maps_and_edits_channels(
    tmp_path: Path,
) -> None:
    entry = ResultEntry.create(
        "lab", result_root=tmp_path / "results", database_root=tmp_path / "database"
    )
    entry.setup.add_component(
        "Q1", kind="qubit/fluxonium", wiring={"L01": {"ch": 2}, "flux": {"ch": 14}}
    )
    point = entry.new_point("working")
    line = point.Q1.wiring.L01
    assert line == {"ch": 2}
    assert isinstance(line, dict)
    line["ch"] = 99
    assert point.Q1.wiring.L01 == {"ch": 2}
    with point.edit() as draft:
        draft.set("Q1.wiring.L01.ch", 3)
    point.refresh()
    assert point.Q1.wiring.L01 == {"ch": 3}
    assert point.Q1.wiring.flux == {"ch": 14}
    assert entry.setup.Q1.wiring.L01 == {"ch": 2}


@pytest.mark.parametrize("channel", [-1, None, True, "3", 3.5])
def test_invalid_channel_preserves_draft_and_allows_other_line_edit(
    tmp_path: Path, channel: object
) -> None:
    entry = ResultEntry.create(
        "lab", result_root=tmp_path / "results", database_root=tmp_path / "database"
    )
    entry.setup.add_component(
        "Q1", kind="qubit/transmon", wiring={"L01": {"ch": 2}, "flux": {"ch": 14}}
    )
    point = entry.new_point("working")
    before = point.meta("Q1.wiring.L01.ch")
    with point.edit() as draft:
        with pytest.raises(ValidationError):
            draft.Q1.wiring.L01 = {"ch": channel}
        assert draft.Q1.wiring.L01 == {"ch": 2}
        draft.set("Q1.wiring.flux.ch", 15)
    point.refresh()
    assert point.Q1.wiring.L01 == {"ch": 2}
    assert point.meta("Q1.wiring.L01.ch") == before
    assert point.Q1.wiring.flux == {"ch": 15}
