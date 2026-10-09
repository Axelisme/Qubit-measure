"""Public JPA selector lowering with cached, disconnected device snapshots."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy

import pytest
from zcu_tools.device import (
    DeviceInfo,
    RohdeSchwarzSGS100AInfo,
    YOKOGS200Info,
)

from zcu_lab.v2._support.measure.jpa_shared import (
    lower_jpa_flux_dev,
    lower_jpa_rf_dev,
    lower_jpa_rf_output_dev,
    lower_jpa_rf_power_dev,
)

LowerJpa = Callable[
    [Mapping[str, object], Mapping[str, DeviceInfo]], dict[str, dict[str, str]]
]

LOWERERS = (
    pytest.param(lower_jpa_rf_dev, "jpa_rf_dev", "frequency", id="frequency"),
    pytest.param(lower_jpa_rf_power_dev, "jpa_rf_dev", "power", id="power"),
    pytest.param(lower_jpa_rf_output_dev, "jpa_rf_dev", "output", id="output"),
    pytest.param(lower_jpa_flux_dev, "jpa_flux_dev", "flux", id="flux"),
)


def _supported_info(role_key: str) -> DeviceInfo:
    if role_key == "jpa_flux_dev":
        return YOKOGS200Info(address="disconnected::flux")
    return RohdeSchwarzSGS100AInfo(address="disconnected::rf")


def _unsupported_info(role_key: str) -> DeviceInfo:
    if role_key == "jpa_flux_dev":
        return RohdeSchwarzSGS100AInfo(address="disconnected::wrong-knob")
    return YOKOGS200Info(address="disconnected::wrong-knob")


@pytest.mark.parametrize(("lower", "role_key", "knob"), LOWERERS)
def test_lowering_returns_only_selected_patch_without_mutating_inputs(
    lower: LowerJpa, role_key: str, knob: str
) -> None:
    del knob
    raw_cfg: dict[str, object] = {
        "dev": {role_key: "selected", "unrelated": "other"},
        "settings": {"unchanged": True},
    }
    snapshot: dict[str, DeviceInfo] = {
        "selected": _supported_info(role_key),
        "other": _unsupported_info(role_key),
    }
    original_cfg = deepcopy(raw_cfg)
    original_snapshot = {name: info.model_dump() for name, info in snapshot.items()}

    assert lower(raw_cfg, snapshot) == {"selected": {"label": role_key}}
    assert raw_cfg == original_cfg
    assert {
        name: info.model_dump() for name, info in snapshot.items()
    } == original_snapshot


@pytest.mark.parametrize(("lower", "role_key", "knob"), LOWERERS)
@pytest.mark.parametrize("dev_section", [None, 0, "selected", []])
def test_lowering_rejects_missing_or_nonmapping_device_section(
    lower: LowerJpa, role_key: str, knob: str, dev_section: object
) -> None:
    del role_key, knob
    raw_cfg: dict[str, object] = {}
    if dev_section is not None:
        raw_cfg["dev"] = dev_section

    with pytest.raises(ValueError, match="cfg has no 'dev' section"):
        lower(raw_cfg, {})


@pytest.mark.parametrize(("lower", "role_key", "knob"), LOWERERS)
@pytest.mark.parametrize("selected", [None, "", 3, False])
def test_lowering_rejects_missing_empty_or_nonstring_role_selector(
    lower: LowerJpa, role_key: str, knob: str, selected: object
) -> None:
    del knob
    selectors: dict[str, object] = {"unrelated": "other"}
    if selected is not None:
        selectors[role_key] = selected
    raw_cfg: dict[str, object] = {"dev": selectors}

    with pytest.raises(ValueError, match=f"dev\\.{role_key}.*is empty"):
        lower(raw_cfg, {})


@pytest.mark.parametrize(("lower", "role_key", "knob"), LOWERERS)
def test_lowering_checks_selected_snapshot_before_knob_capability(
    lower: LowerJpa, role_key: str, knob: str
) -> None:
    del knob
    raw_cfg: dict[str, object] = {"dev": {role_key: "missing"}}
    snapshot: dict[str, DeviceInfo] = {"other": _unsupported_info(role_key)}

    with pytest.raises(ValueError, match="'missing' not found in the device snapshot"):
        lower(raw_cfg, snapshot)


@pytest.mark.parametrize(("lower", "role_key", "knob"), LOWERERS)
def test_lowering_rejects_selected_device_without_required_knob(
    lower: LowerJpa, role_key: str, knob: str
) -> None:
    raw_cfg: dict[str, object] = {"dev": {role_key: "selected"}}
    snapshot: dict[str, DeviceInfo] = {"selected": _unsupported_info(role_key)}

    with pytest.raises(ValueError, match=f"does not support the {knob} knob"):
        lower(raw_cfg, snapshot)
