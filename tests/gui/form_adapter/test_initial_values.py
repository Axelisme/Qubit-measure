"""Captured initial-value factories use the normal Form input and seed contract."""

from types import MappingProxyType

import pytest
from pydantic import JsonValue
from zcu_tools.cfg_model import ConfigBase, Expr, Range
from zcu_tools.device.base import BaseDeviceInfo
from zcu_tools.gui.form import FormEnv, FormErrors, FormState
from zcu_tools.gui.form_adapter import FixedElements, Form
from zcu_tools.program.v2.modules import PulseCfg
from zcu_tools.program.v2.sweep import SweepCfg


class CapturedInfo(BaseDeviceInfo):
    """Synthetic detached coordinate information; no driver is constructed."""

    value: float


class Scan(ConfigBase):
    """A required scan with one ordinary static setting."""

    scan: SweepCfg
    count: int = 7
    position: float


def test_factory_seeds_same_capture_without_replacing_declaration() -> None:
    info = CapturedInfo(address="offline", type="synthetic", value=3.0)
    env = FormEnv(devices={"bias": info})
    captures: list[FormEnv] = []

    def initial_values(captured: FormEnv) -> dict[str, JsonValue]:
        assert captured is env
        captures.append(captured)
        coordinate = captured.devices["bias"]
        if not isinstance(coordinate, CapturedInfo):
            raise TypeError("Expected synthetic coordinate info")
        return {"position": coordinate.value}

    form = Form(
        Scan,
        {"scan": Range(None, None, expts=5), "count": Expr("2 * 4")},
        initial_values=initial_values,
    )
    input = form.default_input(env)
    state = FormState(input)
    result = state.result(env)
    assert isinstance(result, FormErrors)
    assert set(result.messages) == {("scan", "start"), ("scan", "stop")}
    state.write({("scan", "start"): 1.0, ("scan", "stop"): 2.0}, env)
    result = state.result(env)
    if isinstance(result, FormErrors):
        pytest.fail(str(result.messages))
    cfg = form.to_cfg(result, env)
    if isinstance(cfg, FormErrors):
        pytest.fail(str(cfg.messages))
    assert (cfg.position, cfg.count, cfg.scan.expts) == (3.0, 8, 5)
    rows = {row.path: row for row in form.diagnostics(input, env, None)}
    assert rows[("count",)].seed_source == "2 * 4"
    assert rows[("count",)].seed_status == "available"
    assert rows[("position",)].seed_status == "literal"
    assert info.value == 3.0
    assert captures == [env]


def test_factory_rejects_unknown_fields_before_building_input() -> None:
    form = Form(Scan, {}, initial_values=lambda env: {"absent": 3.0})
    with pytest.raises(ValueError, match="absent"):
        form.default_input(FormEnv())


class Tone(ConfigBase):
    """One fixed name for factory source admission."""

    name: str


class Tones(ConfigBase):
    """A fixed complete identity list."""

    tones: list[Tone]
    participants: list[int] = [0]


def test_factory_values_still_pass_fixed_identity_admission() -> None:
    form = Form(
        Tones,
        {},
        fixed_elements=(FixedElements(("tones",), "name", ("participants",)),),
        initial_values=lambda env: {"tones": [{"name": "same"}, {"name": "same"}]},
    )
    with pytest.raises(ValueError, match="same"):
        form.default_input(FormEnv())


class PulsePlan(ConfigBase):
    """A required pulse consumes a captured library for refs."""

    pulse: PulseCfg


def test_factory_library_reference_still_requires_captured_source() -> None:
    form = Form(
        PulsePlan,
        {},
        initial_values=lambda env: {"pulse": {"kind": "ref", "name": "Q.pulses.probe"}},
    )
    with pytest.raises(ValueError, match="library"):
        form.default_input(FormEnv())


def test_factory_invalid_values_remain_required_errors_and_seeds_are_detached() -> None:
    seeds: dict[str, JsonValue] = {"position": "invalid"}
    form = Form(Scan, {}, initial_values=lambda env: MappingProxyType(seeds))
    env = FormEnv()
    input = form.default_input(env)
    seeds["position"] = 2.0
    result = FormState(input).result(env)
    assert isinstance(result, FormErrors)
    assert ("position",) in result.messages
    assert FormState(form.default_input(env)).preview(env).values[("position",)] == 2.0


def test_factory_source_fault_is_not_replaced_by_static_defaults() -> None:
    def unavailable(env: FormEnv) -> dict[str, JsonValue]:
        raise OSError("Captured coordinate unavailable")

    form = Form(Scan, {}, initial_values=unavailable)
    with pytest.raises(OSError, match="Captured coordinate unavailable"):
        form.default_input(FormEnv())
