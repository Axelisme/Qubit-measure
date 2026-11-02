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

    def initial_values(
        captured: FormEnv, defaults: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
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
    form = Form(Scan, {}, initial_values=lambda env, defaults: {"absent": 3.0})
    with pytest.raises(ValueError, match="absent"):
        form.default_input(FormEnv())


class Tone(ConfigBase):
    """One fixed name for factory source admission."""

    name: str


class Tones(ConfigBase):
    """A fixed complete identity list."""

    tones: list[Tone]
    participants: list[int] = [0]


def test_factory_patches_detached_raw_defaults_without_leaking_unreturned_edits() -> (
    None
):
    class GroupedScan(Scan):
        tones: list[Tone]
        participants: list[int] = [0]

    env = FormEnv()
    received: list[dict[str, JsonValue]] = []

    def initial_values(
        captured: FormEnv, defaults: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        assert captured is env
        assert defaults == {
            "scan": {"kind": "linrange", "start": 1.0, "stop": 2.0, "n": 5},
            "count": {"expr": "2 * 4"},
            "position": 3.0,
            "tones": [{"name": "Q1"}, {"name": "Q2"}],
            "participants": [0],
        }
        received.append(defaults)
        tones = defaults["tones"]
        assert isinstance(tones, list)
        tone = tones[0]
        assert isinstance(tone, dict)
        tone["name"] = "patched"
        defaults["participants"] = [1]
        defaults["count"] = 99
        return {"tones": [tone]} if len(received) == 1 else {}

    form = Form(
        GroupedScan,
        {
            "scan": Range(1.0, 2.0, expts=5),
            "count": Expr("2 * 4"),
            "position": 3.0,
            "tones": [{"name": "Q1"}, {"name": "Q2"}],
        },
        initial_values=initial_values,
    )
    first = form.default_input(env)
    fresh = form.default_input(env)

    assert first.data == {
        "scan": {"kind": "linrange", "start": 1.0, "stop": 2.0, "n": 5},
        "count": 8,
        "position": 3.0,
        "tones": [{"name": "patched"}],
        "participants": [0],
    }
    assert fresh.data == {**first.data, "tones": [{"name": "Q1"}, {"name": "Q2"}]}
    received[0].clear()
    assert first.data["tones"] == [{"name": "patched"}]
    assert len(received) == 2


def test_factory_values_still_pass_fixed_identity_admission() -> None:
    form = Form(
        Tones,
        {},
        fixed_elements=(FixedElements(("tones",), "name", ("participants",)),),
        initial_values=lambda env, defaults: {
            "tones": [{"name": "same"}, {"name": "same"}]
        },
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
        initial_values=lambda env, defaults: {
            "pulse": {"kind": "ref", "name": "Q.pulses.probe"}
        },
    )
    with pytest.raises(ValueError, match="library"):
        form.default_input(FormEnv())


def test_factory_invalid_values_remain_required_errors_and_seeds_are_detached() -> None:
    seeds: dict[str, JsonValue] = {"position": "invalid"}
    form = Form(Scan, {}, initial_values=lambda env, defaults: MappingProxyType(seeds))
    env = FormEnv()
    input = form.default_input(env)
    seeds["position"] = 2.0
    result = FormState(input).result(env)
    assert isinstance(result, FormErrors)
    assert ("position",) in result.messages
    assert FormState(form.default_input(env)).preview(env).values[("position",)] == 2.0


def test_factory_source_fault_is_not_replaced_by_static_defaults() -> None:
    def unavailable(
        env: FormEnv, defaults: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        raise OSError("Captured coordinate unavailable")

    form = Form(Scan, {}, initial_values=unavailable)
    with pytest.raises(OSError, match="Captured coordinate unavailable"):
        form.default_input(FormEnv())
