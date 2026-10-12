"""Core-derived forms expose defaults, validation and typed cfg conversion."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated

import pytest
from pydantic import BaseModel, Field
from zcu_tools import cfg_model as nodes
from zcu_tools.cfg_model import ConfigBase, Expr, FieldMeta
from zcu_tools.gui.app.measure import adapter as adapter_api
from zcu_tools.gui.form import FormEnv, FormErrors, FormModel, FormState
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import SavedPoint
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.program.v2.sweep import SweepCfg
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument
from zcu_tools.resources.entry.schema import PointDocument, PointGeneral


class SweepSection(ConfigBase):
    """A sweep section is declared once, by the core."""

    freq: Annotated[SweepCfg, FieldMeta(label="Frequency", unit="MHz")]


class ScalarCfg(ConfigBase):
    """A synthetic cfg exercises section, omission and locked scalar rules."""

    sweep: SweepSection
    count: Annotated[
        int, Field(ge=1), FieldMeta(label="Repeats", description="Repeat count.")
    ] = 3
    amount: float = 2.0
    hidden: float = 4.0
    optional: str | None = None
    text: str = "ordinary"


class DeviceCfg(ConfigBase):
    """Core device names remain independent of aliases and sessions."""

    flux_dev: str
    optional_dev: str | None = None


class Extras(BaseModel):
    """A GUI-only value is lowered without changing the core declaration."""

    multiplier: int = 2


def require_result(state: FormState[FormModel], env: FormEnv) -> FormModel:
    """Return an ordinary successful form result, with useful failure output."""
    result = state.result(env)
    if isinstance(result, FormErrors):
        pytest.fail(str(result.messages))
    return result


def test_core_tree_generates_schema_defaults_and_typed_cfg() -> None:
    assert hasattr(adapter_api, "Form"), "Missing core-derived Form"
    form = adapter_api.Form(
        ScalarCfg,
        {
            "sweep": {"freq": nodes.Range.around("10", halfwidth="2", expts=5)},
            "count": nodes.Locked(7),
            "amount": Expr("2 * 3"),
            "hidden": nodes.Hidden(),
            "text": "$literal",
        },
    )
    env = FormEnv()
    base = form.default_input(env)
    assert list(form.model.model_fields) == [
        "sweep",
        "count",
        "amount",
        "optional",
        "text",
    ]
    assert "optional" not in base.data
    assert base.data["amount"] == {"expr": "2 * 3"}
    state = FormState(base)
    cfg = form.to_cfg(require_result(state, env), env)
    assert cfg == ScalarCfg(
        sweep=SweepSection(freq=SweepCfg(start=8, stop=12, expts=5, step=1)),
        count=7,
        amount=6,
        text="$literal",
    )
    schema = form.model.input_schema(env)
    properties = schema["properties"]
    assert isinstance(properties, Mapping)
    count_schema = properties["count"]
    assert isinstance(count_schema, Mapping)
    assert count_schema["title"] == "Repeats"
    assert count_schema["description"] == "Repeat count."
    state.write({("count",): 8}, env)
    assert isinstance(state.result(env), FormErrors)


def test_range_single_point_conversion_reports_core_error_at_sweep_path() -> None:
    assert hasattr(adapter_api, "Form"), "Missing core-derived Form"
    form = adapter_api.Form(
        ScalarCfg, {"sweep": {"freq": nodes.Range.between(1, 2, expts=1)}}
    )
    env = FormEnv()
    result = form.to_cfg(require_result(FormState(form.default_input(env)), env), env)
    assert isinstance(result, FormErrors)
    assert ("sweep", "freq") in result.messages


@pytest.mark.parametrize("scenario", ["unknown", "hidden_required"])
def test_invalid_authored_tree_fails_at_construction(scenario: str) -> None:
    assert hasattr(adapter_api, "Form"), "Missing core-derived Form"
    tree: dict[str, nodes.FormNode] = (
        {"unknown": 1} if scenario == "unknown" else {"sweep": nodes.Hidden()}
    )
    with pytest.raises(ValueError, match="unknown|sweep"):
        adapter_api.Form(ScalarCfg, tree)


def alias_env() -> FormEnv:
    """Capture aliases without connected drivers, point I/O or hardware."""
    library = LibrarySnapshot(
        ModuleLibraryDocument(format="zcu.module-library", format_version="1.0")
    )
    parameters = ParameterContext(
        point=PointDocument(
            format="zcu.parameter-container",
            format_version="1.0",
            general=PointGeneral(created_at="2026-10-13T00:00:00Z"),
        ),
        library=library,
        target=SavedPoint(entry_id="synthetic", point="synthetic"),
        roles=None,
        storage=None,
    ).bind_aliases((AliasRow("flux_dev", "yoko"), AliasRow("missing", "")), {})
    return FormEnv(parameters=parameters, library=library)


def test_alias_name_resolves_direct_name_and_missing_required_is_invalid() -> None:
    assert hasattr(adapter_api, "Form"), "Missing core-derived Form"
    env = alias_env()
    form = adapter_api.Form(
        DeviceCfg,
        {
            "flux_dev": nodes.AliasName("flux_dev", choices_source=None),
            "optional_dev": nodes.AliasName("missing", choices_source=None),
        },
    )
    assert form.default_input(env).data == {"flux_dev": "yoko"}
    assert form.to_cfg(
        require_result(FormState(form.default_input(env)), env), env
    ) == DeviceCfg(flux_dev="yoko")
    missing = adapter_api.Form(
        DeviceCfg, {"flux_dev": nodes.AliasName("missing", choices_source=None)}
    )
    state = FormState(missing.default_input(env))
    result = state.result(env)
    assert isinstance(result, FormErrors)
    assert ("flux_dev",) in result.messages


def lower_extras(extras: Extras, cfg: ScalarCfg) -> ScalarCfg:
    """Multiply a typed core value using a GUI-only input."""
    return cfg.model_copy(update={"amount": cfg.amount * extras.multiplier})


def test_extras_lower_receives_validated_named_models() -> None:
    assert hasattr(adapter_api, "Form"), "Missing core-derived Form"
    form = adapter_api.Form(
        ScalarCfg,
        {"sweep": {"freq": nodes.Range.between(0, 2, expts=3)}},
        extras=Extras,
        lower=lower_extras,
    )
    env = FormEnv()
    cfg = form.to_cfg(require_result(FormState(form.default_input(env)), env), env)
    assert isinstance(cfg, ScalarCfg)
    assert cfg.amount == 4


def test_loaded_cfg_with_extras_must_roundtrip_before_accepting_input() -> None:
    form = adapter_api.Form(
        ScalarCfg,
        {"sweep": {"freq": nodes.Range.between(0, 2, expts=3)}},
        extras=Extras,
        lower=lower_extras,
    )
    env = FormEnv()
    state = FormState(form.default_input(env))
    current = require_result(state, env)
    native = form.to_cfg(current, env)
    if isinstance(native, FormErrors):
        pytest.fail(str(native.messages))
    before = state.input.data
    assert hasattr(adapter_api, "CfgInputRejected"), "Missing directed Load rejection"
    rejected = form.cfg_input(native, current=current, env=env)
    assert isinstance(rejected, adapter_api.CfgInputRejected)
    assert rejected.differences[0].path == ("amount",)
    assert rejected.differences[0].loaded == 4.0
    assert rejected.differences[0].would_run == 8.0
    assert state.input.data == before
    state.write({("extras", "multiplier"): 1}, env)
    current = require_result(state, env)
    native = form.to_cfg(current, env)
    if isinstance(native, FormErrors):
        pytest.fail(str(native.messages))
    restored = form.cfg_input(native, current=current, env=env)
    if isinstance(restored, adapter_api.CfgInputRejected):
        pytest.fail(str(restored.errors.messages))
    restored_state = FormState(restored)
    assert restored.data["extras"] == {"multiplier": 1}
    assert form.to_cfg(require_result(restored_state, env), env) == native
