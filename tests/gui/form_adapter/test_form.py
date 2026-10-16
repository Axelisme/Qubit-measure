"""Core-derived forms expose defaults, validation and typed cfg conversion."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated

import pytest
from pydantic import BaseModel, Field, JsonValue
from zcu_tools import cfg_model as nodes
from zcu_tools.cfg_model import ConfigBase, Expr, FieldMeta
from zcu_tools.gui import form_adapter
from zcu_tools.gui.form import FormEnv, FormErrors, FormInput, FormModel, FormState
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


class DefaultedCfg(ConfigBase):
    """Static defaults belong to core unless a form authors an expression."""

    amount: float = 2.0
    count: int = Field(default_factory=lambda: 3, ge=1)


@pytest.mark.parametrize("expression", ["$unbound.value", "'not a number'"])
def test_failed_expression_defaults_require_input_without_changing_core(
    expression: str,
) -> None:
    env = FormEnv()
    form = form_adapter.Form(
        DefaultedCfg, {"amount": Expr(expression), "count": Expr(expression)}
    )
    state = FormState(form.default_input(env))
    assert state.input.data == {}
    result = state.result(env)
    assert isinstance(result, FormErrors), result
    assert result.messages == {
        ("amount",): "Field required",
        ("count",): "Field required",
    }
    control = form_adapter.Form(DefaultedCfg, {})
    assert (
        control.to_cfg(require_result(FormState(control.default_input(env)), env), env)
        == DefaultedCfg()
    )
    state.write({("amount",): 8.0, ("count",): 4}, env)
    assert form.to_cfg(require_result(state, env), env) == DefaultedCfg(
        amount=8, count=4
    )


def test_successful_expression_defaults_are_direct_native_values() -> None:
    env = FormEnv()
    form = form_adapter.Form(DefaultedCfg, {"amount": Expr("6 * 2")})
    state = FormState(form.default_input(env))
    assert state.input.data == {"amount": 12, "count": 3}
    assert form.to_cfg(require_result(state, env), env) == DefaultedCfg(amount=12)


@pytest.mark.parametrize("endpoint", ["start", "stop"])
@pytest.mark.parametrize("expression", ["$unbound.value", "'not a number'"])
def test_failed_range_endpoint_requires_input(endpoint: str, expression: str) -> None:
    env = FormEnv()
    node = nodes.Range(
        Expr(expression) if endpoint == "start" else 2.0,
        Expr(expression) if endpoint == "stop" else 8.0,
        4,
    )
    form = form_adapter.Form(ScalarCfg, {"sweep": {"freq": node}})
    state = FormState(form.default_input(env))
    result = state.result(env)
    assert isinstance(result, FormErrors), "Failed Range Expr must not use 0/1"
    assert result.messages == {("sweep", "freq", endpoint): "Field required"}
    state.write({("sweep", "freq", endpoint): 3.0 if endpoint == "start" else 9.0}, env)
    cfg = form.to_cfg(require_result(state, env), env)
    assert isinstance(cfg, ScalarCfg), cfg
    assert cfg.sweep.freq.start == (3.0 if endpoint == "start" else 2.0)
    assert cfg.sweep.freq.stop == (8.0 if endpoint == "start" else 9.0)
    assert cfg.sweep.freq.expts == 4


@pytest.mark.parametrize(
    "factory", [nodes.Range, nodes.Range.between], ids=["direct", "between"]
)
@pytest.mark.parametrize(
    "blank", [("start",), ("stop",), ("start", "stop")], ids=["start", "stop", "both"]
)
def test_blank_range_endpoints_remain_required_when_input_keys_are_absent(
    factory: Callable[[float | None, float | None, int], nodes.Range],
    blank: tuple[str, ...],
) -> None:
    env = FormEnv()
    node = factory(
        None if "start" in blank else 2.0, None if "stop" in blank else 8.0, 4
    )
    form = form_adapter.Form(ScalarCfg, {"sweep": {"freq": node}})
    state = FormState(form.default_input(env))
    initial = state.result(env)
    assert isinstance(initial, FormErrors), (
        "An authored blank endpoint must block submission"
    )
    assert set(initial.messages) == {("sweep", "freq", name) for name in blank}

    # A partial draft must not revive the inherited LinRange 0/1 endpoints.
    partial: dict[str, JsonValue] = {"kind": "linrange", "n": 4}
    if "start" not in blank:
        partial["start"] = 2.0
    if "stop" not in blank:
        partial["stop"] = 8.0
    state = FormState(FormInput(form.model, {"sweep": {"freq": partial}}))
    missing = state.result(env)
    assert isinstance(missing, FormErrors), "Absent blank endpoints must not use 0/1"
    assert missing.messages == {
        ("sweep", "freq", name): "Field required" for name in blank
    }

    state.write(
        {("sweep", "freq", name): 3.0 if name == "start" else 9.0 for name in blank},
        env,
    )
    cfg = form.to_cfg(require_result(state, env), env)
    if isinstance(cfg, FormErrors):
        pytest.fail(str(cfg.messages))
    assert cfg.sweep.freq.start == (3.0 if "start" in blank else 2.0)
    assert cfg.sweep.freq.stop == (9.0 if "stop" in blank else 8.0)
    assert cfg.sweep.freq.expts == 4
    assert cfg.sweep.freq.step == (cfg.sweep.freq.stop - cfg.sweep.freq.start) / 3

    # Requiredness belongs to this authored form, not the shared range class.
    control = form_adapter.Form(
        ScalarCfg, {"sweep": {"freq": nodes.Range(2.0, 8.0, 4)}}
    )
    control_state = FormState(
        FormInput(control.model, {"sweep": {"freq": {"kind": "linrange", "n": 4}}})
    )
    control_cfg = control.to_cfg(require_result(control_state, env), env)
    if isinstance(control_cfg, FormErrors):
        pytest.fail(str(control_cfg.messages))
    assert control_cfg.sweep.freq.start == 0.0
    assert control_cfg.sweep.freq.stop == 1.0


class PreciseSweepCfg(ConfigBase):
    """Sweep precision describes endpoint float controls, not the range object."""

    sweep: Annotated[SweepCfg, FieldMeta(label="Flux", unit="A", decimals=4)]


@pytest.mark.parametrize("authored", [False, True])
def test_sweep_decimals_decorate_only_float_endpoints(authored: bool) -> None:
    form = form_adapter.Form(
        PreciseSweepCfg,
        {"sweep": nodes.Range.between("2 + 1", 9.0, 4)} if authored else {},
    )
    env = FormEnv()
    state = FormState(form.default_input(env))
    description = state.describe(("sweep",), env)
    sweep = description.target
    assert sweep is not None and sweep.decimals is None
    assert sweep.title == "Flux" and sweep.unit == "A"
    fields = {field.path[-1]: field for field in description.children}
    assert fields["start"].decimals == fields["stop"].decimals == 4
    assert fields["n"].decimals is None and fields["n"].title == "Expts"


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
    assert hasattr(form_adapter, "Form"), "Missing core-derived Form"
    form = form_adapter.Form(
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
    assert base.data["amount"] == 6
    state = FormState(base)
    cfg = form.to_cfg(require_result(state, env), env)
    assert cfg == ScalarCfg(
        sweep=SweepSection(freq=SweepCfg(start=8, stop=12, expts=5, step=1)),
        count=7,
        amount=6,
        text="$literal",
    )
    count = state.describe(("count",), env).target
    assert count is not None
    assert count.title == "Repeats" and count.description == "Repeat count."
    state.write({("count",): 8}, env)
    assert isinstance(state.result(env), FormErrors)


def test_range_single_point_conversion_reports_core_error_at_sweep_path() -> None:
    assert hasattr(form_adapter, "Form"), "Missing core-derived Form"
    form = form_adapter.Form(
        ScalarCfg, {"sweep": {"freq": nodes.Range.between(1, 2, expts=1)}}
    )
    env = FormEnv()
    result = form.to_cfg(require_result(FormState(form.default_input(env)), env), env)
    assert isinstance(result, FormErrors)
    assert ("sweep", "freq") in result.messages


@pytest.mark.parametrize("scenario", ["unknown", "hidden_required"])
def test_invalid_authored_tree_fails_at_construction(scenario: str) -> None:
    assert hasattr(form_adapter, "Form"), "Missing core-derived Form"
    tree: dict[str, nodes.FormNode] = (
        {"unknown": 1} if scenario == "unknown" else {"sweep": nodes.Hidden()}
    )
    with pytest.raises(ValueError, match="unknown|sweep"):
        form_adapter.Form(ScalarCfg, tree)


def alias_env() -> FormEnv:
    """Capture aliases without connected drivers, point I/O or hardware."""
    library = LibrarySnapshot(
        ModuleLibraryDocument(format="zcu.module-library", format_version="2.0")
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
    assert hasattr(form_adapter, "Form"), "Missing core-derived Form"
    env = alias_env()
    form = form_adapter.Form(
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
    missing = form_adapter.Form(
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
    assert hasattr(form_adapter, "Form"), "Missing core-derived Form"
    form = form_adapter.Form(
        ScalarCfg,
        {"sweep": {"freq": nodes.Range.between(0, 2, expts=3)}},
        extras=Extras,
        lower=lower_extras,
    )
    env = FormEnv()
    cfg = form.to_cfg(require_result(FormState(form.default_input(env)), env), env)
    assert isinstance(cfg, ScalarCfg)
    assert cfg.amount == 4


class NullableDeviceNames(ConfigBase):
    """Nullable and required are separate core constraints."""

    optional_dev: str | None = None
    nullable_required_dev: str | None


def test_nullable_alias_choices_accept_none_without_device_sources() -> None:
    env = FormEnv()
    form = form_adapter.Form(
        NullableDeviceNames,
        {
            "optional_dev": nodes.AliasName("missing"),
            "nullable_required_dev": nodes.AliasName("also_missing"),
        },
    )
    state = FormState(form.default_input(env))
    missing = state.result(env)
    assert isinstance(missing, FormErrors)
    assert ("nullable_required_dev",) in missing.messages
    state.write({("nullable_required_dev",): None}, env)
    result = require_result(state, env)
    assert form.to_cfg(result, env) == NullableDeviceNames(nullable_required_dev=None)
    assert form.model.choices(env) == {
        ("optional_dev",): (None,),
        ("nullable_required_dev",): (None,),
    }
    required = form_adapter.Form(DeviceCfg, {"flux_dev": nodes.AliasName("missing")})
    assert required.model.choices(env)[("flux_dev",)] == ()
    assert isinstance(FormState(required.default_input(env)).result(env), FormErrors)


class DisplayedCfg(ConfigBase):
    amplitude: Annotated[float, FieldMeta(label="Amplitude", unit="V", decimals=3)] = (
        0.5
    )


def test_field_meta_label_unit_and_decimals_reach_description() -> None:
    form = form_adapter.Form(DisplayedCfg, {})
    env = FormEnv()
    amplitude = FormState(form.default_input(env)).describe(("amplitude",), env).target
    assert amplitude is not None
    assert (amplitude.title, amplitude.unit, amplitude.decimals) == (
        "Amplitude",
        "V",
        3,
    )


class GroupedCfg(ConfigBase):
    amplitude: Annotated[float, FieldMeta(decimals=3, group="Advanced")] = 0.5


def test_field_meta_group_reaches_description() -> None:
    form = form_adapter.Form(GroupedCfg, {})
    env = FormEnv()
    amplitude = FormState(form.default_input(env)).describe(("amplitude",), env).target
    assert amplitude is not None and amplitude.group == "Advanced"


def test_locked_field_with_decimals_metadata_has_description() -> None:
    form = form_adapter.Form(DisplayedCfg, {"amplitude": nodes.Locked(0.25)})
    env = FormEnv()

    state = FormState(form.default_input(env))
    amplitude = state.describe(("amplitude",), env).target
    result = state.result(env)

    assert amplitude is not None and amplitude.fixed and amplitude.fixed_value == 0.25
    assert not isinstance(result, FormErrors)
    cfg = form.to_cfg(result, env)
    assert isinstance(cfg, DisplayedCfg) and cfg.amplitude == 0.25
