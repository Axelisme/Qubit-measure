"""Core-derived forms expose defaults, validation and typed cfg conversion."""

from __future__ import annotations

import typing
from collections.abc import Mapping
from typing import Annotated, Literal

import pytest
from pydantic import Field, JsonValue, create_model
from zcu_tools import cfg_model as nodes
from zcu_tools.cfg_model import ConfigBase, Expr, FieldMeta
from zcu_tools.gui import form_adapter
from zcu_tools.gui.form import FormEnv, FormErrors, FormInput, FormModel, FormState
from zcu_tools.gui.session.context_types import SavedPoint
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument
from zcu_tools.resources.entry.schema import PointDocument, PointGeneral
from zcu_tools.program.v2.sweep import SweepCfg


class SweepSection(ConfigBase):
    """A sweep section is declared once, by the core."""

    freq: Annotated[SweepCfg, FieldMeta(label="Frequency", unit="MHz")]


class FrequencyBranch(ConfigBase):
    """A native tagged section with an editable frequency sweep."""

    mode: Literal["frequency"] = "frequency"
    sweep: SweepCfg


class DelayBranch(ConfigBase):
    """A second native branch with the same sweep contract."""

    mode: Literal["delay"] = "delay"
    sweep: SweepCfg


class TaggedSweepCfg(ConfigBase):
    branch: Annotated[FrequencyBranch | DelayBranch, Field(discriminator="mode")]


class DirectSweepCfg(ConfigBase):
    branch: FrequencyBranch


@pytest.mark.parametrize("model", [DirectSweepCfg, TaggedSweepCfg])
@pytest.mark.parametrize(
    ("sweep", "fields"),
    [
        ({"kind": "linrange", "start": 10, "stop": 14, "n": 3}, ["start", "stop", "n"]),
        ({"kind": "center", "center": 12, "span": 4, "n": 3}, ["center", "span", "n"]),
        (
            {"kind": "linrange_step", "start": 10, "stop": 14, "step": 2},
            ["start", "stop", "step"],
        ),
        (
            {"kind": "center_step", "center": 12, "span": 4, "step": 2},
            ["center", "span", "step"],
        ),
    ],
)
def test_nested_sweep_projects_editable_coordinates_and_lowers_to_native(
    model: type[DirectSweepCfg] | type[TaggedSweepCfg],
    sweep: JsonValue,
    fields: list[str],
) -> None:
    env = FormEnv()
    form = form_adapter.Form(model, {})
    state = FormState(
        FormInput(
            form.model,
            {"branch": {"mode": "frequency", "sweep": sweep}},
        )
    )
    assert [
        child.name for child in state.describe(("branch", "sweep"), env).children
    ] == fields
    cfg = form.to_cfg(require_result(state, env), env)
    assert cfg == model(
        branch=FrequencyBranch(sweep=SweepCfg(start=10, stop=14, expts=3, step=2))
    )


class DefaultTaggedSweepCfg(ConfigBase):
    branch: Annotated[FrequencyBranch | DelayBranch, Field(discriminator="mode")] = (
        Field(
            default_factory=lambda: FrequencyBranch(
                sweep=SweepCfg(start=10, stop=14, expts=3, step=2)
            )
        )
    )


def test_native_tagged_default_projects_sweep_without_changing_coordinates() -> None:
    env = FormEnv()
    form = form_adapter.Form(DefaultTaggedSweepCfg, {})
    state = FormState(form.default_input(env))
    assert state.input.data == {
        "branch": {
            "mode": "frequency",
            "sweep": {"kind": "linrange", "start": 10, "stop": 14, "n": 3},
        }
    }
    assert form.to_cfg(require_result(state, env), env) == DefaultTaggedSweepCfg()


class OptionalTaggedSweepCfg(ConfigBase):
    branch: (
        Annotated[FrequencyBranch | DelayBranch, Field(discriminator="mode")] | None
    ) = None


def test_tagged_branch_selection_uses_shared_sweep_conversions_and_missing_defaults() -> (
    None
):
    env = FormEnv()
    form = form_adapter.Form(OptionalTaggedSweepCfg, {})
    state = FormState(form.default_input(env))
    state.select(("branch",), {"mode": "frequency"}, env)
    result = state.result(env)
    assert isinstance(result, FormErrors)
    assert set(result.messages) == {
        ("branch", "sweep", name) for name in ("start", "stop", "n")
    }
    state.write(
        {
            ("branch", "sweep", "start"): 10,
            ("branch", "sweep", "stop"): 14,
            ("branch", "sweep", "n"): 3,
        },
        env,
    )
    state.select(("branch", "sweep"), {"kind": "center_step"}, env)
    state.select(("branch",), {"mode": "delay"}, env)
    assert state.input.data["branch"] == {
        "mode": "delay",
        "sweep": {"kind": "center_step", "center": 12, "span": 4, "step": 2},
    }
    assert form.to_cfg(require_result(state, env), env) == OptionalTaggedSweepCfg(
        branch=DelayBranch(sweep=SweepCfg(start=10, stop=14, expts=3, step=2))
    )


class SharedSweepBranch(ConfigBase):
    mode: Literal["shared"] = "shared"
    sweep: form_adapter.SweepInput


class MixedSweepCfg(ConfigBase):
    branch: Annotated[SharedSweepBranch | DelayBranch, Field(discriminator="mode")]


def test_explicit_shared_sweep_inside_native_tagged_branch_keeps_declared_type() -> (
    None
):
    env = FormEnv()
    form = form_adapter.Form(MixedSweepCfg, {})
    state = FormState(
        FormInput(
            form.model,
            {
                "branch": {
                    "mode": "shared",
                    "sweep": {"kind": "linrange", "start": 10, "stop": 14, "n": 3},
                },
            },
        )
    )
    assert form.to_cfg(require_result(state, env), env) == MixedSweepCfg(
        branch=SharedSweepBranch(sweep=form_adapter.LinRange(start=10, stop=14, n=3))
    )


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


def test_field_order_is_local_partial_and_preserves_nested_order() -> None:
    env = FormEnv()
    tree = {"sweep": {"freq": nodes.Range(1.0, 2.0, expts=3)}}
    default = form_adapter.Form(ScalarCfg, tree)
    try:
        ordered = form_adapter.Form(ScalarCfg, tree, field_order=("amount", "sweep"))
    except TypeError as error:
        pytest.fail(f"A form must accept local field ordering: {error}")
    state = FormState(ordered.default_input(env))
    assert [child.path for child in state.describe((), env).children] == [
        ("amount",),
        ("sweep",),
        ("count",),
        ("hidden",),
        ("optional",),
        ("text",),
    ]
    assert [child.path for child in state.describe(("sweep",), env).children] == [
        ("sweep", "freq"),
    ]
    original = FormState(default.default_input(env))
    assert [child.path for child in original.describe((), env).children] == [
        ("sweep",),
        ("count",),
        ("amount",),
        ("hidden",),
        ("optional",),
        ("text",),
    ]
    assert ordered.to_cfg(require_result(state, env), env) == default.to_cfg(
        require_result(original, env), env
    )
    for invalid in (("missing",), ("amount", "amount")):
        with pytest.raises(ValueError, match="field_order"):
            form_adapter.Form(ScalarCfg, tree, field_order=invalid)


type StringMapping = Mapping[str, str]


class MappingSection(ConfigBase):
    """Typed mappings are whole data leaves, not scalar expression targets."""

    labels: Mapping[str, str] = Field(default_factory=dict)
    dictionary: dict[str, str] = Field(default_factory=dict)
    optional: Mapping[str, str] | None = None
    aliased: StringMapping = Field(default_factory=dict)


class MappingCfg(ConfigBase):
    """Nested declarations must identify the consuming mapping field."""

    section: MappingSection = Field(default_factory=MappingSection)


@pytest.mark.parametrize("field", ["labels", "dictionary", "optional", "aliased"])
def test_whole_mapping_expr_is_rejected_at_form_declaration(field: str) -> None:
    with pytest.raises(TypeError) as failure:
        form_adapter.Form(MappingCfg, {"section": {field: Expr("$labels")}})
    message = str(failure.value)
    assert "section" in message and field in message
    assert "Expr" in message and "str" in message
    assert "Mapping" in message or "dict" in message


@pytest.mark.parametrize(
    "annotation",
    [dict, Mapping, typing.Mapping, dict | None, Mapping | str, typing.Mapping | None],
)
def test_unparameterized_mapping_expr_is_rejected_at_form_declaration(
    annotation: object,
) -> None:
    core = create_model(
        "MappingDeclaration", __base__=ConfigBase, value=(annotation, ...)
    )
    with pytest.raises(TypeError, match="value: whole mapping Expr"):
        form_adapter.Form(core, {"value": Expr("$labels")})


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
    assert form.seed_sources[("amount",)].source == expression
    assert form.seed_sources[("amount",)].status == "missing"
    assert form.seed_sources[("count",)].status == "missing"
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


def test_successful_expression_defaults_keep_text_and_lower_to_native_values() -> None:
    env = FormEnv()
    form = form_adapter.Form(DefaultedCfg, {"amount": Expr("6 * 2")})
    state = FormState(form.default_input(env))
    assert state.input.data == {"amount": {"expr": "6 * 2"}, "count": 3}
    assert form.seed_sources[("amount",)].source == "6 * 2"
    assert form.seed_sources[("amount",)].status == "available"
    assert form.to_cfg(require_result(state, env), env) == DefaultedCfg(amount=12)


@pytest.fixture
def indexed_expression_env() -> FormEnv:
    """Capture list values, mapping values and their index/key sources."""
    return FormEnv(
        parameters=ParameterContext(
            point=PointDocument(
                format="zcu.parameter-container",
                format_version="2.0",
                schema_version=1,
                general=PointGeneral(
                    created_at="2026-10-08T00:00:00Z",
                    ext={
                        "channels": [2, 5],
                        "periods": {"bias": 4.0},
                        "device": "bias",
                        "index": 1,
                    },
                ),
            ),
            library=LibrarySnapshot(
                ModuleLibraryDocument(format="zcu.module-library", format_version="2.0")
            ),
            target=SavedPoint(entry_name="entry", point="synthetic"),
            roles=None,
        )
    )


def test_indexed_expression_defaults_preserve_full_sources_and_numeric_results(
    indexed_expression_env: FormEnv,
) -> None:
    env = indexed_expression_env
    expressions = {
        "amount": "max(1, min(9, general.ext.periods[general.ext.device] * 2))",
        "count": "general.ext.channels[general.ext.index]",
    }
    form = form_adapter.Form(
        DefaultedCfg, {name: Expr(text) for name, text in expressions.items()}
    )
    state = FormState(form.default_input(env))
    assert state.input.data == {
        name: {"expr": text} for name, text in expressions.items()
    }
    for name, text in expressions.items():
        assert form.seed_sources[(name,)].source == text
        assert form.seed_sources[(name,)].status == "available"
    preview = state.preview(env)
    assert preview.errors is None
    assert preview.values == {("amount",): 8.0, ("count",): 5}
    assert form.to_cfg(require_result(state, env), env) == DefaultedCfg(amount=8, count=5)


@pytest.mark.parametrize(
    "expression",
    [
        "general.ext.channels[2]",
        "general.ext.periods['missing']",
        "general.ext.channels[1.0]",
        "general.ext.periods[0]",
        "general.ext.device[0]",
        "general.ext.device + 'suffix'",
        "min(1)",
    ],
)
def test_failed_indexed_defaults_record_missing_source_and_require_input(
    indexed_expression_env: FormEnv, expression: str
) -> None:
    env = indexed_expression_env
    form = form_adapter.Form(DefaultedCfg, {"amount": Expr(expression)})
    state = FormState(form.default_input(env))
    assert state.input.data == {"count": 3}
    assert form.seed_sources[("amount",)].source == expression
    assert form.seed_sources[("amount",)].status == "missing"
    result = state.result(env)
    assert isinstance(result, FormErrors), result
    assert result.messages == {("amount",): "Field required"}
    preview = state.preview(env)
    assert preview.errors == result
    assert preview.values == {("count",): 3}


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
    "blank", [("start",), ("stop",), ("start", "stop")], ids=["start", "stop", "both"]
)
def test_blank_range_endpoints_remain_required_when_input_keys_are_absent(
    blank: tuple[str, ...],
) -> None:
    env = FormEnv()
    node = nodes.Range(
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

    # Even a literal-authored form requires endpoints absent from a restored draft.
    control = form_adapter.Form(
        ScalarCfg, {"sweep": {"freq": nodes.Range(2.0, 8.0, 4)}}
    )
    control_state = FormState(
        FormInput(control.model, {"sweep": {"freq": {"kind": "linrange", "n": 4}}})
    )
    missing_control = control_state.result(env)
    assert isinstance(missing_control, FormErrors)
    assert missing_control.messages == {
        ("sweep", "freq", name): "Field required" for name in ("start", "stop")
    }


def test_generated_sweep_fresh_input_requires_all_coordinates() -> None:
    env = FormEnv()
    form = form_adapter.Form(ScalarCfg, {})
    state = FormState(form.default_input(env))
    assert state.input.data["sweep"] == {"freq": {"kind": "linrange"}}
    missing = state.result(env)
    assert isinstance(missing, FormErrors)
    assert {
        ("sweep", "freq", name) for name in ("start", "stop", "n")
    } <= missing.messages.keys()
    state.write(
        {
            ("sweep", "freq", "start"): 2,
            ("sweep", "freq", "stop"): 8,
            ("sweep", "freq", "n"): 4,
        },
        env,
    )
    cfg = form.to_cfg(require_result(state, env), env)
    assert isinstance(cfg, ScalarCfg)
    assert cfg.sweep.freq == SweepCfg(start=2, stop=8, expts=4, step=2)


class PreciseSweepCfg(ConfigBase):
    """Sweep precision describes endpoint float controls, not the range object."""

    sweep: Annotated[SweepCfg, FieldMeta(label="Flux", unit="A", decimals=4)]


@pytest.mark.parametrize("authored", [False, True])
def test_sweep_decimals_decorate_only_float_endpoints(authored: bool) -> None:
    form = form_adapter.Form(
        PreciseSweepCfg,
        {"sweep": nodes.Range("2 + 1", 9.0, 4)} if authored else {},
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


class ScanPulseCfg(ConfigBase):
    """Run replaces frequency; gain remains an operator-owned required input."""

    freq: float = Field(ge=0, allow_inf_nan=False)
    gain: float


class ScanCfg(ConfigBase):
    """A synthetic nested cfg keeps hidden supply separate from user input."""

    pulse: ScanPulseCfg
    optional: float | None = 3.0


def test_hidden_value_supplies_required_nested_cfg_without_user_input() -> None:
    env = FormEnv()
    form = form_adapter.Form(ScanCfg, {"pulse": {"freq": nodes.Hidden(0.0)}})
    state = FormState(form.default_input(env))

    assert [field.path for field in state.describe(("pulse",), env).children] == [
        ("pulse", "gain")
    ]
    result = state.result(env)
    assert isinstance(result, FormErrors)
    assert result.messages == {("pulse", "gain"): "Field required"}

    state.write({("pulse", "gain"): 0.25}, env)
    assert form.to_cfg(require_result(state, env), env) == ScanCfg(
        pulse=ScanPulseCfg(freq=0.0, gain=0.25)
    )


@pytest.mark.parametrize("value", [-1.0, float("nan"), "not a frequency"])
def test_hidden_value_still_obeys_core_validation(value: float | str) -> None:
    env = FormEnv()
    form = form_adapter.Form(
        ScanCfg, {"pulse": {"freq": nodes.Hidden(value), "gain": 0.25}}
    )
    result = form.to_cfg(require_result(FormState(form.default_input(env)), env), env)
    assert isinstance(result, FormErrors)
    assert ("pulse", "freq") in result.messages


def test_hidden_explicit_none_overrides_core_default() -> None:
    env = FormEnv()
    form = form_adapter.Form(
        ScanCfg,
        {
            "pulse": {"freq": nodes.Hidden(0.0), "gain": 0.25},
            "optional": nodes.Hidden(None),
        },
    )
    state = FormState(form.default_input(env))
    assert form.to_cfg(require_result(state, env), env) == ScanCfg(
        pulse=ScanPulseCfg(freq=0.0, gain=0.25), optional=None
    )


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
            "count": 7,
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
    count = state.describe(("count",), env).target
    assert count is not None
    assert count.title == "Repeats" and count.description == "Repeat count."


def test_range_single_point_conversion_reports_core_error_at_sweep_path() -> None:
    assert hasattr(form_adapter, "Form"), "Missing core-derived Form"
    form = form_adapter.Form(ScalarCfg, {"sweep": {"freq": nodes.Range(1, 2, expts=1)}})
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
