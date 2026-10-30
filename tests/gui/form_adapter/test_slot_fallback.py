"""Slot fallback selection and Template-owned seed diagnostics at the Form seam."""

import pytest
from zcu_tools.cfg_model import (
    BathResetSpec,
    ConfigBase,
    Expr,
    FieldMeta,
    PulseSpec,
    Slot,
    Template,
    TemplateDef,
)
from zcu_tools.gui.form import (
    FormEnv,
    FormErrors,
    FormState,
)
from zcu_tools.gui.form_adapter import Form
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import SavedPoint
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.program.v2.modules import BathResetCfg, PulseCfg
from zcu_tools.resources.context.library import (
    ComponentLibrary,
    LibrarySnapshot,
    ModuleLibraryDocument,
)
from zcu_tools.resources.entry.schema import PointDocument, PointGeneral


class Core(ConfigBase):
    """A required native pulse field has no GUI-specific declaration."""

    probe: PulseCfg


def templates() -> dict[str, TemplateDef]:
    """A deliberately partial persistent template supplies a required child."""
    return {
        "pulse": TemplateDef(
            PulseSpec,
            "Pulse",
            {
                "ch": 2,
                "nqz": 1,
                "freq": Expr("100 + 20"),
                "gain": 0.1,
                "waveform.style": "const",
                "waveform.length": Expr("2 * .2"),
            },
        )
    }


def environment() -> FormEnv:
    """Capture a library plus an explicitly unbound alias, without source I/O."""
    pulse = PulseSpec.model_validate(
        {
            "ch": 1,
            "nqz": 1,
            "freq": 100,
            "gain": 0.3,
            "waveform": {"style": "const", "length": 0.1},
        }
    )
    library = LibrarySnapshot(
        ModuleLibraryDocument(
            format="zcu.module-library",
            format_version="2.0",
            components={"Q": ComponentLibrary(pulses={"probe": pulse})},
        )
    )
    parameters = ParameterContext(
        point=PointDocument(
            format="zcu.parameter-container",
            format_version="1.0",
            schema_version=1,
            general=PointGeneral(created_at="2026-10-13T00:00:00Z"),
        ),
        library=library,
        target=SavedPoint(entry_name="synthetic", point="synthetic"),
        roles=None,
        storage=None,
    ).bind_aliases((AliasRow("unbound", ""),), {})
    return FormEnv(parameters=parameters, library=library)


def bound_form(node: Slot | Template) -> Form[Core]:
    """Bind the synthetic template catalog through the public Form seam."""
    form = Form(Core, {"probe": node})
    form.bind_templates(templates(), adapter="synthetic")
    return form


class ResetCore(ConfigBase):
    """Nested references are part of the persistent reset Spec, not form modes."""

    reset: BathResetCfg


@pytest.mark.parametrize(
    "path", ["Q.pulses.probe", "Q.pulses.missing", "$unbound.module.probe"]
)
@pytest.mark.parametrize("copy", [False, True])
def test_slot_fallback_matches_explicit_source_selection(path: str, copy: bool) -> None:
    env = environment()
    fallback = Template("pulse", set={"gain": 0.25})
    form = bound_form(Slot(path, copy=copy, fallback=fallback))
    explicit = bound_form(
        Slot(path, copy=copy) if path == "Q.pulses.probe" else fallback
    )
    input = form.default_input(env)
    expected = explicit.default_input(env)
    assert input.data == expected.data
    assert form.diagnostics(input, env, None) == explicit.diagnostics(
        expected, env, None
    )
    value = FormState(input).result(env)
    expected_value = FormState(expected).result(env)
    assert not isinstance(value, FormErrors), value
    assert not isinstance(expected_value, FormErrors), expected_value
    assert form.to_cfg(value, env) == explicit.to_cfg(expected_value, env)


@pytest.mark.parametrize("path", ["Q.pulses.probe", "Q.pulses.missing"])
@pytest.mark.parametrize("override", [False, True])
def test_template_seed_metadata_only_applies_when_adopted(
    path: str, override: bool
) -> None:
    env = environment()
    metadata: dict[tuple[str | int, ...], FieldMeta] = {
        ("probe", "module", "gain"): FieldMeta(
            description="Template starting gain", calibration="uncalibrated"
        )
    }
    form = bound_form(
        Slot(path, copy=True, fallback=Template("pulse", seed_metadata=metadata))
    )
    explicit = bound_form(
        Slot(path, copy=True) if path == "Q.pulses.probe" else Template("pulse")
    ).with_seed_metadata({} if path == "Q.pulses.probe" else metadata)
    if override:
        replacement: dict[tuple[str | int, ...], FieldMeta] = {
            ("probe", "module", "gain"): FieldMeta(description="Explicit source")
        }
        form = form.with_seed_metadata(replacement)
        explicit = explicit.with_seed_metadata(replacement)
    input = form.default_input(env)
    expected = explicit.default_input(env)
    assert input.data == expected.data
    assert form.diagnostics(input, env, None) == explicit.diagnostics(
        expected, env, None
    )
    row = next(
        row
        for row in form.diagnostics(input, env, None)
        if row.path == ("probe", "module", "gain")
    )
    assert row.uncalibrated is (
        True if path.endswith("missing") and not override else None
    )


@pytest.mark.parametrize("available", [False, True])
def test_nested_slot_adopts_its_template_and_metadata_only_when_missing(
    available: bool,
) -> None:
    fallback = Template(
        "pulse",
        set={"gain": 0.25},
        seed_metadata={
            ("reset", "module", "pi2_cfg", "module", "gain"): FieldMeta(
                description="Nested template gain", calibration="uncalibrated"
            )
        },
    )
    catalog = templates() | {
        "reset": TemplateDef(
            BathResetSpec,
            "Reset",
            {
                "cavity_tone_cfg": Slot("Q.pulses.probe"),
                "qubit_tone_cfg": Slot("Q.pulses.probe"),
                "pi2_cfg": Slot(
                    "Q.pulses.probe" if available else "$unbound.module.probe",
                    fallback=fallback,
                ),
            },
        )
    }
    env = environment()
    form = Form(ResetCore, {"reset": Template("reset")})
    form.bind_templates(catalog, adapter="synthetic-nested-fallback")
    input = form.default_input(env)
    value = FormState(input).result(env)
    assert not isinstance(value, FormErrors), value
    cfg = form.to_cfg(value, env)
    assert isinstance(cfg, ResetCore), cfg
    assert cfg.reset.pi2_cfg.freq == (100 if available else 120)
    assert cfg.reset.pi2_cfg.gain == (0.3 if available else 0.25)
    gain_rows = [
        row
        for row in form.diagnostics(input, env, None)
        if row.path == ("reset", "module", "pi2_cfg", "module", "gain")
    ]
    if available:
        assert gain_rows == []  # A reference exposes its selector, not inline fields.
    else:
        assert len(gain_rows) == 1
        assert gain_rows[0].uncalibrated is True
        assert gain_rows[0].seed_source == "Nested template gain"
