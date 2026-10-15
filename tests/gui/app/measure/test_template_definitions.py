"""Public deferred references still serve legacy cfg seeds until retirement."""

import pytest
from zcu_tools import cfg_model
from zcu_tools.experiment.cfg_editing import PROGRAM_SHAPES
from zcu_tools.gui.app.measure.specs import MAIN_PROGRAM_SPEC_POLICY
from zcu_tools.gui.cfg import (
    DirectValue,
    EvalValue,
    ReferenceValue,
    make_custom_reference_key,
    make_default_value,
    read_value_path,
)
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import SavedPoint
from zcu_tools.gui.session.types import ParameterContext, SessionEnv
from zcu_tools.resources.context.library import (
    ComponentLibrary,
    LibrarySnapshot,
    ModuleLibraryDocument,
)
from zcu_tools.resources.entry import ComponentSchema
from zcu_tools.resources.entry.schema import PointDocument, PointGeneral


class TemplateComponent(ComponentSchema):
    module: dict[str, str | None]
    freq: float = 4200.0


@pytest.fixture
def environment() -> SessionEnv:
    point = PointDocument.model_construct(
        format="zcu.parameter-container",
        format_version="1.0",
        general=PointGeneral(created_at="2026-10-13T00:00:00Z"),
        components={
            "Q1": TemplateComponent(
                kind="template/test",
                module={
                    "x90": "Q1.pulses.pi2",
                    "unset": None,
                    "dangling": "Q1.pulses.missing",
                },
            )
        },
    )
    library = LibrarySnapshot(
        ModuleLibraryDocument(
            format="zcu.module-library",
            format_version="1.0",
            components={
                "Q1": ComponentLibrary(
                    pulses={
                        "pi2": cfg_model.PulseSpec(
                            ch=2,
                            nqz=1,
                            freq=4200.0,
                            gain=0.25,
                            waveform=cfg_model.ConstWaveformCfg(length=0.04),
                        )
                    }
                )
            },
        )
    )
    parameters = ParameterContext(
        point=point,
        library=library,
        target=SavedPoint(entry_id="synthetic", point="working"),
        roles=None,
    ).bind_aliases((AliasRow("qubit", "Q1"),), {})
    return SessionEnv(parameters=parameters, soc=None, soccfg=None)


def test_public_reference_resolver_rejects_unsupported_kind_before_source_lookup(
    environment: SessionEnv,
) -> None:
    from zcu_tools.gui.app.measure.reference_defaults import resolve_library_reference

    with pytest.raises(ValueError, match="kind"):
        resolve_library_reference(
            "$unbound.waveforms.base", environment, kind="waveform", fallback=None
        )


def test_public_reference_resolver_rejects_empty_path(environment: SessionEnv) -> None:
    from zcu_tools.gui.app.measure.reference_defaults import resolve_library_reference

    with pytest.raises(ValueError, match="path"):
        resolve_library_reference(" ", environment, kind="module", fallback=None)


def test_public_reference_resolver_returns_fresh_fallback_for_a_missing_root(
    environment: SessionEnv,
) -> None:
    from zcu_tools.gui.app.measure.reference_defaults import resolve_library_reference

    spec = PROGRAM_SHAPES.module("pulse").make_spec(MAIN_PROGRAM_SPEC_POLICY)
    fallback = ReferenceValue(
        make_custom_reference_key("Pulse"), make_default_value(spec)
    )
    result = resolve_library_reference(
        "Q1.pulses.missing", environment, kind="module", fallback=lambda _env: fallback
    )
    assert result == fallback
    assert result is not fallback
    assert (
        resolve_library_reference(
            "Q1.pulses.missing", environment, kind="module", fallback=None
        )
        is None
    )


def seeded_fallback_definitions(slot: str) -> dict[str, cfg_model.TemplateDef]:
    """Synthetic forward reference with nonstructural scalar/expression seeds."""
    return {
        "reset": cfg_model.TemplateDef(
            cfg_model.BathResetSpec,
            "Reset",
            {
                "pi2_cfg": cfg_model.Slot(
                    f"{{qubit}}.module.{slot}",
                )
            },
        ),
        "drive": cfg_model.TemplateDef(
            cfg_model.PulseSpec,
            "Drive",
            {
                "ch": 3,
                "nqz": 1,
                "freq": cfg_model.Expr("{qubit}.freq + 23"),
                "gain": 0.37,
                "waveform.length": 0.08,
            },
        ),
    }


def test_public_reference_defaults_expand_template_using_injected_definitions(
    environment: SessionEnv,
) -> None:
    from zcu_tools.gui.app.measure.reference_defaults import resolve_library_reference

    try:
        first = resolve_library_reference(
            "$qubit.module.unset",
            environment,
            kind="module",
            fallback=cfg_model.Template("drive"),
            templates=seeded_fallback_definitions("unset"),
        )
    except TypeError as exc:
        pytest.fail(
            f"Public resolver must accept Template and injected definitions: {exc}"
        )
    assert isinstance(first, ReferenceValue)
    assert first.chosen_key == make_custom_reference_key("Pulse")
    assert read_value_path(first.value, "freq") == EvalValue("$qubit.freq + 23")
    assert read_value_path(first.value, "gain") == DirectValue(0.37)
