"""Explicit test catalog with context-reading and context-free factories."""

from collections.abc import Callable
from functools import partial

from zcu_tools.experiment.cfg_editing import PROGRAM_SHAPES, ProgramShape
from zcu_tools.gui.app.measure.adapter import SessionEnv
from zcu_tools.gui.app.measure.specs import MAIN_PROGRAM_SPEC_POLICY, make_pulse_spec
from zcu_tools.gui.app.measure.template_catalog import (
    TemplateCatalog,
    TemplateEntry,
    TemplateItemKind,
)
from zcu_tools.gui.cfg import (
    CfgSectionSpec,
    ReferenceValue,
    make_custom_reference_key,
    make_default_value,
)
from zcu_tools.gui.expected_error import InvalidInputError
from zcu_tools.gui.session.parameters import ParameterContext


def blank_factory(shape: Callable[[], CfgSectionSpec], label: str):
    def make_value(_ctx: SessionEnv) -> ReferenceValue:
        return ReferenceValue(
            make_custom_reference_key(label), make_default_value(shape())
        )

    return make_value


def pulse_value(ctx: SessionEnv) -> ReferenceValue:
    value = make_default_value(make_pulse_spec())
    parameters = ctx.parameters
    if parameters is None:
        raise ValueError("Template requires a fixed capture")
    value.with_field("ch", capture_number(parameters, "$resonator.wiring.ch"))
    value.with_field("freq", capture_number(parameters, "$resonator.freq"))
    return ReferenceValue(make_custom_reference_key("pulse"), value)


def readout_shape() -> CfgSectionSpec:
    return PROGRAM_SHAPES.get("module", "readout/pulse").make_spec(
        MAIN_PROGRAM_SPEC_POLICY
    )


def readout_value(ctx: SessionEnv) -> ReferenceValue:
    value = make_default_value(readout_shape())
    # Fully specified inline values make run/config tests independent of calibration.
    for path, item in {
        "gen_ch": 0,
        "nqz": 2,
        "freq": 0.0,
        "gain": 0.1,
        "waveform.length": 1.0,
        "ro_ch": 0,
        "ro_length": 1.0,
        "trig_offset": 0.0,
    }.items():
        value.with_field(path, item)
    return ReferenceValue(make_custom_reference_key("Pulse Readout"), value)


def captured_readout(ctx: SessionEnv) -> ReferenceValue:
    result = readout_value(ctx)
    parameters = ctx.parameters
    if parameters is None:
        raise ValueError("Template requires a fixed capture")
    for path, source in {
        "gen_ch": "$resonator.wiring.ch",
        "freq": "$resonator.freq",
        "ro_ch": "$resonator.wiring.ro_ch",
        "trig_offset": "$resonator.ext.time_of_fly",
    }.items():
        result.value.with_field(path, capture_number(parameters, source))
    return result


def register_all_templates(catalog: TemplateCatalog) -> None:
    catalog.register(
        TemplateEntry(
            "res_probe",
            "Resonator probe",
            "module",
            make_pulse_spec,
            pulse_value,
            "readout_rf",
        )
    )
    catalog.register(
        TemplateEntry(
            "readout",
            "Readout",
            "module",
            readout_shape,
            captured_readout,
            "readout_rf",
        )
    )
    for template_id, discriminator, default in (
        ("bath_reset", "reset/bath", "reset_bath"),
        ("pi_pulse", "pulse", "pi_amp"),
        ("none_reset", "reset/none", "reset_none"),
    ):
        shape = PROGRAM_SHAPES.get("module", discriminator)
        factory = partial(shape.make_spec, MAIN_PROGRAM_SPEC_POLICY)
        catalog.register(
            TemplateEntry(
                template_id,
                template_id,
                "module",
                factory,
                blank_factory(factory, discriminator),
                default,
            )
        )
    blank_shapes: tuple[tuple[TemplateItemKind, tuple[ProgramShape, ...]], ...] = (
        ("module", PROGRAM_SHAPES.modules()),
        ("waveform", PROGRAM_SHAPES.waveforms()),
    )
    for kind, shapes in blank_shapes:
        for shape in shapes:
            factory = partial(shape.make_spec, MAIN_PROGRAM_SPEC_POLICY)
            catalog.register(
                TemplateEntry(
                    f"{shape.discriminator}:blank",
                    f"Blank: {shape.discriminator}",
                    kind,
                    factory,
                    blank_factory(factory, shape.discriminator),
                )
            )


def capture_number(parameters: ParameterContext, path: str) -> int | float:
    item = parameters.lookup(path)
    if not item.present:
        raise InvalidInputError(f"missing parameter: {path}")
    value = item.value
    if not isinstance(value, (int, float)):
        raise InvalidInputError(f"numeric parameter required: {path}")
    return value
