"""Synthetic envelope inputs shared by Form and component scenarios."""

from typing import Annotated

from zcu_tools.cfg_model import (
    ConfigBase,
    ConstWaveformCfg,
    FieldMeta,
    GaussWaveformCfg,
    PulseSpec,
    Template,
    TemplateDef,
)
from zcu_tools.gui.form import FormEnv
from zcu_tools.gui.form_adapter import Form
from zcu_tools.program.v2.modules.waveform import EnvelopeWaveformCfg
from zcu_tools.resources.context.library import (
    ComponentLibrary,
    LibrarySnapshot,
    ModuleLibraryDocument,
)


class EnvelopePlan(ConfigBase):
    """Two independent finite-envelope settings, without pulse or lab policy."""

    first: Annotated[EnvelopeWaveformCfg, FieldMeta(label="First envelope")]
    second: Annotated[EnvelopeWaveformCfg, FieldMeta(label="Second envelope")]


def envelope_env() -> FormEnv:
    """Capture an allowed Gaussian, an excluded const and an unrelated pulse."""
    return FormEnv(
        library=LibrarySnapshot(
            ModuleLibraryDocument(
                format="zcu.module-library",
                format_version="2.0",
                components={
                    "Q": ComponentLibrary(
                        waveforms={
                            "shaped": GaussWaveformCfg(length=0.48, sigma=0.08),
                            "constant": ConstWaveformCfg(length=0.7),
                        },
                        pulses={
                            "drive": PulseSpec(
                                ch=0,
                                nqz=1,
                                freq=100.0,
                                gain=0.2,
                                waveform=ConstWaveformCfg(length=0.1),
                            )
                        },
                    )
                },
            )
        )
    )


def envelope_form() -> Form[EnvelopePlan]:
    """Seed each independent Gaussian through a caller-owned Template catalog."""
    form = Form(
        EnvelopePlan, {"first": Template("envelope"), "second": Template("envelope")}
    )
    form.bind_templates(
        {
            "envelope": TemplateDef(
                GaussWaveformCfg, "Gaussian envelope", {"length": 0.48, "sigma": 0.08}
            )
        },
        adapter="synthetic",
    )
    return form
