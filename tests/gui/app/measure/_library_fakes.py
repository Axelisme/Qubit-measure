"""Prepared independent CfgSchema drafts for library/editor contracts.

These inputs belong to the retained library and writeback editor, not measure
adapter defaults. Each call returns a fresh tree, without source capture.
"""

from zcu_tools.experiment.cfg_editing import PROGRAM_SHAPES
from zcu_tools.gui.app.measure.specs import MAIN_PROGRAM_SPEC_POLICY
from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    ReferenceSpec,
    ReferenceValue,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    make_custom_reference_key,
    make_default_value,
)


def sample_schema() -> CfgSchema:
    """Return scalar/sweep editor input for library remote-edit contracts."""
    return CfgSchema(
        CfgSectionSpec(
            fields={
                "reps": ScalarSpec("Reps", int),
                "rounds": ScalarSpec("Rounds", int),
                "gain": ScalarSpec("Gain", float),
                "noise_scale": ScalarSpec("Noise scale", float),
                "sweep": SweepSpec("Sample"),
            }
        ),
        CfgSectionValue(
            fields={
                "reps": DirectValue(100),
                "rounds": DirectValue(10),
                "gain": DirectValue(0.1),
                "noise_scale": DirectValue(0.1),
                "sweep": SweepValue(start=5.0, stop=6.0, expts=11),
            }
        ),
    )


def readout_schema() -> CfgSchema:
    """Return a library readout reference with direct and pulse variants."""
    variants = [
        PROGRAM_SHAPES.get("module", tag).make_spec(MAIN_PROGRAM_SPEC_POLICY)
        for tag in ("readout/direct", "readout/pulse")
    ]
    variants[1] = (
        variants[1]
        .lock_literal("pulse_cfg.freq", 0.0)
        .lock_literal("ro_cfg.ro_freq", 0.0)
    )
    pulse = variants[1]
    value = make_default_value(pulse)
    for path, item in {
        "pulse_cfg.ch": 0,
        "pulse_cfg.nqz": 2,
        "pulse_cfg.freq": 0.0,
        "pulse_cfg.gain": 0.1,
        "pulse_cfg.waveform.length": 1.0,
        "ro_cfg.ro_ch": 0,
        "ro_cfg.ro_freq": 0.0,
        "ro_cfg.ro_length": 1.0,
        "ro_cfg.trig_offset": 0.0,
    }.items():
        value.with_field(path, item)
    return CfgSchema(
        CfgSectionSpec(
            fields={
                "modules": CfgSectionSpec(
                    fields={
                        "readout": ReferenceSpec(kind="module", allowed=variants),
                    }
                ),
            }
        ),
        CfgSectionValue(
            fields={
                "modules": CfgSectionValue(
                    fields={
                        "readout": ReferenceValue(
                            make_custom_reference_key(pulse.label), value
                        ),
                    }
                ),
            }
        ),
    )


def device_schema() -> CfgSchema:
    """Return an independent required device selector editor input."""
    return CfgSchema(
        CfgSectionSpec(
            fields={
                "dev": CfgSectionSpec(
                    fields={
                        "flux_dev": ScalarSpec(
                            "Flux Device", str, choices_source="devices", required=True
                        ),
                    }
                ),
            }
        ),
        CfgSectionValue(
            fields={
                "dev": CfgSectionValue(fields={"flux_dev": DirectValue("")}),
            }
        ),
    )
