"""Lab runtime version and durable, explicitly invoked offline upgrade plan."""

from zcu_tools.resources.parameter_schema import ParameterSchemaPolicy
from zcu_tools.resources.parameter_upgrade import (
    ParameterDocument,
    UpgradePlan,
    UpgradeStep,
)

from .flux_upgrade import classify_flux_document, convert_flux_v1_to_v2

PARAMETER_SCHEMA_POLICY = ParameterSchemaPolicy(
    current_version=2,
    upgrade_command=(
        "uv",
        "run",
        "--no-sync",
        "--",
        "python",
        "scripts/upgrade_parameters.py",
    ),
)


def _validate(document: ParameterDocument, version: int) -> None:
    classification = classify_flux_document(document)
    expected = "legacy" if version == 1 else "current"
    if version not in (1, 2) or classification.status != expected:
        raise ValueError(
            f"{classification.status}: {'; '.join(classification.diagnostics)}"
        )


def _classify_legacy(document: ParameterDocument) -> bool:
    _validate(document, 1)
    return True


def parameter_upgrade_plan() -> UpgradePlan:
    """Return retained v1 to v2 steps and full offline format classification.

    Bootstrap component models with register_all before executing this plan.
    Unmarked documents are accepted only after complete legacy classification.
    The target matches the runtime policy. Building the plan does not perform
    I/O or registration. Runtime loading never executes this plan.
    """
    return UpgradePlan(
        policy=PARAMETER_SCHEMA_POLICY,
        steps=(UpgradeStep(1, convert_flux_v1_to_v2),),
        validate=_validate,
        classify_legacy=_classify_legacy,
        notes=(
            "Offset is derived from local half/period; coil half/integer references are dropped when a local calibration exists.",
            "Rename every reported unnamed_device/unnamed_coil location to the actual device name. Device existence is not checked.",
        ),
    )
