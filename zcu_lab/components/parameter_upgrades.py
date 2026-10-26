"""Lab runtime version and durable, explicitly invoked offline upgrade plan."""

from zcu_tools.resources.parameter_schema import ParameterSchemaPolicy
from zcu_tools.resources.parameter_upgrade import (
    ParameterDocument,
    UpgradePlan,
    UpgradeStep,
)

from .flux_upgrade import classify_flux_document, convert_flux_v1_to_v2

# Runtime promotion belongs with removal of the retired component fields.
PARAMETER_SCHEMA_POLICY = ParameterSchemaPolicy(
    current_version=1,
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
    The offline target is 2 while runtime stays at 1 until its retired fields are
    removed. During that transition, operate on copies only: output v2 cannot be
    loaded by runtime v1. Building the plan does not perform I/O or registration.
    """
    return UpgradePlan(
        policy=ParameterSchemaPolicy(2, PARAMETER_SCHEMA_POLICY.upgrade_command),
        steps=(UpgradeStep(1, convert_flux_v1_to_v2),),
        validate=_validate,
        classify_legacy=_classify_legacy,
        notes=(
            "Reference points are assumed; recalibrate with all other contributing devices at their integer references.",
            f"Offline target is 2; runtime is {PARAMETER_SCHEMA_POLICY.current_version}. If runtime is 1, use copies only; it cannot load v2 output.",
            "Device name syntax is checked, not device existence. Check names when opening the point in the GUI.",
        ),
    )
