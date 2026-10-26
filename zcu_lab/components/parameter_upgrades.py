"""Lab component version declaration and durable offline upgrade plan.

The current version remains 1 until the corresponding structural schema change
is published. Legacy classification and future steps belong here, not in lib or
in the disposable storage-import tools.
"""

from pathlib import Path

from zcu_tools.format_version import FormatVersion, validate_header
from zcu_tools.resources.entry.aliases import parse_entry_aliases
from zcu_tools.resources.entry.schema import PointDocument, SetupDocument
from zcu_tools.resources.parameter_schema import ParameterSchemaPolicy
from zcu_tools.resources.parameter_upgrade import ParameterDocument, UpgradePlan

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


def _validate_current(document: ParameterDocument, version: int) -> None:
    if version != PARAMETER_SCHEMA_POLICY.current_version:
        raise ValueError(f"No lab validator for schema_version {version}")
    source = document.source_path or Path("<parameter-document>")
    if document.data.get("format") == "zcu.entry-template":
        validate_header(
            document.data,
            expected_format="zcu.entry-template",
            supported_version=FormatVersion(1, 0),
            source=source,
        )
        for section in ("components", "aliases"):
            if section not in document.data:
                raise ValueError(f"missing {section} section")
        PointDocument.validate_components(document.data["components"])
        parse_entry_aliases(document.data["aliases"], source=source)
    else:

        class CurrentPoint(PointDocument):
            _source = source

        class CurrentSetup(SetupDocument):
            _source = source

        model = CurrentSetup if source.name == "setup.yaml" else CurrentPoint
        model.model_validate(document.data)


def _unavailable_legacy_classifier(_document: ParameterDocument) -> bool:
    raise ValueError(
        "Cannot determine source schema_version: the complete lab legacy "
        "classifier is not installed. No source version was inferred; "
        "no document was changed."
    )


def parameter_upgrade_plan() -> UpgradePlan:
    """Return this lab's current-version checks and retained pure upgrade steps.

    Bootstrap component models/policy with register_all before running the plan.
    Version 1 has no structural steps yet. Tagged current files are validated
    and skipped; unversioned files are explicitly rejected until the complete
    lab legacy classifier is supplied. They are not classified as corrupt or
    assumed to be v1. No I/O or registration occurs when building this plan.
    """
    return UpgradePlan(
        policy=PARAMETER_SCHEMA_POLICY,
        steps=(),
        validate=_validate_current,
        classify_legacy=_unavailable_legacy_classifier,
    )
