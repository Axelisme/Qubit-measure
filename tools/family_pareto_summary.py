"""Build, persist and compare single-scenario summaries without raw samples.

JSON is the numeric source. Markdown is generated from the same schema; neither
format contains dominance conclusions or requires local reference archives.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

from pydantic import TypeAdapter

from tools.family_pareto_inputs import canonical_json
from tools.family_pareto_results import FamilyResult, FamilyStatus, InputResult


@dataclass(frozen=True)
class Provenance:
    """Measurement identity and annotations, never eligibility gates.

    head/tree are Git object IDs; dirty describes tracked/untracked Git state.
    started_at_utc is the tool's ISO-8601 UTC start time. exclusive is the
    caller's declaration, not a verified runner lock or receipt fact.
    """

    head: str
    tree: str
    dirty: bool
    started_at_utc: str
    exclusive: bool


@dataclass(frozen=True)
class ModeSummary:
    """One history mode's seconds, count and explicit missing-measurement reason.

    wall_median is wall seconds; cpu_median is whole-process CPU seconds.
    Both are None when unmeasured, never zero-fill.
    sample_count counts retained samples. unmeasured_reason is None if measured.
    """

    wall_median: float | None
    cpu_median: float | None
    sample_count: int
    unmeasured_reason: str | None


@dataclass(frozen=True)
class PolicySummary:
    """A pinned/default/fixed policy's outcome and timing ownership.

    status comes only from FamilyResult.status. error is whole-sequence maximum
    density difference or None on refusal. modes maps history names to medians.
    whitelist=None means unrestricted routing. companions excludes the target
    only for target/target_and_companions attribution; group policies have no
    target companions. attribution names the measured cost's owner, not isolated
    target timing. refusal_reason preserves the public refusal text, if any.
    """

    status: FamilyStatus
    error: float | None
    modes: dict[str, ModeSummary]
    whitelist: tuple[str, ...] | None
    companions: tuple[str, ...]
    attribution: str
    refusal_reason: str | None


@dataclass(frozen=True)
class ScenarioSummary:
    """Version-1 numeric record for exactly one Input.name.

    name identifies the builder. manifest_sha256 hashes the current frozen
    manifest; physical_sha256 includes model identity, oracle_sha256 omits it.
    measured_families lists only measured pinned families, not default/fixed.
    policies includes those pinned policies and separate default/fixed results.
    provenance records this single run; summaries never merge policy sources.
    """

    schema_version: int
    name: str
    manifest_sha256: str
    physical_sha256: str
    oracle_sha256: str
    measured_families: tuple[str, ...]
    provenance: Provenance
    policies: dict[str, PolicySummary]


def _mode_summaries(policy: FamilyResult) -> dict[str, ModeSummary]:
    if not policy.modes:
        reason = policy.companion_blocked or policy.unavailable
        return {
            name: ModeSummary(None, None, 0, reason)
            for name in ("first_evolve", "warm_clear_history", "warm_retain_history")
        }
    return {
        mode.mode: ModeSummary(
            mode.wall_median, mode.cpu_median, len(mode.samples), mode.unmeasured_reason
        )
        for mode in policy.modes
    }


def build_summary(
    row: InputResult,
    *,
    manifest_sha256: str,
    measured_families: tuple[str, ...],
    provenance: Provenance,
) -> ScenarioSummary:
    """Reduce a completed input row without classifying or inventing measurements.

    Preserve all policy modes and missing values. FamilyResult.status raises
    when required accuracy/refusal facts are missing. No files are read/written.
    """
    policies = [*row.policies]
    if row.fixed_default_group is not None:
        policies.append(row.fixed_default_group)
    return ScenarioSummary(
        1,
        row.name,
        manifest_sha256,
        row.physical_sha256,
        row.oracle_sha256,
        measured_families,
        provenance,
        {
            policy.family: PolicySummary(
                policy.status,
                policy.error,
                _mode_summaries(policy),
                policy.whitelist,
                tuple(
                    name for name in (policy.whitelist or ()) if name != policy.family
                )
                if policy.attribution in {"target", "target_and_companions"}
                else (),
                policy.attribution,
                policy.companion_blocked or policy.unavailable,
            )
            for policy in policies
        },
    )


def read_summary(path: Path) -> ScenarioSummary:
    """Read a current schema summary; I/O, invalid schema and missing fields raise.

    This is not a reader for complete matrices or historical result formats.
    """
    summary = TypeAdapter(ScenarioSummary).validate_json(
        path.read_text(encoding="utf-8")
    )
    if summary.schema_version != 1:
        raise ValueError(f"unsupported summary schema: {summary.schema_version}")
    return summary


def render_summary(summary: ScenarioSummary) -> str:
    """Render provenance, ownership and all mode values, with no timing verdicts."""
    lines = [
        "**Readers:** framework-dev. Read one scenario's measured outcomes and provenance.",
        "",
        f"# {summary.name}",
        "",
        f"Provenance: {canonical_json(asdict(summary.provenance))}",
        "",
        f"Measured pinned families: {', '.join(summary.measured_families)}.",
        "",
        f"Manifest: `{summary.manifest_sha256}`. Physical: `{summary.physical_sha256}`. Oracle: `{summary.oracle_sha256}`.",
        "",
        "Times describe this table only; differences between runs do not establish faster/slower families.",
        "",
        "| Policy | Status | Final density error | Attribution | Whitelist | Companions | Refusal |",
        "| --- | --- | ---: | --- | --- | --- | --- |",
    ]
    for family, policy in summary.policies.items():
        lines.append(
            f"| {family} | {policy.status} | {policy.error} | {policy.attribution} | "
            f"{policy.whitelist} | {policy.companions} | {policy.refusal_reason or '—'} |"
        )
    lines += [
        "",
        "| Policy | Mode | Wall median s | Process CPU median s | Samples | Unmeasured reason |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for family, policy in summary.policies.items():
        for name, mode in policy.modes.items():
            lines.append(
                f"| {family} | {name} | {mode.wall_median} | {mode.cpu_median} | "
                f"{mode.sample_count} | {mode.unmeasured_reason or '—'} |"
            )
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class ValueChange:
    """Before/after values and numeric after-minus-before delta, if both measured.

    None is preserved, including an unmeasured time. A delta is descriptive only.
    """

    before: object
    after: object
    delta: float | None = None


@dataclass(frozen=True)
class SummaryComparison:
    """Non-gating comparison of two separate runs.

    comparable is false on name/manifest/physical/oracle mismatch; then policies
    is empty. fingerprint_differences names those mismatches. baseline_only and
    current_only list pinned families missing from the other measurement.
    metadata_changes maps changed provenance fields and measured_families to
    before/after values. policies maps common pinned/default/fixed names to
    status, error, ownership and mode-field value pairs, including equal values.
    """

    comparable: bool
    fingerprint_differences: tuple[str, ...]
    baseline_only: tuple[str, ...]
    current_only: tuple[str, ...]
    metadata_changes: dict[str, ValueChange]
    policies: dict[str, dict[str, ValueChange]]


def _numeric_change(before: float | None, after: float | None) -> ValueChange:
    return ValueChange(
        before, after, None if before is None or after is None else after - before
    )


def _policy_comparison(
    before: PolicySummary, after: PolicySummary
) -> dict[str, ValueChange]:
    result = {
        "status": ValueChange(before.status, after.status),
        "error": _numeric_change(before.error, after.error),
        "whitelist": ValueChange(before.whitelist, after.whitelist),
        "companions": ValueChange(before.companions, after.companions),
        "attribution": ValueChange(before.attribution, after.attribution),
        "refusal_reason": ValueChange(before.refusal_reason, after.refusal_reason),
    }
    for name in dict.fromkeys([*before.modes, *after.modes]):
        if name not in before.modes or name not in after.modes:
            result[name] = ValueChange(before.modes.get(name), after.modes.get(name))
            continue
        old, new = before.modes[name], after.modes[name]
        result[f"{name}.wall_median"] = _numeric_change(
            old.wall_median, new.wall_median
        )
        result[f"{name}.cpu_median"] = _numeric_change(old.cpu_median, new.cpu_median)
        result[f"{name}.sample_count"] = ValueChange(old.sample_count, new.sample_count)
        result[f"{name}.unmeasured_reason"] = ValueChange(
            old.unmeasured_reason, new.unmeasured_reason
        )
    return result


def compare_summaries(
    baseline: ScenarioSummary, current: ScenarioSummary
) -> SummaryComparison:
    """Compare identical input fingerprints and shared measured policies only.

    Always return metadata and family-set differences, even if incomparable.
    Include separate default/fixed policies when both runs contain them. Never
    declare cross-run speed, dominance or write eligibility. Inputs are unchanged.
    """
    before = asdict(baseline.provenance) | {
        "measured_families": baseline.measured_families
    }
    after = asdict(current.provenance) | {
        "measured_families": current.measured_families
    }
    differences = tuple(
        name
        for name, old, new in (
            ("name", baseline.name, current.name),
            ("manifest_sha256", baseline.manifest_sha256, current.manifest_sha256),
            ("physical_sha256", baseline.physical_sha256, current.physical_sha256),
            ("oracle_sha256", baseline.oracle_sha256, current.oracle_sha256),
        )
        if old != new
    )
    shared = set(baseline.measured_families) & set(current.measured_families)
    shared.update(
        {"default", "fixed_default_group"}
        & baseline.policies.keys()
        & current.policies.keys()
    )
    return SummaryComparison(
        not differences,
        differences,
        tuple(
            name
            for name in baseline.measured_families
            if name not in current.measured_families
        ),
        tuple(
            name
            for name in current.measured_families
            if name not in baseline.measured_families
        ),
        {
            name: ValueChange(old, after[name])
            for name, old in before.items()
            if old != after[name]
        },
        {
            name: _policy_comparison(policy, current.policies[name])
            for name, policy in baseline.policies.items()
            if name in shared
        }
        if not differences
        else {},
    )


def render_comparison(comparison: SummaryComparison) -> str:
    """Render value pairs/deltas; incomparable inputs retain only metadata changes."""
    return (
        "\n".join(
            [
                "Summary comparison (descriptive only; no cross-run speed verdict):",
                "Comparable"
                if comparison.comparable
                else "Not comparable: input fingerprints differ",
                canonical_json(asdict(comparison)),
            ]
        )
        + "\n"
    )


def publish_summary(summary: ScenarioSummary, path: Path, baseline: Path) -> None:
    """Print a non-gating baseline comparison, then write this run's JSON/Markdown.

    Read baseline before writing, so path may equal baseline. Missing/invalid
    baseline prints a diagnostic and still writes. Output failures propagate.
    """
    try:
        comparison = compare_summaries(read_summary(baseline), summary)
        message = render_comparison(comparison)
    except (OSError, ValueError, KeyError) as error:
        message = f"Comparison unavailable: {baseline}: {error}"
    print(message)
    write_summary(summary, path)


def write_summary(summary: ScenarioSummary, path: Path) -> None:
    """Create parents and overwrite caller-selected JSON and sibling .md outputs.

    No clean/exclusive/all-family gate applies. Filesystem failures propagate;
    a failed Markdown write can leave the JSON already updated.
    """
    if path.suffix != ".json":
        raise ValueError("summary destination must have .json suffix")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(asdict(summary)) + "\n", encoding="utf-8")
    path.with_suffix(".md").write_text(render_summary(summary), encoding="utf-8")
