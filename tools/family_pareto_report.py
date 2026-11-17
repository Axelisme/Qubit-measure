"""Human-readable real/synthetic Pareto diagnostics."""

from __future__ import annotations

import json
from dataclasses import replace
from statistics import median

from pydantic import TypeAdapter

from tools.family_pareto_results import (
    ALL_FAMILIES,
    DIRECTLY_RETAINED,
    InputResult,
    dominance,
)


def render_saved_summary(matrix_json: str) -> str:
    """Render a saved family_pareto matrix JSON without running any measurement.

    matrix_json is the complete JSON text written by family_pareto.main, with
    inputs and measured_families. Restore its public result dataclasses and use
    render_summary; retain recorded errors and timings, but recompute dominance
    using the current qualification rule. The source matrix is not modified.
    Extra provenance/derived keys are ignored. Missing keys, malformed JSON or
    invalid result fields raise KeyError or ValueError. No files are written.
    """
    matrix = json.loads(matrix_json)
    rows = TypeAdapter(list[InputResult]).validate_python(matrix["inputs"])
    rows = [replace(row, dominators=dominance(row.policies)) for row in rows]
    families = TypeAdapter(tuple[str, ...]).validate_python(matrix["measured_families"])
    return render_summary(rows, families)


def _render_fixed_group(row: InputResult) -> list[str]:
    fixed = row.fixed_default_group
    if fixed is None:
        return []
    default = next(policy for policy in row.policies if policy.family == "default")
    defaults = {mode.mode: mode for mode in default.modes}
    lines = [
        f"Default vs fixed same family group {fixed.whitelist} (group latency, no single-family attribution):",
        "",
        "| Mode/metric | Default seconds | Fixed seconds | Fixed samples | Fixed path | Default/fixed | >2x difference |",
        "| --- | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for mode in fixed.modes:
        original = defaults[mode.mode]
        if not mode.samples or not original.samples:
            lines.append(
                f"| {mode.mode} | — | — | {len(mode.samples)} | slow family not measured | — | — |"
            )
            continue
        for metric, default_time, fixed_time in (
            ("wall", original.wall_median, mode.wall_median),
            ("cpu", original.cpu_median, mode.cpu_median),
        ):
            if default_time is None or fixed_time is None:
                raise ValueError("measured mode has missing time")
            ratio = default_time / fixed_time if fixed_time > 0 else None
            ratio_text = "undefined" if ratio is None else f"{ratio:.6g}"
            finding = ratio is not None and (ratio > 2 or ratio < 0.5)
            lines.append(
                f"| {mode.mode}/{metric} | {default_time:.9g} | {fixed_time:.9g} | "
                f"{len(mode.samples)} | {mode.samples[0].paths} | {ratio_text} | {finding} |"
            )
    return [*lines, ""]


def _family_cpu_lines(row: InputResult) -> list[str]:
    lines = []
    policies = [*row.policies]
    if row.fixed_default_group is not None:
        policies.append(row.fixed_default_group)
    for policy in policies:
        for mode in policy.modes:
            names = sorted({name for sample in mode.samples for name in sample.routing})
            for name in names:
                reports = [
                    sample.routing[name]
                    for sample in mode.samples
                    if name in sample.routing
                ]
                rows = [report.rows for report in reports]
                durations = [report.duration_ns for report in reports]
                role = "group member"
                if name == policy.family:
                    role = "target"
                elif policy.attribution in {"target", "target_and_companions"}:
                    role = (
                        "companion"
                        if name in (policy.whitelist or ())
                        else "preparation diagnostic"
                    )
                lines.append(
                    f"| {policy.family} | {mode.mode} | {name} | {len(reports)} | "
                    f"{min(rows)}..{max(rows)} | {min(durations):.12g}..{max(durations):.12g} | "
                    f"{median(report.prepare_cpu_s for report in reports):.9g} | "
                    f"{median(report.apply_cpu_s for report in reports):.9g} | "
                    f"{median(report.actual_cpu_s for report in reports):.9g} | "
                    f"{role} | "
                    f"{median(sample.family_cpu_shares[name] for sample in mode.samples if name in sample.routing):.6g} |"
                )
    if not lines:
        return []
    return [
        "Family diagnostics (calling-thread CPU, not wall or policy process CPU):",
        "",
        "Logical row/ns ranges and phase medians use reported samples only. Fractions divide each family prepare+apply CPU by the summed reported thread CPU in that sample, then take the median. They exclude unreported overhead and are not wall/process CPU shares. Guard rejection preparation can have CPU with zero rows. Raw reports retain each sample separately.",
        "",
        "| Policy | Mode | Family | Reported samples | Logical rows range | Cumulative ns range | Median prepare CPU s | Median apply CPU s | Median summed CPU s | Role | Median fraction of reported thread CPU |",
        "| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | --- | ---: |",
        *lines,
        "",
    ]


def _policy_lines(row: InputResult) -> list[str]:
    lines: list[str] = []
    for policy in row.policies:
        if policy.error is None:
            lines.append(
                f"| {policy.family} | — | "
                f"{policy.status}: "
                f"{policy.companion_blocked or policy.unavailable}; "
                f"first-group classification: {policy.refusal.family_reasons if policy.refusal else {}} | — | — | — | — | — | — |"
            )
            continue
        for mode in policy.modes:
            if not mode.samples:
                lines.append(
                    f"| {policy.family} | — | 慢 family 未量: {mode.unmeasured_reason} | {mode.mode} | — | 0 | — | — | — |"
                )
                continue
            for metric, value in (
                ("wall", mode.wall_median),
                ("cpu", mode.cpu_median),
            ):
                if value is None:
                    raise ValueError("measured mode has missing time")
                cell = f"{mode.mode}/{metric}"
                competitors = row.dominators.get(policy.family, {}).get(cell, [])
                comparisons = row.dominance_same_companions.get(policy.family, {}).get(
                    cell, {}
                )
                labeled_competitors = [
                    f"{name} (same companions)"
                    if comparisons[name]
                    else f"{name} (different companions; observation only)"
                    for name in competitors
                ]
                peers = row.timing_equivalents.get(policy.family, {}).get(cell, [])
                lines.append(
                    f"| {policy.family} | {policy.error:.12g} | {'PASS' if policy.status == 'qualified' else 'FAIL'} | "
                    f"{cell} | {value:.9g} | {len(mode.samples)} | {mode.samples[0].paths} | {', '.join(labeled_competitors) or 'none'} | {', '.join(peers) or 'none'} |"
                )
    return lines


def render_summary(
    rows: list[InputResult], families: tuple[str, ...] = ALL_FAMILIES
) -> str:
    """Render per-family conclusions and all measured mode/clock cells.

    families names the measured pinned policies, in conclusion order. A subset
    marks the report as partial; omitted families get no availability conclusion.
    Dominance is limited to measured inputs and families. No family deletion or
    accuracy exception follows. Default is shown but never a competitor.
    Diagnostic roles distinguish the target, its whitelisted companions and
    non-whitelisted preparation reports. All reported CPU enters the shares,
    including zero-row guard preparation. Default/fixed policies use group members.
    """
    lines = [
        "# Engine family Pareto table",
        "",
        *(
            [
                "Partial measurement: omitted families are unmeasured, not unavailable. Dominance compares only measured families; this table cannot justify deletion.",
                "",
            ]
            if set(families) != set(ALL_FAMILIES)
            else []
        ),
        f"Measured pinned families: {', '.join(families)}. Default routing and its fixed observed family group remain separate policies.",
        "",
        "Timers cover Engine.evolve only, not program.acquire. Normal modes have five samples. A first evolve >60 wall seconds keeps n=1, skips clear-history with n=0, then retains history in the same session for n=1. No separate probe or warmup runs for slow families.",
        "Errors compare complete complex density to the two-tolerance converged full-Lindblad oracle.",
        "Pinned target and named companions form one timed group. Group wall/process CPU is not isolated target cost. All timing samples enable routing telemetry; family prepare/apply thread CPU is diagnostic only.",
        "Logical segment counts and cumulative ns are not per-span observation. Equal-length spans cannot be distinguished by cumulative duration.",
        "Only policies with complete-sequence final density error <= 2e-3 compete. Qualified policies are equally accurate; time < 0.9 * target time dominates. Exactly 10% is equivalent.",
        "t4, adiabatic_map, flux_dop853, slow_envelope, t6 and dop853 are directly retained, never dominators or dominated targets. Default/fixed groups also stay outside dominance.",
        "No epsilon or weighted score is used. FAIL means error > 2e-3, not permission to keep an inaccurate route.",
        "Conclusions apply only to the measured input set; no deletion is authorized.",
        "Every dominance relation marks companion equality. Different-companion dominance is observation only and cannot alone justify deletion. Companion-blocked rows do not classify the target as unavailable or guard-refused.",
        "",
        "## Family conclusions",
        "",
    ]
    conclusion_rows = [row for row in rows if row.category == "real"]
    if not conclusion_rows:
        conclusion_rows = rows
        lines += [
            "No real workloads measured. Synthetic results cannot justify deletion.",
            "",
        ]
    for family in families:
        available = [
            (r, p)
            for r in conclusion_rows
            for p in r.policies
            if p.family == family and p.error is not None
        ]
        nondominated = [
            (r.name, cell)
            for r, p in available
            if p.status == "qualified"
            for cell in r.dominators.get(family, {})
            if not any(r.dominance_same_companions[family][cell].values())
        ]
        lines += [
            f"### {family}",
            "",
            f"Available on {len(available)} inputs; "
            f"companion-blocked on {sum(p.companion_blocked is not None for r in conclusion_rows for p in r.policies if p.family == family)}; "
            f"unavailable on {sum(p.unavailable is not None for r in conclusion_rows for p in r.policies if p.family == family)}.",
        ]
        if family in DIRECTLY_RETAINED:
            lines.append(
                "Directly retained for its exclusive domain or last-candidate role. No dominance comparison applies; recorded errors remain unchanged."
            )
        elif not available:
            lines.append("NOT EXERCISED. No dominance conclusion is possible.")
        else:
            worst = max(p.error for _, p in available if p.error is not None)
            lines.append(
                f"Maximum density error {worst:.12g}: "
                f"{'FAIL' if any(p.status == 'over_limit' for _, p in available) else 'PASS'}."
            )
            if not any(p.status == "qualified" for _, p in available):
                lines.append(
                    "No qualified input. Over-limit results are not dominance evidence."
                )
            elif nondominated:
                lines.append(
                    "Not dominated within equal-companion groups on these input/mode/metric cells:"
                )
                lines += [f"- `{name}`: {cell}" for name, cell in nondominated]
            else:
                lines.append(
                    "Every measured qualified input/mode/metric cell is dominated; refused, over-limit and unmeasured inputs are not dominance evidence."
                )
        lines.append("")
    lines += [
        "## Complete measured matrix",
        "",
        "Unavailable cells retain their exact refusal and first-group whitelist classification; they have no error or timing value. not_candidate means absent from the reported candidates; guard_or_precondition means offered but rejected. Later groups are not classified.",
        "Only qualified non-retained pinned cells have dominance comparisons. Empty competitor lists on other rows do not establish non-dominance.",
        "Raw samples, per-sample paths and full inputs are in the sibling JSON.",
        "",
    ]
    category = None
    for row in sorted(rows, key=lambda row: row.category):
        if row.category != category:
            category = row.category
            lines += [f"## {category.capitalize()} inputs", ""]
        lines += [
            f"### {row.name}",
            "",
            f"Oracle convergence error: {row.oracle_convergence_error:.12g}; solve/cache-load wall {row.oracle_wall_seconds:.9g}s, CPU {row.oracle_cpu_seconds:.9g}s.",
            "",
            *[
                f"- {policy.family} ({policy.attribution}): whitelist {policy.whitelist}; expected (logical rows, ns) {policy.expected_route}."
                for policy in row.policies
                if policy.expected_route
            ],
            "",
            "| Family | Density error | Status | Mode/metric | Median seconds | Samples | Actual path | Dominators | Time-equivalent peers |",
            "| --- | ---: | --- | --- | ---: | ---: | --- | --- | --- |",
        ]
        lines += _policy_lines(row)
        lines.append("")
        lines += _render_fixed_group(row)
        lines += _family_cpu_lines(row)
    return "\n".join(lines) + "\n"
