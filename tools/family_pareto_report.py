"""Human-readable real/synthetic Pareto diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests.simulate.evolution.contract.accuracy import ALL_FAMILIES

if TYPE_CHECKING:
    from tools.family_pareto import InputResult


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


def _policy_lines(row: InputResult) -> list[str]:
    lines: list[str] = []
    for policy in row.policies:
        if policy.error is None:
            lines.append(
                f"| {policy.family} | — | unavailable: {policy.unavailable} | — | — | — | — | — | — |"
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
                peers = row.timing_equivalents.get(policy.family, {}).get(cell, [])
                lines.append(
                    f"| {policy.family} | {policy.error:.12g} | {'PASS' if policy.error <= 2e-3 else 'FAIL'} | "
                    f"{cell} | {value:.9g} | {len(mode.samples)} | {mode.samples[0].paths} | {', '.join(competitors) or 'none'} | {', '.join(peers) or 'none'} |"
                )
    return lines


def render_summary(
    rows: list[InputResult], families: tuple[str, ...] = ALL_FAMILIES
) -> str:
    """Render per-family conclusions and all measured mode/clock cells.

    Dominance is limited to measured inputs. No family deletion or accuracy
    exception follows from this report. Default is shown but never a competitor.
    """
    lines = [
        "# Engine family Pareto table",
        "",
        "Timers cover Engine.evolve only, not program.acquire. Normal modes have five samples. A first evolve >60 wall seconds keeps n=1, skips clear-history with n=0, then retains history in the same session for n=1. No separate probe or warmup runs for slow families.",
        "Errors compare complete complex density to the two-level converged full-Lindblad oracle.",
        "A dominator has mode-local maximum sample error <= target error and time < 0.9 * target time. Exactly 10% is equivalent.",
        "Displayed density error and PASS/FAIL use the overall maximum across probe and all modes, not the mode-local comparator error.",
        "No epsilon or weighted score is used. FAIL means error > 2e-3, not permission to keep an inaccurate route.",
        "Conclusions apply only to the measured input set; no deletion is authorized.",
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
            for r, _ in available
            for cell, competitors in r.dominators[family].items()
            if not competitors
        ]
        lines += [
            f"### {family}",
            "",
            f"Available on {len(available)} inputs; unavailable on {len(conclusion_rows) - len(available)}.",
        ]
        if not available:
            lines.append("NOT EXERCISED. No dominance conclusion is possible.")
        else:
            worst = max(p.error for _, p in available if p.error is not None)
            lines.append(
                f"Maximum density error {worst:.12g}: {'PASS' if worst <= 2e-3 else 'FAIL'}."
            )
            if nondominated:
                lines.append("Not dominated on these input/mode/metric cells:")
                lines += [f"- `{name}`: {cell}" for name, cell in nondominated]
            else:
                lines.append(
                    "Every measured available input/mode/metric cell is dominated; this is not a proof for unmeasured inputs."
                )
        lines.append("")
    lines += [
        "## Complete measured matrix",
        "",
        "Unavailable cells retain their exact refusal; they have no error or timing value.",
        "Each measured cell lists its dominators. An empty list means not dominated.",
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
                f"- {policy.family} ({policy.attribution}): domains {policy.domains}; expected segments {policy.expected_segments}; full-input subset refusals {policy.subset_refusals}."
                for policy in row.policies
                if policy.domains
            ],
            "",
            "| Family | Density error | Status | Mode/metric | Median seconds | Samples | Actual path | Dominators | Time-equivalent peers |",
            "| --- | ---: | --- | --- | ---: | ---: | --- | --- | --- |",
        ]
        lines += _policy_lines(row)
        lines.append("")
        lines += _render_fixed_group(row)
    return "\n".join(lines) + "\n"
