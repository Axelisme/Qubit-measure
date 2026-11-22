"""Center identifiability for the lab's GE diagnostics and calibration review."""

import math

from zcu_tools.analysis.fitting import FitQuality

_CENTER_LABELS = {"sg": "g-center", "se": "e-center"}
_MAX_RELATIVE_ERROR = 0.5


def ge_center_relative_error_reasons(quality: FitQuality) -> list[str]:
    """Explain joint relative center errors that fail the <=0.5 gate.

    Report each center in g/e order, listing its relative-error issues before
    its value or unavailability. Issues on the standard errors alone do not
    count. The returned list is empty when both centers pass.
    """
    reasons: list[str] = []
    for name, label in _CENTER_LABELS.items():
        path = f"relative_parameter_errors.{name}"
        issues = [
            f"joint {issue.path}: {issue.reason}"
            for issue in quality.invalid
            if issue.path == path
        ]
        reasons.extend(issues)
        error = quality.relative_parameter_errors.get(name)
        if error is None or not math.isfinite(error):
            reasons.append(
                f"relative {label} error unavailable (required <={_MAX_RELATIVE_ERROR})"
            )
        elif issues:
            reasons.append(
                f"relative {label} error={error} (required <={_MAX_RELATIVE_ERROR})"
            )
        elif error > _MAX_RELATIVE_ERROR:
            reasons.append(f"relative {label} error={error} >{_MAX_RELATIVE_ERROR}")
    return reasons


def ge_centers_identifiable(quality: FitQuality) -> bool:
    """Check only the two joint relative center errors, not overall fit quality."""
    return not ge_center_relative_error_reasons(quality)
