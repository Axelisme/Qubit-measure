"""Center identifiability for the lab's GE diagnostics and calibration review."""

import math

from zcu_tools.analysis.fitting import FitQuality

_CENTER_LABELS = {"sg": "g-center", "se": "e-center"}
_MAX_RELATIVE_ERROR = 0.5
GE_CENTER_ERROR_PATHS = frozenset(
    f"relative_parameter_errors.{name}" for name in _CENTER_LABELS
)


def ge_center_relative_error_reasons(quality: FitQuality) -> list[str]:
    """Report missing, invalid or excessive center errors in g/e order.

    For a finite invalid error, report the value and requirement without
    judging the value against the threshold. Callers include invalid issues
    from GE_CENTER_ERROR_PATHS separately to preserve their joint issue order.
    """
    invalid_paths = {issue.path for issue in quality.invalid}
    reasons: list[str] = []
    for name, label in _CENTER_LABELS.items():
        error = quality.relative_parameter_errors.get(name)
        if error is None or not math.isfinite(error):
            reasons.append(
                f"relative {label} error unavailable (required <={_MAX_RELATIVE_ERROR})"
            )
        elif f"relative_parameter_errors.{name}" in invalid_paths:
            reasons.append(
                f"relative {label} error={error} (required <={_MAX_RELATIVE_ERROR})"
            )
        elif error > _MAX_RELATIVE_ERROR:
            reasons.append(f"relative {label} error={error} >{_MAX_RELATIVE_ERROR}")
    return reasons


def ge_centers_identifiable(quality: FitQuality) -> bool:
    """Check only the two joint relative center errors, not overall fit quality."""
    return not (
        any(issue.path in GE_CENTER_ERROR_PATHS for issue in quality.invalid)
        or ge_center_relative_error_reasons(quality)
    )
