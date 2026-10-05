"""Shared repetition declarations for the plain and scan ZigZag adapters."""

from zcu_lab.v2._support.measure import MeasureCfgBuilder

_REPEAT_ON_CHOICES = ("X180_pulse", "X90_pulse")

EXPECTS_ML = (
    "Needs an X90 pulse module (prefers the calibrated library pi/2 pulse "
    "'pi2_amp' / 'pi2_len'), an X180 pulse module (prefers 'pi_amp' / "
    "'pi_len'), and a readout module (calibrated 'readout_dpm' / 'readout_rf' "
    "/ 'readout' / 'res_readout', else a blank pulse-readout). Optional reset "
    "references a library reset when present, else stays disabled."
)


def add_gate_modules(builder: MeasureCfgBuilder) -> MeasureCfgBuilder:
    """Declare reset, X90/X180 pulse roles and readout on the given builder.

    Return the same builder. Role defaults come from MeasureCfgBuilder;
    duplicate or invalid declarations propagate its errors.
    """
    return (
        builder.reset(optional=True)
        .pulse("X90_pulse", role_id="pi2_pulse", label="X90 Pulse")
        .pulse("X180_pulse", role_id="pi_pulse", label="X180 Pulse")
        .readout()
    )


def add_repeat_fields(builder: MeasureCfgBuilder, *, n_times: int) -> MeasureCfgBuilder:
    """Add repetition count and pulse choice fields to the given builder.

    Use n_times as the form's default maximum repetition count, and return the
    same builder. Duplicate or invalid declarations propagate builder errors.
    """
    return builder.int(
        "n_times",
        label="Max repetitions",
        default=n_times,
        tooltip="Repetition counts run from 0 to this value.",
    ).choice(
        "repeat_on",
        label="Repeat pulse",
        choices=_REPEAT_ON_CHOICES,
        default="X180_pulse",
        tooltip="X90_pulse repeats in pairs, so each count adds one pi rotation.",
    )
