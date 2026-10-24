"""Seed provenance for the shared pulse-readout template amplitude."""

from zcu_tools.cfg_model import FieldMeta


def pulse_readout_seed_metadata(
    *, gain: float
) -> dict[tuple[str | int, ...], FieldMeta]:
    """Describe the actual normalized gain authored by a readout Template/Form.

    gain must match the caller\'s candidate: the shared Template supplies 0.1;
    one-tone Forms override it with 0.05. Callers using a captured slot must
    not apply this metadata: stored settings remain unknown.
    No cfg values, sources, units or calibration records are changed.
    """
    return {
        ("readout", "module", "gain"): FieldMeta(
            description=f"未校準的起始值，來源：固定常數。readout Template/Form gain {gain:g} normalized; operator confirmation required",
            calibration="uncalibrated",
        ),
    }
