"""Source descriptions for shared pulse and mux form initial values."""

from zcu_tools.cfg_model import FieldMeta


def seed_metadata(source: str, *, fallback: bool = False) -> FieldMeta:
    """Describe a captured source, or flag an authorized fixed-constant fallback.

    source names the captured paths/formula or explains the constant and units.
    fallback=True adds the operator warning and uncalibrated diagnostic. Other
    sources retain unknown calibration status; they are not marked measured.
    """
    if fallback:
        return FieldMeta(
            description=f"未校準的起始值，來源：固定常數。{source}",
            calibration="uncalibrated",
        )
    return FieldMeta(description=source)


def qub_probe_gain_seed_metadata(
    path: tuple[str | int, ...],
) -> dict[tuple[str | int, ...], FieldMeta]:
    """Describe the uncalibrated gain 1.0 a.u. authored by a qub_probe Template.

    path is the root-relative canonical Form gain path, including module
    wrappers and list indexes. Attach this metadata to the Template only;
    captured slots retain their existing unknown calibration status.
    No cfg values, sources or calibration records are changed.
    """
    return {
        path: FieldMeta(
            description="未校準的起始值，來源：固定常數。qub_probe Template gain 1.0 a.u. within [-1, 1]; operator confirmation required",
            calibration="uncalibrated",
        )
    }


def pulse_readout_seed_metadata(
    *, gain: float
) -> dict[tuple[str | int, ...], FieldMeta]:
    """Describe the actual gain in a.u. within [-1, 1] authored by a readout Template/Form.

    gain must match the caller\'s candidate: the shared Template supplies 0.1;
    one-tone Forms override it with 0.05. Callers using a captured slot must
    not apply this metadata: stored settings remain unknown.
    No cfg values, sources, units or calibration records are changed.
    """
    return {
        ("readout", "module", "gain"): FieldMeta(
            description=f"未校準的起始值，來源：固定常數。readout Template/Form gain {gain:g} a.u. within [-1, 1]; operator confirmation required",
            calibration="uncalibrated",
        ),
    }
