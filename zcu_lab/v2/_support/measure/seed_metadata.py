"""Source descriptions for shared mux form initial values."""

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
