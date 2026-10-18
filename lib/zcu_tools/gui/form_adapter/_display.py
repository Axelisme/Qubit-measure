"""FieldMeta translation shared by core cfg fields and generated module fields."""

from pydantic.fields import FieldInfo

from zcu_tools.cfg_model import get_field_meta
from zcu_tools.gui.form import Display


def field_display(info: FieldInfo, *, locked: bool) -> Display | None:
    """Translate the field's FieldMeta to form Display metadata, or None without one.

    label, unit, description and group copy unchanged. decimals copies unless
    locked is True: a locked field becomes a single-value Literal, which has no
    float precision to show. The field is not modified; Display rejects
    negative decimals, and describe rejects decimals on a non-float field.
    """
    meta = get_field_meta(info)
    if meta is None:
        return None
    return Display(
        label=meta.label,
        unit=meta.unit,
        decimals=None if locked else meta.decimals,
        description=meta.description,
        group=meta.group,
        readonly=meta.readonly,
        element_label=meta.element_label,
    )
