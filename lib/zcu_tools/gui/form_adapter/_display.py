"""FieldMeta translation shared by core cfg fields and generated module fields."""

from pydantic.fields import FieldInfo

from zcu_tools.cfg_model import get_field_meta
from zcu_tools.gui.form import Display


def field_display(info: FieldInfo) -> Display | None:
    """Translate the field's FieldMeta to form Display metadata, or None without one.

    label, unit, decimals and description copy unchanged. The field is not
    modified; Display rejects negative decimals, and input_schema rejects
    decimals on a non-float field.
    """
    meta = get_field_meta(info)
    if meta is None:
        return None
    return Display(
        label=meta.label,
        unit=meta.unit,
        decimals=meta.decimals,
        description=meta.description,
    )
