"""User-owned experiment definitions, alias preset and frontend attachments."""

from zcu_tools.gui.session.aliases import AliasRow

# Ordered session-only defaults. Device binding remains an explicit user action.
ALIAS_PRESET: tuple[AliasRow, ...] = (
    AliasRow("qubit", ""),
    AliasRow("resonator", "$qubit.resonator"),
    AliasRow("flux_dev", ""),
)

__all__ = ["ALIAS_PRESET"]
