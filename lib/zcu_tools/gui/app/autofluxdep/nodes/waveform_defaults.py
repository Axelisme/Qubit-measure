from __future__ import annotations

from typing import Any


def waveform_or_const(ml: Any, name: object, *, length: object) -> Any:
    """Resolve a named waveform when present, otherwise use an inline const pulse."""
    if isinstance(name, str) and name and name in ml.waveforms:
        return ml.get_waveform(name, {"length": length})
    return {"style": "const", "length": length}
