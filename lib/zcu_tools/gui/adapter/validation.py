from __future__ import annotations

from .types import RunRequest, SocCfgHandle, SocHandle


def require_soc_handles(req: RunRequest) -> tuple[SocHandle, SocCfgHandle]:
    """Validate the real-hardware handles required by an adapter run."""
    if req.soc is None:
        raise RuntimeError("RunRequest.soc is required for real experiment adapters")
    if req.soccfg is None:
        raise RuntimeError("RunRequest.soccfg is required for real experiment adapters")
    return req.soc, req.soccfg
