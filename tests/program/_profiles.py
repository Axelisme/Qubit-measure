"""Caller-owned YAML compile profiles shared by offline framework tests."""

from pathlib import Path

import yaml
from qick import QickConfig
from zcu_tools.program import build_soccfg


def load_soccfg(
    n_gens: int | None = None, n_readouts: int | None = None
) -> QickConfig:
    """Load the ordinary test YAML; optionally select channel group counts."""
    with (Path(__file__).parent / "profiles" / "ordinary.yaml").open() as stream:
        profile = yaml.safe_load(stream)
    if n_gens is not None:
        profile["generators"][0]["count"] = n_gens
    if n_readouts is not None:
        profile["readouts"][0]["count"] = n_readouts
    return build_soccfg(profile)


def load_mux_soccfg(
    n_gens: int, n_readouts: int, *, gen_ch: int, ro_chs: tuple[int, ...]
) -> QickConfig:
    """Load test mux capabilities with caller-selected channel counts and wiring."""
    with (Path(__file__).parent / "profiles" / "mux.yaml").open() as stream:
        profile = yaml.safe_load(stream)
    profile["generators"][0]["count"] = n_gens
    profile["readouts"][0]["count"] = n_readouts
    profile["mux"][0].update(gen_ch=gen_ch, ro_chs=list(ro_chs))
    return build_soccfg(profile)
