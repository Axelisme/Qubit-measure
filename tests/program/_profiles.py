"""Caller-owned YAML compile profiles shared by offline framework tests."""

from pathlib import Path

import yaml
from qick import QickConfig
from zcu_tools.program import build_soccfg


def profile_soccfg(name: str) -> QickConfig:
    """Build a named board from tests/program/profiles/<name>.yaml."""
    with (Path(__file__).parent / "profiles" / f"{name}.yaml").open() as stream:
        return build_soccfg(yaml.safe_load(stream))


def load_soccfg(n_gens: int | None = None, n_readouts: int | None = None) -> QickConfig:
    """Load the ordinary test YAML; optionally select channel group counts."""
    with (Path(__file__).parent / "profiles" / "ordinary.yaml").open() as stream:
        profile = yaml.safe_load(stream)
    if n_gens is not None:
        profile["generators"][0]["count"] = n_gens
    if n_readouts is not None:
        profile["readouts"][0]["count"] = n_readouts
    return build_soccfg(profile)


# Cycle-count scenarios use the same caller-owned YAML as their compiled programs.
_ordinary = load_soccfg()
TPROC_MHZ = _ordinary["tprocs"][0]["f_time"]
GEN_MHZ = _ordinary["gens"][0]["f_fabric"]
RO_MHZ = _ordinary["readouts"][0]["f_output"]
GEN_SAMPLE_MHZ = GEN_MHZ * _ordinary["gens"][0]["samps_per_clk"]
GEN_MAXV = _ordinary["gens"][0]["maxv"]


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
