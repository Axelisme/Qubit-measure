"""Caller-owned YAML compile profiles shared by offline framework tests."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from qick import QickConfig
from zcu_tools.program import build_soccfg


def profile_soccfg(name: str) -> QickConfig:
    """Build a named board from tests/program/profiles/<name>.yaml."""
    with (Path(__file__).parent / "profiles" / f"{name}.yaml").open() as stream:
        return build_soccfg(yaml.safe_load(stream))


def _expand_channels(profile: dict[str, Any], n_gens: int, n_readouts: int) -> None:
    """Replace channels with detached copies wired in sequential fixture order."""
    gen = profile["generators"][0]
    readout = profile["readouts"][0]
    profile["generators"] = [
        {**deepcopy(gen), "dac": f"{ch:02d}", "tproc_ch": ch} for ch in range(n_gens)
    ]
    profile["readouts"] = [
        {
            **deepcopy(readout),
            "adc": f"{ch:02d}",
            "tproc_ctrl": n_gens + ch,
            "tproc_ch": ch,
            "trigger_bit": ch,
        }
        for ch in range(n_readouts)
    ]


def load_soccfg(n_gens: int | None = None, n_readouts: int | None = None) -> QickConfig:
    """Load ordinary capabilities with optional generator/readout counts."""
    with (Path(__file__).parent / "profiles" / "ordinary.yaml").open() as stream:
        profile = yaml.safe_load(stream)
    _expand_channels(
        profile,
        len(profile["generators"]) if n_gens is None else n_gens,
        len(profile["readouts"]) if n_readouts is None else n_readouts,
    )
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
    mux = profile["mux"][0]
    b_phase = profile["readouts"][mux["ro_chs"][0]]["b_phase"]
    _expand_channels(profile, n_gens, n_readouts)
    for port, ch in enumerate(ro_chs):
        profile["readouts"][ch].update(
            tproc_ctrl=None, trigger_bit=port, b_phase=b_phase
        )
    mux.update(gen_ch=gen_ch, ro_chs=list(ro_chs))
    return build_soccfg(profile)
