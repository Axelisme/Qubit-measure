"""Standalone native envelope fields use the same ref/inline Form contract as modules."""

from pathlib import Path

import numpy as np
import pytest
from zcu_tools.cfg_model import ArbWaveformCfg as CoreArbWaveformCfg
from zcu_tools.cfg_model import Slot
from zcu_tools.gui.form import FormEnv, FormErrors, FormState
from zcu_tools.gui.form_adapter import Form
from zcu_tools.program.v2.modules.waveform import (
    ArbWaveformCfg,
    CosineWaveformCfg,
    GaussWaveformCfg,
)
from zcu_tools.resources.context.library import (
    ComponentLibrary,
    LibrarySnapshot,
    ModuleLibraryDocument,
)
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from ._waveform_models import EnvelopePlan, envelope_env, envelope_form


def test_standalone_envelopes_filter_sources_and_materialize_independently() -> None:
    env = envelope_env()
    unseeded = Form(EnvelopePlan, {})
    labels = [
        option.label for option in unseeded.model.options(env).get(("first",), ())
    ]
    assert labels == [
        "Q.waveforms.shaped",
        "Inline: cosine",
        "Inline: gauss",
        "Inline: drag",
        "Inline: arb",
    ]
    form = envelope_form()
    state = FormState(form.default_input(env))
    selected = form.prepare_logical_selection(
        state.input, ("first",), "Q.waveforms.shaped", env
    )
    mixed = form.prepare_logical_selection(
        selected.input, ("second",), "Inline: cosine", env
    )
    state = FormState(mixed.input)
    result = state.result(env)
    assert not isinstance(result, FormErrors), result
    cfg = form.to_cfg(result, env)
    assert not isinstance(cfg, FormErrors), cfg
    assert cfg.first == GaussWaveformCfg(length=0.48, sigma=0.08)
    assert cfg.second == CosineWaveformCfg(length=0.48)
    copied = form.prepare_logical_command(state.input, ("first",), "to_inline", env)
    observation = form.read_module(copied.input, ("first",), env)
    assert not isinstance(observation, FormErrors), observation
    assert observation.mode == "inline"
    assert observation.value["style"] == "gauss"
    assert observation.value["sigma"] == 0.08


@pytest.mark.parametrize("copy", [False, True])
def test_standalone_waveform_slot_uses_captured_family(copy: bool) -> None:
    form = Form(
        EnvelopePlan,
        {
            "first": Slot("Q.waveforms.shaped", copy=copy),
            "second": Slot("Q.waveforms.shaped"),
        },
    )
    env = envelope_env()
    state = FormState(form.default_input(env))
    result = state.result(env)
    assert not isinstance(result, FormErrors), result
    cfg = form.to_cfg(result, env)
    assert not isinstance(cfg, FormErrors), cfg
    assert cfg.first == cfg.second == GaussWaveformCfg(length=0.48, sigma=0.08)
    observed = form.read_module(state.input, ("first",), env)
    assert not isinstance(observed, FormErrors), observed
    assert observed.mode == ("inline" if copy else "key")


def test_standalone_arb_materializes_path_from_fixed_asset_capture(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "asset.npz"
    np.savez(
        asset,
        time=np.array([0.0, 0.1, 0.2]),
        idata=np.array([0.0, 0.5, 0.0]),
        qdata=np.zeros(3),
    )
    env = FormEnv(
        library=LibrarySnapshot(
            ModuleLibraryDocument(
                format="zcu.module-library",
                format_version="2.0",
                components={
                    "Q": ComponentLibrary(
                        waveforms={"sample": CoreArbWaveformCfg(data="asset")}
                    )
                },
            ),
            assets=ArbWaveformDatabase.info_at(tmp_path),
        )
    )
    form = Form(
        EnvelopePlan,
        {
            "first": Slot("Q.waveforms.sample"),
            "second": Slot("Q.waveforms.sample", copy=True),
        },
    )
    state = FormState(form.default_input(env))
    result = state.result(env)
    assert not isinstance(result, FormErrors), result
    cfg = form.to_cfg(result, env)
    assert not isinstance(cfg, FormErrors), cfg
    assert (
        cfg.first == cfg.second == ArbWaveformCfg(data="asset", path=asset.as_posix())
    )
    assert cfg.first.length == 0.2
