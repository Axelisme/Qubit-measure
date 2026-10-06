"""Core waveform knobs and non-serialized asset-binding contracts."""

from copy import deepcopy

from zcu_tools.cfg_model import CfgValidationContext, WaveformCfgFactory

from ._support import AssetInfo


def test_deepcopy_preserves_current_asset_binding():
    assets = AssetInfo()
    cfg = WaveformCfgFactory.from_raw(
        {"style": "arb", "data": "shape"},
        context=CfgValidationContext(assets=assets),
    )
    copied = deepcopy(cfg)

    assets.current_duration = 4.0

    assert cfg.length == 4.0
    assert copied.length == 4.0
    assert copied.to_dict() == {"style": "arb", "data": "shape"}
