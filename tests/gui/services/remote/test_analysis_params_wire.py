from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from zcu_tools.experiment.v2_gui.adapters.onetone.freq import (
    OneToneFreqAnalyzeParams,
)
from zcu_tools.gui.app.main.services.remote.handlers.analysis import (
    _h_tab_analyze,
    _h_tab_get_analyze_params,
)
from zcu_tools.gui.remote.errors import RemoteError


def _adapter_with_params(params: OneToneFreqAnalyzeParams) -> MagicMock:
    control = MagicMock()
    control.has_tab.return_value = True
    control.get_tab_snapshot.return_value = SimpleNamespace(
        analyze_params=params,
        interaction=None,
    )
    control.analyze.return_value = "op-1"
    adapter = MagicMock()
    adapter.run_analyze_control = control
    return adapter


def test_remote_analyze_params_exposes_and_accepts_amplitude_slope_key() -> None:
    adapter = _adapter_with_params(OneToneFreqAnalyzeParams())

    reply = _h_tab_get_analyze_params(adapter, {"tab_id": "t"})
    assert reply["analyze_params"] == {
        "model_type": "hm",
        "fit_bg_amp_slope": True,
        "edelay_mode": "auto",
        "manual_edelay": None,
        "max_edelay_search_radius": 100.0,
    }

    started = _h_tab_analyze(
        adapter,
        {
            "tab_id": "t",
            "updates": {
                "fit_bg_amp_slope": False,
                "edelay_mode": "manual",
                "manual_edelay": 11.3,
                "max_edelay_search_radius": 150.0,
            },
        },
    )
    assert started == {"operation_id": "op-1"}
    forwarded = adapter.run_analyze_control.analyze.call_args.args[1]
    assert forwarded == OneToneFreqAnalyzeParams(
        fit_bg_amp_slope=False,
        edelay_mode="manual",
        manual_edelay=11.3,
        max_edelay_search_radius=150.0,
    )


def test_remote_analyze_params_rejects_removed_key() -> None:
    adapter = _adapter_with_params(OneToneFreqAnalyzeParams())
    removed_key = "_".join(("fit", "bg", "slope"))

    with pytest.raises(RemoteError):
        _h_tab_analyze(
            adapter,
            {"tab_id": "t", "updates": {removed_key: True}},
        )
