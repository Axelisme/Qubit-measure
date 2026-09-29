"""Population acquisition receives the calibration displayed in its cfg."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from zcu_tools.experiment.utils import make_comment, parse_comment
from zcu_tools.experiment.v2.runtime import ProgramBuilder
from zcu_tools.experiment.v2.singleshot import ac_stark
from zcu_tools.experiment.v2.singleshot.mist import freq, power, power_freq
from zcu_tools.experiment.v2.singleshot.t1 import t1, t1_with_tone, t1_with_tone_sweep
from zcu_tools.experiment.v2_gui.measure.adapters.singleshot.ac_stark import (
    SsAcStarkAdapter,
)
from zcu_tools.experiment.v2_gui.measure.adapters.singleshot.mist.freq import (
    MistFreqAdapter,
)
from zcu_tools.experiment.v2_gui.measure.adapters.singleshot.mist.power import (
    MistPowerAdapter,
)
from zcu_tools.experiment.v2_gui.measure.adapters.singleshot.mist.power_freq import (
    MistPowerFreqAdapter,
)
from zcu_tools.experiment.v2_gui.measure.adapters.singleshot.t1 import SsT1Adapter
from zcu_tools.experiment.v2_gui.measure.adapters.singleshot.t1_tone import (
    SsT1ToneAdapter,
)
from zcu_tools.experiment.v2_gui.measure.adapters.singleshot.t1_tone_sweep import (
    SsT1ToneSweepFreqAdapter,
    SsT1ToneSweepGainAdapter,
)
from zcu_tools.gui.app.measure.adapter import RunRequest, SessionEnv
from zcu_tools.gui.app.measure.adapter.lowering import schema_to_resolved_dict
from zcu_tools.gui.app.measure.cfg_binding import MeasureCfgBindings
from zcu_tools.gui.cfg import DirectValue, EvalValue
from zcu_tools.resources.context import MetaDict, ModuleLibrary


@pytest.fixture(
    params=[
        (SsAcStarkAdapter, ac_stark),
        (MistFreqAdapter, freq),
        (MistPowerAdapter, power),
        (MistPowerFreqAdapter, power_freq),
        (SsT1Adapter, t1),
        (SsT1ToneAdapter, t1_with_tone),
        (SsT1ToneSweepGainAdapter, t1_with_tone_sweep),
        (SsT1ToneSweepFreqAdapter, t1_with_tone_sweep),
    ],
    ids=[
        "ac-stark",
        "mist-freq",
        "mist-power",
        "mist-power-freq",
        "t1",
        "t1-tone",
        "t1-tone-gain",
        "t1-tone-freq",
    ],
)
def calibration(request, monkeypatch):
    adapter_type, module = request.param
    md = MetaDict()
    md.g_center = -1 + 0.25j
    md.e_center = 2 - 0.5j
    md.ge_radius = 0.75
    md.q_f = 4000.0
    md.r_f = 6200.0
    md.readout_f = 6210.0
    md.res_ch = 2
    md.qub_ch = 3
    md.rf_w = 4.0
    ml = ModuleLibrary()
    host = MagicMock()
    host.get_current_md.return_value = md
    host.get_current_ml.return_value = ml
    host.list_device_names.return_value = []
    host.list_arb_waveforms.return_value = []
    schema = adapter_type.cfg_definition().instantiate(
        SessionEnv(md=md, ml=ml, soc=None, soccfg=None)
    )
    draft = MeasureCfgBindings(host).new_draft(schema)
    if adapter_type in (
        SsT1Adapter,
        SsT1ToneAdapter,
        SsT1ToneSweepGainAdapter,
        SsT1ToneSweepFreqAdapter,
    ):
        draft.set_target("uniform", True)

    monkeypatch.setattr(module, "setup_devices", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "sweep2array", lambda *args, **kwargs: np.array([0.1]))
    viewer = MagicMock()
    viewer.__enter__.return_value = viewer
    for name in ("LivePlot1D", "LivePlot2D", "MultiLivePlot"):
        monkeypatch.setattr(module, name, lambda *args, **kwargs: viewer, raising=False)
    monkeypatch.setattr(
        module,
        "make_plot_frame",
        lambda rows, cols, **kwargs: (
            MagicMock(),
            [[MagicMock() for _ in range(cols)] for _ in range(rows)],
        ),
        raising=False,
    )
    monkeypatch.setattr(module, "close_figure", lambda *args: None, raising=False)
    acquisition = MagicMock()
    monkeypatch.setattr(ProgramBuilder, "build_and_acquire", acquisition)
    try:
        yield adapter_type(), md, ml, draft, acquisition
    finally:
        draft.close()


def test_population_run_uses_cfg_calibration_and_preserves_result_snapshot(calibration):
    adapter, md, _ml, draft, acquisition = calibration
    snapshot = draft.snapshot()
    assert snapshot.value.fields["g_center"] == EvalValue(
        "g_center", resolved=-1 + 0.25j
    )
    md.g_center = 100j
    result = adapter.run(
        RunRequest(soc=MagicMock(), soccfg=MagicMock(), device_snapshot={}),
        schema_to_resolved_dict(snapshot),
    )
    acquisition.assert_called()
    for call in acquisition.call_args_list:
        assert (
            call.kwargs["g_center"],
            call.kwargs["e_center"],
            call.kwargs["ge_radius"],
        ) == (-1 + 0.25j, 2 - 0.5j, 0.75)
    cfg = result.cfg_snapshot
    assert (cfg.g_center, cfg.e_center, cfg.radius) == (-1 + 0.25j, 2 - 0.5j, 0.75)
    draft.set_target("g_center", DirectValue(5j))
    assert cfg.g_center == -1 + 0.25j
    saved, _, _ = parse_comment(make_comment(cfg))
    restored = adapter.ExpCfg_cls.model_validate(saved)
    assert (restored.g_center, restored.e_center, restored.radius) == (
        -1 + 0.25j,
        2 - 0.5j,
        0.75,
    )


def test_population_direct_override_does_not_write_md(calibration):
    adapter, md, _ml, draft, acquisition = calibration
    draft.set_target("g_center", DirectValue(3 + 4j))
    result = adapter.run(
        RunRequest(soc=MagicMock(), soccfg=MagicMock(), device_snapshot={}),
        schema_to_resolved_dict(draft.snapshot()),
    )
    assert result.cfg_snapshot.g_center == 3 + 4j
    assert acquisition.call_args.kwargs["g_center"] == 3 + 4j
    assert md.g_center == -1 + 0.25j


def test_population_invalid_calibration_rejects_acquisition(calibration):
    adapter, _md, _ml, draft, acquisition = calibration
    draft.set_target("g_center", EvalValue("missing_center"))
    assert not draft.is_valid()
    with pytest.raises(RuntimeError, match="missing_center"):
        adapter.run(
            RunRequest(soc=MagicMock(), soccfg=MagicMock(), device_snapshot={}),
            schema_to_resolved_dict(draft.snapshot()),
        )
    acquisition.assert_not_called()
