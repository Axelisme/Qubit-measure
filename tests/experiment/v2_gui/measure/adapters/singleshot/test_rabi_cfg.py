"""Rabi calibration follows the visible cfg instead of runtime md reads."""

from unittest.mock import MagicMock

import pytest
from zcu_tools.experiment.cfg_assembler import assemble_experiment_cfg
from zcu_tools.experiment.utils import make_comment, parse_comment
from zcu_tools.experiment.v2.singleshot import amp_rabi, len_rabi
from zcu_tools.experiment.v2.singleshot.amp_rabi import AmpRabiCfg, AmpRabiExp
from zcu_tools.experiment.v2.singleshot.len_rabi import LenRabiCfg, LenRabiExp
from zcu_tools.experiment.v2_gui.measure.adapters.singleshot.amp_rabi import (
    SsAmpRabiAdapter,
)
from zcu_tools.experiment.v2_gui.measure.adapters.singleshot.len_rabi import (
    SsLenRabiAdapter,
)
from zcu_tools.gui.app.measure.adapter import RunRequest, SessionEnv
from zcu_tools.gui.app.measure.adapter.lowering import (
    schema_to_raw_dict,
    schema_to_resolved_dict,
)
from zcu_tools.gui.app.measure.cfg_binding import MeasureCfgBindings
from zcu_tools.gui.cfg import DirectValue, EvalValue
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.resources.context import MetaDict, ModuleLibrary


@pytest.fixture(
    params=[
        (SsAmpRabiAdapter, AmpRabiCfg, AmpRabiExp, amp_rabi),
        (SsLenRabiAdapter, LenRabiCfg, LenRabiExp, len_rabi),
    ],
    ids=["amp", "len"],
)
def rabi_api(request):
    return request.param


@pytest.fixture
def calibration_draft(rabi_api):
    adapter_type, _, _, _ = rabi_api
    md = MetaDict()
    md.g_center = -1 + 0.25j
    md.e_center = 2 - 0.5j
    md.ge_radius = 0.75
    ml = ModuleLibrary()
    ctx = SessionEnv(md=md, ml=ml, soc=None, soccfg=None)
    host = MagicMock()
    host.get_current_md.return_value = md
    host.get_current_ml.return_value = ml
    host.list_device_names.return_value = []
    host.list_arb_waveforms.return_value = []
    schema = adapter_type.cfg_definition().instantiate(ctx)
    draft = MeasureCfgBindings(host).new_draft(schema)
    try:
        yield md, ml, draft
    finally:
        draft.close()


def test_rabi_missing_calibration_is_invalid_and_cannot_acquire(
    calibration_draft, rabi_api, monkeypatch
):
    adapter_type, _, experiment_type, _ = rabi_api
    _md, _ml, draft = calibration_draft
    draft.set_target("g_center", EvalValue("missing_center"))
    assert not draft.is_valid()
    acquire = MagicMock()
    monkeypatch.setattr(experiment_type, "run", acquire)
    request = RunRequest(soc=MagicMock(), soccfg=MagicMock(), device_snapshot={})
    with pytest.raises(RuntimeError, match="missing_center"):
        adapter_type().run(request, schema_to_resolved_dict(draft.snapshot()))
    acquire.assert_not_called()


def test_rabi_uses_cfg_calibration_after_md_changes(
    calibration_draft, rabi_api, monkeypatch
):
    adapter_type, _, experiment_type, _ = rabi_api
    md, _ml, draft = calibration_draft
    snapshot = draft.snapshot()
    assert snapshot.value.fields["g_center"] == EvalValue(
        "g_center", resolved=-1 + 0.25j
    )
    assert snapshot.value.fields["e_center"] == EvalValue("e_center", resolved=2 - 0.5j)
    assert snapshot.value.fields["radius"] == EvalValue("ge_radius", resolved=0.75)
    md.g_center = 100 + 100j
    observed = []

    def record_run(self, cfg, *, context):
        observed.append((cfg, context))
        return "acquired"

    monkeypatch.setattr(experiment_type, "run", record_run)
    request = RunRequest(soc=MagicMock(), soccfg=MagicMock(), device_snapshot={})
    plots = Plots(NonPresentingHost())
    try:
        result = adapter_type().run(
            request, schema_to_resolved_dict(snapshot), plots=plots
        )
        assert result == "acquired"
        cfg, context = observed[0]
        assert context.soc is request.soc
        assert context.soccfg is request.soccfg
        assert context.plots is plots
    finally:
        plots.finish(present=False)
        plots.release()
    assert (cfg.g_center, cfg.e_center, cfg.radius) == (-1 + 0.25j, 2 - 0.5j, 0.75)


def test_rabi_experiment_result_preserves_calibration_used_for_live_classification(
    calibration_draft, rabi_api, monkeypatch
):
    import numpy as np
    from zcu_tools.experiment.v2.runtime import ProgramBuilder

    _, cfg_type, experiment_type, experiment_module = rabi_api
    md, ml, draft = calibration_draft
    draft.set_target("reps" if cfg_type is AmpRabiCfg else "shots", 2)
    cfg = assemble_experiment_cfg(
        schema_to_raw_dict(draft.snapshot(), md, ml),
        cfg_type,
        ml=ml,
        device_snapshot={},
    )
    viewer = MagicMock()
    viewer.__enter__.return_value = viewer
    monkeypatch.setattr(experiment_module, "LivePlot1D", lambda *args, **kwargs: viewer)
    monkeypatch.setattr(
        experiment_module, "setup_devices", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        experiment_module, "sweep2array", lambda *args, **kwargs: np.array([0.1])
    )
    monkeypatch.setattr(ProgramBuilder, "build", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(
        experiment_module,
        "raw_shots_to_signal",
        lambda _program: (
            np.array([[-1 + 0.25j], [2 - 0.5j]])
            if cfg_type is AmpRabiCfg
            else np.array([-1 + 0.25j, 2 - 0.5j])
        ),
    )
    if cfg_type is AmpRabiCfg:
        result = experiment_type().run(MagicMock(), MagicMock(), cfg)
    else:
        with pytest.warns(UserWarning, match="reps will be overwritten"):
            result = experiment_type().run(MagicMock(), MagicMock(), cfg)
    np.testing.assert_array_equal(
        viewer.update.call_args.args[1], [[0.5], [0.5], [0.0]]
    )
    assert result.cfg_snapshot is not None
    assert result.cfg_snapshot.g_center == -1 + 0.25j
    assert result.cfg_snapshot.e_center == 2 - 0.5j
    assert result.cfg_snapshot.radius == 0.75
    cfg.g_center = 10j
    assert result.cfg_snapshot.g_center == -1 + 0.25j


def test_rabi_direct_override_and_experiment_comment_round_trip(
    calibration_draft, rabi_api
):
    _, cfg_type, _, _ = rabi_api
    md, ml, draft = calibration_draft
    draft.set_target("g_center", DirectValue(3 + 4j))
    raw = schema_to_raw_dict(draft.snapshot(), md, ml)
    cfg = assemble_experiment_cfg(raw, cfg_type, ml=ml, device_snapshot={})
    assert cfg.g_center == 3 + 4j
    assert md.g_center == -1 + 0.25j
    saved, _, _ = parse_comment(make_comment(cfg))
    restored = cfg_type.model_validate(saved)
    assert (restored.g_center, restored.e_center, restored.radius) == (
        3 + 4j,
        2 - 0.5j,
        0.75,
    )
