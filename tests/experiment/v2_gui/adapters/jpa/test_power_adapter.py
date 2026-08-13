"""jpa/power adapter — registry reachability, device preflight, run/persistence,
analysis writeback and operator guide."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from matplotlib.figure import Figure
from zcu_tools.device import (
    AnritsuMG3692Info,
    FakeDevice,
    GlobalDeviceManager,
    RohdeSchwarzSGS100AInfo,
    YOKOGS200Info,
)
from zcu_tools.experiment.v2.jpa import PowerCfg, PowerExp
from zcu_tools.experiment.v2.jpa.jpa_power import PowerResult
from zcu_tools.experiment.v2_gui.adapters._support import MeasureCfgDefinition
from zcu_tools.experiment.v2_gui.adapters.jpa import (
    JpaPowerAdapter,
    JpaPowerAnalyzeResult,
)
from zcu_tools.experiment.v2_gui.adapters.jpa._shared import (
    JPA_RF_LABEL,
    JPA_RF_ROLE_KEY,
    lower_jpa_rf_power_dev,
)
from zcu_tools.experiment.v2_gui.registry import ADAPTERS, register_all
from zcu_tools.gui.app.main.adapter import (
    AnalyzeRequest,
    ExpAdapterProtocol,
    LoadDataRequest,
    MetaDictWriteback,
    NoAnalyzeParams,
    RunRequest,
    SaveDataRequest,
    WritebackRequest,
)
from zcu_tools.gui.app.main.adapter.lowering import schema_to_raw_dict
from zcu_tools.gui.app.main.registry import Registry
from zcu_tools.gui.cfg import (
    CfgSectionValue,
    DirectValue,
    EvalValue,
    SweepValue,
)
from zcu_tools.meta_tool import MetaDict, ModuleLibrary

_SGS = RohdeSchwarzSGS100AInfo(address="sgs")
_SGS2 = RohdeSchwarzSGS100AInfo(address="sgs2")
_ANRITSU = AnritsuMG3692Info(address="anritsu")
_YOKO = YOKOGS200Info(address="yoko")


def _make_ml() -> MagicMock:
    ml = MagicMock()
    ml.modules = {}
    ml.waveforms = {}
    ml.make_cfg.return_value = MagicMock()
    return ml


def _make_ctx(ml: MagicMock | None = None, **md_values: float) -> MagicMock:
    ctx = MagicMock()
    ctx.ml = ml or _make_ml()
    md = MetaDict()
    for key, value in md_values.items():
        setattr(md, key, value)
    ctx.md = md
    return ctx


def _make_req(ml: MagicMock | None = None, **kwargs: Any) -> RunRequest:
    return RunRequest(
        md=kwargs.pop("md", MetaDict()),
        ml=ml or _make_ml(),
        soc=kwargs.pop("soc", None),
        soccfg=kwargs.pop("soccfg", None),
    )


def _raw_with_selection(**dev_selection: str) -> dict[str, object]:
    return {"dev": {"jpa_rf_dev": dev_selection.get("jpa_rf_dev", "")}}


def _register_fake(name: str) -> None:
    GlobalDeviceManager.register_device(name, FakeDevice())


def _drop_fake(name: str) -> None:
    GlobalDeviceManager.drop_device(name, ignore_error=True)


# --- A1: registry reachability + fresh cfg ---------------------------------


def test_jpa_power_registered_listable_and_creatable() -> None:
    assert "jpa/power" in ADAPTERS
    assert ADAPTERS["jpa/power"] is JpaPowerAdapter

    registry = Registry()
    register_all(registry)
    assert "jpa/power" in registry.list_names()
    adapter = registry.create("jpa/power")
    assert isinstance(adapter, JpaPowerAdapter)
    assert isinstance(adapter, ExpAdapterProtocol)


def test_fresh_cfg_is_valid_and_carries_device_selector_and_sweep() -> None:
    ml = _make_ml()
    ctx = _make_ctx(ml, best_jpa_power=-15.0)
    schema = JpaPowerAdapter().make_default_cfg(ctx)  # validates internally

    raw = schema_to_raw_dict(schema, ctx.md, ml)
    dev_raw = raw["dev"]
    assert isinstance(dev_raw, dict)
    assert dev_raw == {JPA_RF_ROLE_KEY: ""}  # empty selection: valid cfg
    sweep_raw = raw["sweep"]
    assert isinstance(sweep_raw, dict)
    sweep = sweep_raw["jpa_power"]
    from zcu_tools.program.v2 import SweepCfg

    assert isinstance(sweep, SweepCfg)

    # EvalValue edges are kept live in the schema.
    sweep_section = schema.value.fields["sweep"]
    assert isinstance(sweep_section, CfgSectionValue)
    gui_sweep = sweep_section.fields["jpa_power"]
    assert isinstance(gui_sweep, SweepValue)
    start = gui_sweep.start
    assert isinstance(start, EvalValue)
    assert start.expr == "best_jpa_power - 5.0"
    stop = gui_sweep.stop
    assert isinstance(stop, EvalValue)
    assert stop.expr == "best_jpa_power + 5.0"


def test_fresh_sweep_prefers_existing_best_jpa_power() -> None:
    ctx = _make_ctx(_make_ml(), best_jpa_power=-12.0)
    schema = JpaPowerAdapter().make_default_cfg(ctx)
    sweep_section = schema.value.fields["sweep"]
    assert isinstance(sweep_section, CfgSectionValue)
    gui_sweep = sweep_section.fields["jpa_power"]
    assert isinstance(gui_sweep, SweepValue)
    start = gui_sweep.start
    assert isinstance(start, EvalValue)
    assert start.expr == "best_jpa_power - 5.0"
    stop = gui_sweep.stop
    assert isinstance(stop, EvalValue)
    assert stop.expr == "best_jpa_power + 5.0"


def test_fresh_sweep_falls_back_to_conservative_low_power_seed_without_md() -> None:
    ctx = _make_ctx(_make_ml())
    schema = JpaPowerAdapter().make_default_cfg(ctx)
    sweep_section = schema.value.fields["sweep"]
    assert isinstance(sweep_section, CfgSectionValue)
    gui_sweep = sweep_section.fields["jpa_power"]
    assert isinstance(gui_sweep, SweepValue)
    # Conservative low-power notebook-derived survey seed (-20..-5 dBm).
    assert gui_sweep.start == -20.0
    assert gui_sweep.stop == -5.0
    assert gui_sweep.expts == 101


# --- A2: device preflight --------------------------------------------------


def test_lower_jpa_rf_power_dev_requires_selection() -> None:
    with pytest.raises(ValueError, match="missing JPA RF device selection"):
        lower_jpa_rf_power_dev({}, {})
    with pytest.raises(ValueError, match="missing JPA RF device selection"):
        lower_jpa_rf_power_dev({"dev": {JPA_RF_ROLE_KEY: ""}}, {})
    with pytest.raises(ValueError, match="missing JPA RF device selection"):
        lower_jpa_rf_power_dev({"dev": {}}, {})


def test_lower_jpa_rf_power_dev_requires_known_device() -> None:
    with pytest.raises(ValueError, match="not found in the device snapshot"):
        lower_jpa_rf_power_dev(_raw_with_selection(jpa_rf_dev="ghost"), {"sgs": _SGS})


def test_lower_jpa_rf_power_dev_fast_fails_unsupported_power_knob() -> None:
    with pytest.raises(ValueError, match="does not support the power knob"):
        lower_jpa_rf_power_dev(_raw_with_selection(jpa_rf_dev="dev"), {"dev": _YOKO})
    with pytest.raises(ValueError, match="YOKOGS200Info"):
        lower_jpa_rf_power_dev(_raw_with_selection(jpa_rf_dev="dev"), {"dev": _YOKO})


def test_lower_jpa_rf_power_dev_accepts_power_capable_rf_devices() -> None:
    patch = lower_jpa_rf_power_dev(
        _raw_with_selection(jpa_rf_dev="sgs"),
        {"sgs": _SGS, "anritsu": _ANRITSU},
    )
    assert patch == {"sgs": {"label": JPA_RF_LABEL}}
    patch2 = lower_jpa_rf_power_dev(
        _raw_with_selection(jpa_rf_dev="anritsu"),
        {"sgs": _SGS, "anritsu": _ANRITSU},
    )
    assert patch2 == {"anritsu": {"label": JPA_RF_LABEL}}


def test_lower_jpa_rf_power_dev_produces_exactly_one_labeled_patch() -> None:
    patch = lower_jpa_rf_power_dev(
        _raw_with_selection(jpa_rf_dev="sgs"),
        {"sgs": _SGS, "sgs2": _SGS2},
    )
    # Exactly one selected device, labeled jpa_rf_dev — the assembler patch.
    assert patch == {"sgs": {"label": JPA_RF_LABEL}}
    assert len(patch) == 1
    assert set(patch["sgs"]) == {"label"}
    assert patch["sgs"]["label"] == JPA_RF_LABEL


def test_validate_run_request_preflights_selection(monkeypatch) -> None:
    snapshot = {"sgs": _SGS, "yoko": _YOKO}
    monkeypatch.setattr(
        "zcu_tools.experiment.v2_gui.adapters.jpa.power.cached_device_snapshot",
        lambda: snapshot,
    )
    adapter = JpaPowerAdapter()
    req = _make_req()

    adapter.validate_run_request(req, _raw_with_selection(jpa_rf_dev="sgs"))

    with pytest.raises(ValueError, match="missing JPA RF device selection"):
        adapter.validate_run_request(req, _raw_with_selection())
    with pytest.raises(ValueError, match="not found"):
        adapter.validate_run_request(req, _raw_with_selection(jpa_rf_dev="ghost"))
    with pytest.raises(ValueError, match="power knob"):
        adapter.validate_run_request(req, _raw_with_selection(jpa_rf_dev="yoko"))


def test_preflight_refusals_and_pass_happen_without_hardware_queries(
    monkeypatch,
) -> None:
    """The production preflight path never queries or commands a live device.

    The registry membership is patched to return probe devices whose
    ``get_info`` would raise — so any hardware query in preflight would fail
    the test. All three applicable refusals (and the pass) must still complete
    before any hardware work.
    """

    queried: list[str] = []

    def _probe_device(info_model: type) -> Any:
        class _Probe:
            info_model: Any  # assigned below (class body cannot see the closure)
            address = "probe"

            def get_info(self) -> Any:  # pragma: no cover - must never run
                queried.append("get_info")
                raise AssertionError("preflight must not query hardware")

        _Probe.info_model = info_model
        return _Probe()

    def _devices(**named: type) -> dict[str, Any]:
        return {name: _probe_device(info_model) for name, info_model in named.items()}

    monkeypatch.setattr(
        "zcu_tools.device.GlobalDeviceManager.get_all_devices",
        lambda: _devices(sgs=RohdeSchwarzSGS100AInfo, yoko=YOKOGS200Info),
    )
    adapter = JpaPowerAdapter()
    req = _make_req()

    adapter.validate_run_request(req, _raw_with_selection(jpa_rf_dev="sgs"))

    with pytest.raises(ValueError, match="missing JPA RF device selection"):
        adapter.validate_run_request(req, _raw_with_selection())
    with pytest.raises(ValueError, match="not found"):
        adapter.validate_run_request(req, _raw_with_selection(jpa_rf_dev="ghost"))
    with pytest.raises(ValueError, match="power knob"):
        adapter.validate_run_request(req, _raw_with_selection(jpa_rf_dev="yoko"))

    assert queried == []  # preflight completed without a single hardware query


# --- A3: run lowering + canonical persistence ------------------------------


def test_run_lowers_dev_patch_and_returns_typed_result(monkeypatch) -> None:
    _register_fake("sgs")
    try:
        ml = _make_ml()
        adapter = JpaPowerAdapter()
        ctx = _make_ctx(ml)
        req = _make_req(ml, soc=MagicMock(), soccfg=MagicMock())
        schema = adapter.make_default_cfg(ctx)
        dev_section = schema.value.fields["dev"]
        assert isinstance(dev_section, CfgSectionValue)
        dev_section.fields[JPA_RF_ROLE_KEY] = DirectValue("sgs")

        captured: dict[str, Any] = {}

        def _fake_run(soc: Any, soccfg: Any, cfg: PowerCfg) -> PowerResult:
            captured["cfg"] = cfg
            return PowerResult(
                powers=np.array([1.0, 2.0]), signals=np.array([1.0, 2.0])
            )

        monkeypatch.setattr(PowerExp, "run", staticmethod(_fake_run))
        result = adapter.run(req, schema)

        assert isinstance(result, PowerResult)
        cfg = captured["cfg"]
        assert isinstance(cfg, PowerCfg)
        assert cfg.dev is not None
        # Production lowering: exactly one selected device labeled jpa_rf_dev.
        assert len(cfg.dev) == 1
        assert cfg.dev["sgs"].label == JPA_RF_LABEL
    finally:
        _drop_fake("sgs")


def test_canonical_save_load_roundtrip(tmp_path) -> None:
    _register_fake("sgs")
    try:
        ml = _make_ml()
        adapter = JpaPowerAdapter()
        ctx = _make_ctx(ml)
        req = _make_req(ml)
        schema = adapter.make_default_cfg(ctx)
        raw = schema_to_raw_dict(schema, ctx.md, ml)
        raw["dev"] = {JPA_RF_ROLE_KEY: "sgs"}
        cfg = adapter.build_exp_cfg(raw, req)
        assert isinstance(cfg, PowerCfg)

        result = PowerResult(
            powers=np.array([-20.0, -15.0, -10.0]),
            signals=np.array([1.0, 3.0, 2.0]),
            cfg_snapshot=cfg,
        )
        path = str(tmp_path / "jpa_power.hdf5")
        adapter.save(
            SaveDataRequest(
                data_path=path,
                run_result=result,
                md=MetaDict(),
                ml=ml,
                chip_name="chip",
                qub_name="Q1",
                res_name="R1",
                active_label="1",
            )
        )

        loaded = adapter.load(LoadDataRequest(data_path=path, md=MetaDict(), ml=ml))
        assert isinstance(loaded, PowerResult)
        np.testing.assert_allclose(loaded.powers, result.powers)
        np.testing.assert_allclose(loaded.signals, result.signals)
        assert loaded.cfg_snapshot is not None
        assert isinstance(loaded.cfg_snapshot, PowerCfg)
    finally:
        _drop_fake("sgs")


# --- A4/A5: analysis + writeback -------------------------------------------


def test_analyze_projects_best_power_and_figure() -> None:
    result = PowerResult(
        powers=np.array([-20.0, -15.0, -10.0]),
        signals=np.array([1.0, 3.0, 2.0]),
    )
    adapter = JpaPowerAdapter()
    analyze_result = adapter.analyze(
        AnalyzeRequest(
            run_result=result,
            analyze_params=NoAnalyzeParams(),
            md=MetaDict(),
            ml=ModuleLibrary(),
            predictor=None,
        )
    )
    assert isinstance(analyze_result, JpaPowerAnalyzeResult)
    assert analyze_result.best_power == -15.0
    assert isinstance(analyze_result.figure, Figure)


def test_analyze_figure_xaxis_is_pump_power_dbm() -> None:
    """The GUI review figure relabels the core figure at the adapter boundary:
    the x-axis names the swept quantity (JPA pump power, dBm) instead of the
    core figure's wrong 'JPA Frequency (MHz)'.
    """

    result = PowerResult(
        powers=np.array([-20.0, -15.0, -10.0]),
        signals=np.array([1.0, 3.0, 2.0]),
    )
    adapter = JpaPowerAdapter()
    analyze_result = adapter.analyze(
        AnalyzeRequest(
            run_result=result,
            analyze_params=NoAnalyzeParams(),
            md=MetaDict(),
            ml=ModuleLibrary(),
            predictor=None,
        )
    )
    ax = analyze_result.figure.axes[0]
    assert ax.get_xlabel() == "JPA pump power (dBm)"


def test_writeback_proposes_only_best_jpa_power() -> None:
    adapter = JpaPowerAdapter()
    items = list(
        adapter.get_writeback_items(
            WritebackRequest(
                run_result=MagicMock(),
                analyze_result=JpaPowerAnalyzeResult(
                    best_power=-15.0, figure=MagicMock()
                ),
                ctx=_make_ctx(),
            )
        )
    )
    assert len(items) == 1
    item = items[0]
    assert isinstance(item, MetaDictWriteback)
    assert item.target_name == "best_jpa_power"
    assert item.proposed_value == -15.0
    # A draft only: never applied to hardware, never named cur_jpa_A.
    assert item.target_name != "cur_jpa_A"


# --- A6: guide wording -----------------------------------------------------


def test_guide_warns_to_review_device_and_sweep() -> None:
    guide = JpaPowerAdapter.guide()
    text = f"{guide.behavior} {guide.recommended}".lower()
    assert "review" in text
    assert "device" in text
    assert "sweep" in text


def test_cfg_definition_type() -> None:
    assert isinstance(JpaPowerAdapter.cfg_definition(), MeasureCfgDefinition)
