"""jpa/flux adapter — registry reachability, device preflight, run/persistence,
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
from zcu_tools.experiment.v2.jpa import FluxCfg, FluxExp
from zcu_tools.experiment.v2.jpa.jpa_flux import FluxResult
from zcu_tools.experiment.v2_gui.adapters._support import MeasureCfgDefinition
from zcu_tools.experiment.v2_gui.adapters.jpa import (
    JpaFluxAdapter,
    JpaFluxAnalyzeResult,
)
from zcu_tools.experiment.v2_gui.adapters.jpa._shared import (
    JPA_FLUX_LABEL,
    JPA_FLUX_ROLE_KEY,
    lower_jpa_flux_dev,
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

_YOKO = YOKOGS200Info(address="yoko")
_YOKO2 = YOKOGS200Info(address="yoko2")
_SGS = RohdeSchwarzSGS100AInfo(address="sgs")
_ANRITSU = AnritsuMG3692Info(address="anritsu")


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
    return {"dev": {"jpa_flux_dev": dev_selection.get("jpa_flux_dev", "")}}


def _register_fake(name: str) -> None:
    GlobalDeviceManager.register_device(name, FakeDevice())


def _drop_fake(name: str) -> None:
    GlobalDeviceManager.drop_device(name, ignore_error=True)


# --- A1: registry reachability + fresh cfg ---------------------------------


def test_jpa_flux_registered_listable_and_creatable() -> None:
    assert "jpa/flux" in ADAPTERS
    assert ADAPTERS["jpa/flux"] is JpaFluxAdapter

    registry = Registry()
    register_all(registry)
    assert "jpa/flux" in registry.list_names()
    adapter = registry.create("jpa/flux")
    assert isinstance(adapter, JpaFluxAdapter)
    assert isinstance(adapter, ExpAdapterProtocol)


def test_fresh_cfg_is_valid_and_carries_device_selector_and_sweep() -> None:
    ml = _make_ml()
    ctx = _make_ctx(ml, best_jpa_flux=2.0e-3)
    schema = JpaFluxAdapter().make_default_cfg(ctx)  # validates internally

    raw = schema_to_raw_dict(schema, ctx.md, ml)
    dev_raw = raw["dev"]
    assert isinstance(dev_raw, dict)
    assert dev_raw == {JPA_FLUX_ROLE_KEY: ""}  # empty selection: valid cfg
    sweep_raw = raw["sweep"]
    assert isinstance(sweep_raw, dict)
    sweep = sweep_raw["jpa_flux"]
    from zcu_tools.program.v2 import SweepCfg

    assert isinstance(sweep, SweepCfg)

    # EvalValue edges are kept live in the schema.
    sweep_section = schema.value.fields["sweep"]
    assert isinstance(sweep_section, CfgSectionValue)
    gui_sweep = sweep_section.fields["jpa_flux"]
    assert isinstance(gui_sweep, SweepValue)
    start = gui_sweep.start
    assert isinstance(start, EvalValue)
    assert start.expr == "best_jpa_flux - 0.005"
    stop = gui_sweep.stop
    assert isinstance(stop, EvalValue)
    assert stop.expr == "best_jpa_flux + 0.005"


def test_fresh_sweep_prefers_existing_best_jpa_flux() -> None:
    ctx = _make_ctx(_make_ml(), best_jpa_flux=-1.0e-3)
    schema = JpaFluxAdapter().make_default_cfg(ctx)
    sweep_section = schema.value.fields["sweep"]
    assert isinstance(sweep_section, CfgSectionValue)
    gui_sweep = sweep_section.fields["jpa_flux"]
    assert isinstance(gui_sweep, SweepValue)
    start = gui_sweep.start
    assert isinstance(start, EvalValue)
    assert start.expr == "best_jpa_flux - 0.005"
    stop = gui_sweep.stop
    assert isinstance(stop, EvalValue)
    assert stop.expr == "best_jpa_flux + 0.005"


def test_fresh_sweep_falls_back_to_literal_seed_without_md() -> None:
    ctx = _make_ctx(_make_ml())
    schema = JpaFluxAdapter().make_default_cfg(ctx)
    sweep_section = schema.value.fields["sweep"]
    assert isinstance(sweep_section, CfgSectionValue)
    gui_sweep = sweep_section.fields["jpa_flux"]
    assert isinstance(gui_sweep, SweepValue)
    assert gui_sweep.start == -0.005
    assert gui_sweep.stop == 0.005
    assert gui_sweep.expts == 101


# --- A2: device preflight --------------------------------------------------


def test_lower_jpa_flux_dev_requires_selection() -> None:
    with pytest.raises(ValueError, match="missing JPA flux device selection"):
        lower_jpa_flux_dev({}, {})
    with pytest.raises(ValueError, match="missing JPA flux device selection"):
        lower_jpa_flux_dev({"dev": {JPA_FLUX_ROLE_KEY: ""}}, {})
    with pytest.raises(ValueError, match="missing JPA flux device selection"):
        lower_jpa_flux_dev({"dev": {}}, {})


def test_lower_jpa_flux_dev_requires_known_device() -> None:
    with pytest.raises(ValueError, match="not found in the device snapshot"):
        lower_jpa_flux_dev(_raw_with_selection(jpa_flux_dev="ghost"), {"yoko": _YOKO})


@pytest.mark.parametrize(
    ("info", "device_type"),
    [
        (_SGS, "RohdeSchwarzSGS100AInfo"),
        (_ANRITSU, "AnritsuMG3692Info"),
    ],
)
def test_lower_jpa_flux_dev_fast_fails_unsupported_flux_knob(
    info: Any, device_type: str
) -> None:
    with pytest.raises(ValueError, match="does not support the flux knob"):
        lower_jpa_flux_dev(_raw_with_selection(jpa_flux_dev="dev"), {"dev": info})
    with pytest.raises(ValueError, match=device_type):
        lower_jpa_flux_dev(_raw_with_selection(jpa_flux_dev="dev"), {"dev": info})


def test_lower_jpa_flux_dev_produces_exactly_one_labeled_patch() -> None:
    patch = lower_jpa_flux_dev(
        _raw_with_selection(jpa_flux_dev="yoko"),
        {"yoko": _YOKO, "yoko2": _YOKO2},
    )
    # Exactly one selected device, labeled jpa_flux_dev — the assembler patch.
    assert patch == {"yoko": {"label": JPA_FLUX_LABEL}}
    assert len(patch) == 1
    assert set(patch["yoko"]) == {"label"}
    assert patch["yoko"]["label"] == JPA_FLUX_LABEL


def test_validate_run_request_preflights_selection(monkeypatch) -> None:
    snapshot = {"yoko": _YOKO, "sgs": _SGS}
    monkeypatch.setattr(
        "zcu_tools.experiment.v2_gui.adapters.jpa.flux.cached_device_snapshot",
        lambda: snapshot,
    )
    adapter = JpaFluxAdapter()
    req = _make_req()

    adapter.validate_run_request(req, _raw_with_selection(jpa_flux_dev="yoko"))

    with pytest.raises(ValueError, match="missing JPA flux device selection"):
        adapter.validate_run_request(req, _raw_with_selection())
    with pytest.raises(ValueError, match="not found"):
        adapter.validate_run_request(req, _raw_with_selection(jpa_flux_dev="ghost"))
    with pytest.raises(ValueError, match="flux knob"):
        adapter.validate_run_request(req, _raw_with_selection(jpa_flux_dev="sgs"))


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
        lambda: _devices(yoko=YOKOGS200Info, sgs=RohdeSchwarzSGS100AInfo),
    )
    adapter = JpaFluxAdapter()
    req = _make_req()

    adapter.validate_run_request(req, _raw_with_selection(jpa_flux_dev="yoko"))

    with pytest.raises(ValueError, match="missing JPA flux device selection"):
        adapter.validate_run_request(req, _raw_with_selection())
    with pytest.raises(ValueError, match="not found"):
        adapter.validate_run_request(req, _raw_with_selection(jpa_flux_dev="ghost"))
    with pytest.raises(ValueError, match="flux knob"):
        adapter.validate_run_request(req, _raw_with_selection(jpa_flux_dev="sgs"))

    assert queried == []  # preflight completed without a single hardware query


# --- A3: run lowering + canonical persistence ------------------------------


def test_run_lowers_dev_patch_and_returns_typed_result(monkeypatch) -> None:
    _register_fake("yoko")
    try:
        ml = _make_ml()
        adapter = JpaFluxAdapter()
        ctx = _make_ctx(ml)
        req = _make_req(ml, soc=MagicMock(), soccfg=MagicMock())
        schema = adapter.make_default_cfg(ctx)
        dev_section = schema.value.fields["dev"]
        assert isinstance(dev_section, CfgSectionValue)
        dev_section.fields[JPA_FLUX_ROLE_KEY] = DirectValue("yoko")

        captured: dict[str, Any] = {}

        def _fake_run(soc: Any, soccfg: Any, cfg: FluxCfg) -> FluxResult:
            captured["cfg"] = cfg
            return FluxResult(fluxes=np.array([1.0, 2.0]), signals=np.array([1.0, 2.0]))

        monkeypatch.setattr(FluxExp, "run", staticmethod(_fake_run))
        result = adapter.run(req, schema)

        assert isinstance(result, FluxResult)
        cfg = captured["cfg"]
        assert isinstance(cfg, FluxCfg)
        assert cfg.dev is not None
        # Production lowering: exactly one selected device labeled jpa_flux_dev.
        assert len(cfg.dev) == 1
        assert cfg.dev["yoko"].label == JPA_FLUX_LABEL
    finally:
        _drop_fake("yoko")


def test_canonical_save_load_roundtrip(tmp_path) -> None:
    _register_fake("yoko")
    try:
        ml = _make_ml()
        adapter = JpaFluxAdapter()
        ctx = _make_ctx(ml)
        req = _make_req(ml)
        schema = adapter.make_default_cfg(ctx)
        raw = schema_to_raw_dict(schema, ctx.md, ml)
        raw["dev"] = {JPA_FLUX_ROLE_KEY: "yoko"}
        cfg = adapter.build_exp_cfg(raw, req)
        assert isinstance(cfg, FluxCfg)

        result = FluxResult(
            fluxes=np.array([-0.005, 0.0, 0.005]),
            signals=np.array([1.0, 3.0, 2.0]),
            cfg_snapshot=cfg,
        )
        path = str(tmp_path / "jpa_flux.hdf5")
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
        assert isinstance(loaded, FluxResult)
        np.testing.assert_allclose(loaded.fluxes, result.fluxes)
        np.testing.assert_allclose(loaded.signals, result.signals)
        assert loaded.cfg_snapshot is not None
        assert isinstance(loaded.cfg_snapshot, FluxCfg)
    finally:
        _drop_fake("yoko")


# --- A4/A5: analysis + writeback -------------------------------------------


def test_analyze_projects_best_flux_and_figure() -> None:
    result = FluxResult(
        fluxes=np.array([-0.005, 0.0, 0.005]),
        signals=np.array([1.0, 5.0, 2.0]),
    )
    adapter = JpaFluxAdapter()
    analyze_result = adapter.analyze(
        AnalyzeRequest(
            run_result=result,
            analyze_params=NoAnalyzeParams(),
            md=MetaDict(),
            ml=ModuleLibrary(),
            predictor=None,
        )
    )
    assert isinstance(analyze_result, JpaFluxAnalyzeResult)
    assert analyze_result.best_flux == 0.0
    assert isinstance(analyze_result.figure, Figure)


def test_analyze_figure_uses_neutral_flux_device_value_labels() -> None:
    """The GUI review figure relabels the core figure at the adapter boundary:
    the flux axis and the optimum legend use the neutral 'JPA flux device
    value' vocabulary (A6) with no physical-unit (a.u.) claim.
    """

    result = FluxResult(
        fluxes=np.array([-0.005, 0.0, 0.005]),
        signals=np.array([1.0, 5.0, 2.0]),
    )
    adapter = JpaFluxAdapter()
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
    assert ax.get_xlabel() == "JPA flux device value"
    assert "a.u." not in ax.get_xlabel()
    legend = ax.get_legend()
    assert legend is not None
    legend_texts = [text.get_text() for text in legend.get_texts()]
    assert any(
        text.startswith("best JPA flux device value = ") for text in legend_texts
    )
    assert not any("a.u." in text for text in legend_texts)


def test_writeback_proposes_only_best_jpa_flux() -> None:
    adapter = JpaFluxAdapter()
    items = list(
        adapter.get_writeback_items(
            WritebackRequest(
                run_result=MagicMock(),
                analyze_result=JpaFluxAnalyzeResult(
                    best_flux=2.0e-3, figure=MagicMock()
                ),
                ctx=_make_ctx(),
            )
        )
    )
    assert len(items) == 1
    item = items[0]
    assert isinstance(item, MetaDictWriteback)
    assert item.target_name == "best_jpa_flux"
    assert item.proposed_value == 2.0e-3
    # A draft only: never applied to hardware, never named cur_jpa_A.
    assert item.target_name != "cur_jpa_A"


# --- A6: neutral flux device value wording ---------------------------------


def test_sweep_label_and_guide_use_neutral_flux_device_value_wording() -> None:
    # The GUI sweep label is fixed to the neutral 'JPA flux device value'
    # vocabulary — no physical-unit migration claim.
    guide = JpaFluxAdapter.guide()
    text = f"{guide.behavior} {guide.typical_writeback}".lower()
    assert "flux device value" in text
    for unit_claim in ("a.u.", "ampere", " ma", "current (a)"):
        assert unit_claim not in text

    from zcu_tools.gui.cfg import CfgSectionSpec, SweepSpec

    schema = JpaFluxAdapter().make_default_cfg(_make_ctx())
    sweep_spec_section = schema.spec.fields["sweep"]
    assert isinstance(sweep_spec_section, CfgSectionSpec)
    sweep_spec = sweep_spec_section.fields["jpa_flux"]
    assert isinstance(sweep_spec, SweepSpec)
    assert sweep_spec.label == "JPA flux device value"


def test_guide_warns_to_review_device_and_sweep() -> None:
    guide = JpaFluxAdapter.guide()
    text = f"{guide.behavior} {guide.recommended}".lower()
    assert "review" in text
    assert "device" in text
    assert "sweep" in text


def test_cfg_definition_type() -> None:
    assert isinstance(JpaFluxAdapter.cfg_definition(), MeasureCfgDefinition)
