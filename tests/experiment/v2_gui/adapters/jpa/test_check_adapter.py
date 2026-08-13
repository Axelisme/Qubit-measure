"""jpa/check adapter — registry reachability, RF device preflight, off/on run
delegation, figure-only analysis (no scalar/writeback), canonical persistence
and operator guide."""

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
from zcu_tools.experiment.v2.jpa import CheckCfg, CheckExp
from zcu_tools.experiment.v2.jpa.jpa_check import CheckResult
from zcu_tools.experiment.v2_gui.adapters._support import MeasureCfgDefinition
from zcu_tools.experiment.v2_gui.adapters.base import BaseAdapter
from zcu_tools.experiment.v2_gui.adapters.jpa import (
    JpaCheckAdapter,
    JpaCheckAnalyzeResult,
)
from zcu_tools.experiment.v2_gui.adapters.jpa._shared import (
    JPA_RF_LABEL,
    JPA_RF_ROLE_KEY,
    lower_jpa_rf_output_dev,
)
from zcu_tools.experiment.v2_gui.registry import ADAPTERS, register_all
from zcu_tools.gui.app.main.adapter import (
    AnalysisMode,
    AnalyzeRequest,
    ExpAdapterProtocol,
    LoadDataRequest,
    NoAnalyzeParams,
    RunRequest,
    SaveDataRequest,
    WritebackRequest,
)
from zcu_tools.gui.app.main.adapter.lowering import schema_to_raw_dict
from zcu_tools.gui.app.main.registry import Registry
from zcu_tools.gui.cfg import (
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    SweepValue,
)
from zcu_tools.meta_tool import MetaDict
from zcu_tools.utils.datasaver import load_labber_data

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
    return {"dev": {"jpa_rf_dev": dev_selection.get("jpa_rf_dev", "")}}


def _register_fake(name: str) -> None:
    GlobalDeviceManager.register_device(name, FakeDevice())


def _drop_fake(name: str) -> None:
    GlobalDeviceManager.drop_device(name, ignore_error=True)


def _sweep_section(schema: Any) -> CfgSectionValue:
    section = schema.value.fields["sweep"]
    assert isinstance(section, CfgSectionValue)
    return section


# --- A1: registry reachability + fresh cfg ---------------------------------


def test_jpa_check_registered_listable_and_creatable() -> None:
    assert "jpa/check" in ADAPTERS
    assert ADAPTERS["jpa/check"] is JpaCheckAdapter

    registry = Registry()
    register_all(registry)
    assert "jpa/check" in registry.list_names()
    adapter = registry.create("jpa/check")
    assert isinstance(adapter, JpaCheckAdapter)
    assert isinstance(adapter, ExpAdapterProtocol)


def test_fresh_cfg_is_valid_and_carries_rf_selector_and_freq_sweep() -> None:
    ml = _make_ml()
    ctx = _make_ctx(ml, r_f=6500.0, rf_w=10.0)
    schema = JpaCheckAdapter().make_default_cfg(ctx)  # validates internally

    raw = schema_to_raw_dict(schema, ctx.md, ml)
    dev_raw = raw["dev"]
    assert isinstance(dev_raw, dict)
    assert dev_raw == {JPA_RF_ROLE_KEY: ""}  # empty selection: valid cfg
    sweep_raw = raw["sweep"]
    assert isinstance(sweep_raw, dict)
    from zcu_tools.program.v2 import SweepCfg

    assert isinstance(sweep_raw["freq"], SweepCfg)

    # The freq sweep seed stays live in the schema.
    freq_sweep = _sweep_section(schema).fields["freq"]
    assert isinstance(freq_sweep, SweepValue)
    start = freq_sweep.start
    assert isinstance(start, EvalValue)
    assert start.expr == "r_f - 1.5 * rf_w"
    stop = freq_sweep.stop
    assert isinstance(stop, EvalValue)
    assert stop.expr == "r_f + 1.5 * rf_w"
    assert freq_sweep.expts == 101


def test_fresh_freq_sweep_falls_back_to_literal_seed_without_md() -> None:
    ctx = _make_ctx(_make_ml())
    schema = JpaCheckAdapter().make_default_cfg(ctx)
    gui_sweep = _sweep_section(schema).fields["freq"]
    assert isinstance(gui_sweep, SweepValue)
    # proper_res_freq_range fallback: 6500 ± 1.5*500 MHz.
    assert gui_sweep.start == 6500.0 - 1.5 * 500.0
    assert gui_sweep.stop == 6500.0 + 1.5 * 500.0
    assert gui_sweep.expts == 101


def test_cfg_defines_only_the_rf_device_selector() -> None:
    """The core commands the RF device's output — the cfg must carry exactly
    that one selector."""

    schema = JpaCheckAdapter().make_default_cfg(_make_ctx())
    dev_section = schema.value.fields["dev"]
    assert isinstance(dev_section, CfgSectionValue)
    assert set(dev_section.fields) == {JPA_RF_ROLE_KEY}

    dev_spec_section = schema.spec.fields["dev"]
    assert isinstance(dev_spec_section, CfgSectionSpec)
    assert set(dev_spec_section.fields) == {JPA_RF_ROLE_KEY}


# --- A2: device preflight --------------------------------------------------


def test_lower_jpa_rf_output_dev_requires_selection() -> None:
    with pytest.raises(ValueError, match="missing JPA RF device selection"):
        lower_jpa_rf_output_dev({}, {})
    with pytest.raises(ValueError, match="missing JPA RF device selection"):
        lower_jpa_rf_output_dev({"dev": {JPA_RF_ROLE_KEY: ""}}, {})
    with pytest.raises(ValueError, match="missing JPA RF device selection"):
        lower_jpa_rf_output_dev({"dev": {}}, {})


def test_lower_jpa_rf_output_dev_requires_known_device() -> None:
    with pytest.raises(ValueError, match="not found in the device snapshot"):
        lower_jpa_rf_output_dev(_raw_with_selection(jpa_rf_dev="ghost"), {"sgs": _SGS})


def test_lower_jpa_rf_output_dev_fast_fails_unsupported_output_knob() -> None:
    # YOKOGS200Info exposes only the flux knob — no output knob.
    with pytest.raises(ValueError, match="does not support the output knob"):
        lower_jpa_rf_output_dev(_raw_with_selection(jpa_rf_dev="dev"), {"dev": _YOKO})
    with pytest.raises(ValueError, match="YOKOGS200Info"):
        lower_jpa_rf_output_dev(_raw_with_selection(jpa_rf_dev="dev"), {"dev": _YOKO})


def test_lower_jpa_rf_output_dev_produces_exactly_one_labeled_patch() -> None:
    patch = lower_jpa_rf_output_dev(
        _raw_with_selection(jpa_rf_dev="sgs"),
        {"sgs": _SGS, "anritsu": _ANRITSU},
    )
    # Exactly one selected device, labeled jpa_rf_dev — the assembler patch.
    assert patch == {"sgs": {"label": JPA_RF_LABEL}}
    assert len(patch) == 1
    assert set(patch["sgs"]) == {"label"}
    assert patch["sgs"]["label"] == JPA_RF_LABEL


def test_validate_run_request_preflights_selection(monkeypatch) -> None:
    snapshot = {"sgs": _SGS, "yoko": _YOKO}
    monkeypatch.setattr(
        "zcu_tools.experiment.v2_gui.adapters.jpa.check.cached_device_snapshot",
        lambda: snapshot,
    )
    adapter = JpaCheckAdapter()
    req = _make_req()

    adapter.validate_run_request(req, _raw_with_selection(jpa_rf_dev="sgs"))

    with pytest.raises(ValueError, match="missing JPA RF device selection"):
        adapter.validate_run_request(req, _raw_with_selection())
    with pytest.raises(ValueError, match="not found"):
        adapter.validate_run_request(req, _raw_with_selection(jpa_rf_dev="ghost"))
    with pytest.raises(ValueError, match="output knob"):
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
    adapter = JpaCheckAdapter()
    req = _make_req()

    adapter.validate_run_request(req, _raw_with_selection(jpa_rf_dev="sgs"))

    with pytest.raises(ValueError, match="missing JPA RF device selection"):
        adapter.validate_run_request(req, _raw_with_selection())
    with pytest.raises(ValueError, match="not found"):
        adapter.validate_run_request(req, _raw_with_selection(jpa_rf_dev="ghost"))
    with pytest.raises(ValueError, match="output knob"):
        adapter.validate_run_request(req, _raw_with_selection(jpa_rf_dev="yoko"))

    assert queried == []  # preflight completed without a single hardware query


# --- A3: run delegation + core off/on contract -----------------------------


def test_core_run_contract_is_off_then_on_with_terminal_on() -> None:
    """The adapter delegates the core scan unchanged: outputs are scanned
    [0, 1] and the output map labels 0 -> off, 1 -> on, so the terminal pump
    state after the run is ON — the adapter adds no reset/close step."""

    assert CheckExp.OUTPUT_MAP == {0: "off", 1: "on"}


def test_run_delegates_core_with_lowered_rf_patch_and_no_extra_ops(
    monkeypatch,
) -> None:
    _register_fake("sgs")
    try:
        ml = _make_ml()
        adapter = JpaCheckAdapter()
        ctx = _make_ctx(ml)
        req = _make_req(ml, soc=MagicMock(), soccfg=MagicMock())
        schema = adapter.make_default_cfg(ctx)
        dev_section = schema.value.fields["dev"]
        assert isinstance(dev_section, CfgSectionValue)
        dev_section.fields[JPA_RF_ROLE_KEY] = DirectValue("sgs")

        captured: dict[str, Any] = {}

        def _fake_run(soc: Any, soccfg: Any, cfg: CheckCfg) -> CheckResult:
            captured["cfg"] = cfg
            return CheckResult(
                outputs=np.array([0, 1]),
                freqs=np.array([6490.0, 6500.0, 6510.0]),
                signals=np.zeros((2, 3), dtype=np.complex128),
            )

        monkeypatch.setattr(CheckExp, "run", staticmethod(_fake_run))
        result = adapter.run(req, schema)

        assert isinstance(result, CheckResult)
        cfg = captured["cfg"]
        assert isinstance(cfg, CheckCfg)
        assert cfg.dev is not None
        # Production lowering: exactly one selected device labeled jpa_rf_dev,
        # which the core's set_output_in_dev_cfg(label="jpa_rf_dev") drives.
        assert len(cfg.dev) == 1
        assert cfg.dev["sgs"].label == JPA_RF_LABEL
        # Pure delegation: the adapter adds no post-run reset/close of the pump.
        assert type(adapter).run is BaseAdapter.run
    finally:
        _drop_fake("sgs")


# --- A4: figure-only analysis, no writeback --------------------------------


def test_capabilities_declare_fit_analysis() -> None:
    caps = JpaCheckAdapter.capabilities
    assert caps.analysis is AnalysisMode.FIT
    assert caps.requires_soc is True


def test_analyze_returns_figure_only_comparison_result(monkeypatch) -> None:
    fig = Figure()
    monkeypatch.setattr(CheckExp, "analyze", staticmethod(lambda result: fig))
    adapter = JpaCheckAdapter()
    req = AnalyzeRequest(
        run_result=CheckResult(
            outputs=np.array([0, 1]),
            freqs=np.array([6490.0, 6500.0, 6510.0]),
            signals=np.zeros((2, 3), dtype=np.complex128),
        ),
        analyze_params=NoAnalyzeParams(),
        md=MetaDict(),
        ml=_make_ml(),
        predictor=None,
    )
    result = adapter.analyze(req)
    assert isinstance(result, JpaCheckAnalyzeResult)
    assert result.figure is fig
    # Figure-only: exactly one field, no scalar, hence an empty summary dict.
    import dataclasses

    assert [f.name for f in dataclasses.fields(result)] == ["figure"]
    assert result.to_summary_dict() == {}


def test_no_writeback_items_are_produced() -> None:
    adapter = JpaCheckAdapter()
    items = list(
        adapter.get_writeback_items(
            WritebackRequest(
                run_result=MagicMock(),
                analyze_result=MagicMock(),
                ctx=_make_ctx(),
            )
        )
    )
    assert items == []


# --- A5: canonical persistence ---------------------------------------------


def test_canonical_save_load_roundtrip_restores_typed_result_and_both_signal_sets(
    tmp_path,
) -> None:
    """Both pump states (off row 0, on row 1) round-trip through the adapter's
    canonical save/load path as a typed CheckResult."""

    _register_fake("sgs")
    try:
        ml = _make_ml()
        adapter = JpaCheckAdapter()
        ctx = _make_ctx(ml)
        req = _make_req(ml)
        schema = adapter.make_default_cfg(ctx)
        raw = schema_to_raw_dict(schema, ctx.md, ml)
        raw["dev"] = {JPA_RF_ROLE_KEY: "sgs"}
        cfg = adapter.build_exp_cfg(raw, req)
        assert isinstance(cfg, CheckCfg)

        freqs = np.linspace(6490.0, 6510.0, 5, dtype=np.float64)
        outputs = np.array([0.0, 1.0], dtype=np.float64)
        real = np.arange(10, dtype=np.float64).reshape(2, 5)
        signals = (real + 1j * (real + 0.5)).astype(np.complex128)
        result = CheckResult(
            outputs=outputs,
            freqs=freqs,
            signals=signals,
            cfg_snapshot=cfg,
        )
        path = str(tmp_path / "jpa_check.hdf5")
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

        # On-disk axes are inner-first: inner frequency axis first, outer
        # output axis last; z keeps both signal sets (off row, on row).
        raw_data = load_labber_data(path)
        assert [axis.name for axis in raw_data.axes] == ["Frequency", "JPA Output"]
        assert raw_data.z.shape == (2, 5)

        loaded = adapter.load(LoadDataRequest(data_path=path, md=MetaDict(), ml=ml))
        assert isinstance(loaded, CheckResult)
        np.testing.assert_array_equal(loaded.outputs, outputs)
        np.testing.assert_array_equal(loaded.freqs, freqs)
        assert loaded.signals.shape == (2, 5)
        assert loaded.signals.dtype == np.complex128
        np.testing.assert_allclose(loaded.signals, signals, rtol=0, atol=0)
        assert loaded.cfg_snapshot is not None
        assert isinstance(loaded.cfg_snapshot, CheckCfg)
        # The snapshot round-trips the exact cfg built by the adapter.
        assert loaded.cfg_snapshot.sweep.freq.expts == cfg.sweep.freq.expts
    finally:
        _drop_fake("sgs")


# --- A6: guide discloses pump external effect + operator preflight ---------


def test_guide_warns_pump_ends_on_and_covers_operator_preflight() -> None:
    guide = JpaCheckAdapter.guide()
    text = f"{guide.behavior} {guide.recommended}".lower()
    assert "pump" in text
    assert "confirm" in text  # operator preflight before the run
    # The external effect is disclosed: the run leaves the pump output on.
    assert "leaves" in text and " on " in text
    assert "not turned off" in text


def test_guide_has_no_writeback_claim() -> None:
    guide = JpaCheckAdapter.guide()
    assert "no writeback" in guide.typical_writeback.lower()


# --- A7: checks ------------------------------------------------------------


def test_cfg_definition_type() -> None:
    assert isinstance(JpaCheckAdapter.cfg_definition(), MeasureCfgDefinition)
