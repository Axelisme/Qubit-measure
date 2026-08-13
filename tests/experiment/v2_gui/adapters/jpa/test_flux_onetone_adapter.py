"""jpa/flux_onetone adapter — registry reachability, device preflight,
run/persistence, no-analysis capability and operator guide."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from zcu_tools.device import (
    AnritsuMG3692Info,
    FakeDevice,
    GlobalDeviceManager,
    RohdeSchwarzSGS100AInfo,
    YOKOGS200Info,
)
from zcu_tools.experiment.v2.jpa import OneToneFluxCfg, OneToneFluxExp
from zcu_tools.experiment.v2.jpa.jpa_flux_onetone import OneToneFluxResult
from zcu_tools.experiment.v2_gui.adapters._support import MeasureCfgDefinition
from zcu_tools.experiment.v2_gui.adapters.jpa import JpaFluxOneToneAdapter
from zcu_tools.experiment.v2_gui.adapters.jpa._shared import (
    JPA_FLUX_LABEL,
    JPA_FLUX_ROLE_KEY,
    JPA_RF_ROLE_KEY,
    lower_jpa_flux_dev,
)
from zcu_tools.experiment.v2_gui.registry import ADAPTERS, register_all
from zcu_tools.gui.app.main.adapter import (
    AnalysisMode,
    ExpAdapterProtocol,
    LoadDataRequest,
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
    return {"dev": {"jpa_flux_dev": dev_selection.get("jpa_flux_dev", "")}}


def _register_fake(name: str) -> None:
    GlobalDeviceManager.register_device(name, FakeDevice())


def _drop_fake(name: str) -> None:
    GlobalDeviceManager.drop_device(name, ignore_error=True)


def _sweep_section(schema: Any) -> CfgSectionValue:
    section = schema.value.fields["sweep"]
    assert isinstance(section, CfgSectionValue)
    return section


# --- A1: registry reachability + fresh cfg ---------------------------------


def test_jpa_flux_onetone_registered_listable_and_creatable() -> None:
    assert "jpa/flux_onetone" in ADAPTERS
    assert ADAPTERS["jpa/flux_onetone"] is JpaFluxOneToneAdapter

    registry = Registry()
    register_all(registry)
    assert "jpa/flux_onetone" in registry.list_names()
    adapter = registry.create("jpa/flux_onetone")
    assert isinstance(adapter, JpaFluxOneToneAdapter)
    assert isinstance(adapter, ExpAdapterProtocol)


def test_fresh_cfg_is_valid_and_carries_flux_selector_and_both_sweeps() -> None:
    ml = _make_ml()
    ctx = _make_ctx(ml, best_jpa_flux=2.0e-3, r_f=6500.0, rf_w=10.0)
    schema = JpaFluxOneToneAdapter().make_default_cfg(ctx)  # validates internally

    raw = schema_to_raw_dict(schema, ctx.md, ml)
    dev_raw = raw["dev"]
    assert isinstance(dev_raw, dict)
    assert dev_raw == {JPA_FLUX_ROLE_KEY: ""}  # empty selection: valid cfg
    sweep_raw = raw["sweep"]
    assert isinstance(sweep_raw, dict)
    from zcu_tools.program.v2 import SweepCfg

    assert isinstance(sweep_raw["jpa_flux"], SweepCfg)
    assert isinstance(sweep_raw["freq"], SweepCfg)

    # Both sweep seeds stay live in the schema.
    sweep_section = _sweep_section(schema)
    flux_sweep = sweep_section.fields["jpa_flux"]
    assert isinstance(flux_sweep, SweepValue)
    start = flux_sweep.start
    assert isinstance(start, EvalValue)
    assert start.expr == "best_jpa_flux - 0.005"
    stop = flux_sweep.stop
    assert isinstance(stop, EvalValue)
    assert stop.expr == "best_jpa_flux + 0.005"

    freq_sweep = sweep_section.fields["freq"]
    assert isinstance(freq_sweep, SweepValue)
    freq_start = freq_sweep.start
    assert isinstance(freq_start, EvalValue)
    assert freq_start.expr == "r_f - 1.5 * rf_w"
    freq_stop = freq_sweep.stop
    assert isinstance(freq_stop, EvalValue)
    assert freq_stop.expr == "r_f + 1.5 * rf_w"
    assert freq_sweep.expts == 101


def test_fresh_flux_sweep_prefers_existing_best_jpa_flux() -> None:
    ctx = _make_ctx(_make_ml(), best_jpa_flux=-1.0e-3)
    schema = JpaFluxOneToneAdapter().make_default_cfg(ctx)
    gui_sweep = _sweep_section(schema).fields["jpa_flux"]
    assert isinstance(gui_sweep, SweepValue)
    start = gui_sweep.start
    assert isinstance(start, EvalValue)
    assert start.expr == "best_jpa_flux - 0.005"
    stop = gui_sweep.stop
    assert isinstance(stop, EvalValue)
    assert stop.expr == "best_jpa_flux + 0.005"


def test_fresh_flux_sweep_falls_back_to_literal_seed_without_md() -> None:
    ctx = _make_ctx(_make_ml())
    schema = JpaFluxOneToneAdapter().make_default_cfg(ctx)
    gui_sweep = _sweep_section(schema).fields["jpa_flux"]
    assert isinstance(gui_sweep, SweepValue)
    assert gui_sweep.start == -0.005
    assert gui_sweep.stop == 0.005
    assert gui_sweep.expts == 101


def test_fresh_freq_sweep_falls_back_to_literal_seed_without_md() -> None:
    ctx = _make_ctx(_make_ml())
    schema = JpaFluxOneToneAdapter().make_default_cfg(ctx)
    gui_sweep = _sweep_section(schema).fields["freq"]
    assert isinstance(gui_sweep, SweepValue)
    # proper_res_freq_range fallback: 6500 ± 1.5*500 MHz.
    assert gui_sweep.start == 6500.0 - 1.5 * 500.0
    assert gui_sweep.stop == 6500.0 + 1.5 * 500.0
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
        "zcu_tools.experiment.v2_gui.adapters.jpa.flux_onetone.cached_device_snapshot",
        lambda: snapshot,
    )
    adapter = JpaFluxOneToneAdapter()
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
    adapter = JpaFluxOneToneAdapter()
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
        adapter = JpaFluxOneToneAdapter()
        ctx = _make_ctx(ml)
        req = _make_req(ml, soc=MagicMock(), soccfg=MagicMock())
        schema = adapter.make_default_cfg(ctx)
        dev_section = schema.value.fields["dev"]
        assert isinstance(dev_section, CfgSectionValue)
        dev_section.fields[JPA_FLUX_ROLE_KEY] = DirectValue("yoko")

        captured: dict[str, Any] = {}

        def _fake_run(soc: Any, soccfg: Any, cfg: OneToneFluxCfg) -> OneToneFluxResult:
            captured["cfg"] = cfg
            return OneToneFluxResult(
                fluxes=np.array([-0.005, 0.0, 0.005]),
                freqs=np.array([6490.0, 6500.0, 6510.0]),
                signals=np.zeros((3, 3), dtype=np.complex128),
            )

        monkeypatch.setattr(OneToneFluxExp, "run", staticmethod(_fake_run))
        result = adapter.run(req, schema)

        assert isinstance(result, OneToneFluxResult)
        cfg = captured["cfg"]
        assert isinstance(cfg, OneToneFluxCfg)
        assert cfg.dev is not None
        # Production lowering: exactly one selected device labeled jpa_flux_dev.
        assert len(cfg.dev) == 1
        assert cfg.dev["yoko"].label == JPA_FLUX_LABEL
    finally:
        _drop_fake("yoko")


@pytest.mark.parametrize(("n_flux", "n_freq"), [(3, 5), (5, 3)])
def test_canonical_save_load_roundtrip_preserves_non_square_axes(
    tmp_path, n_flux: int, n_freq: int
) -> None:
    """Non-square flux × freq datasets round-trip with inner-freq / outer-flux
    axes semantics preserved (the adapter's canonical save/load path)."""

    _register_fake("yoko")
    try:
        ml = _make_ml()
        adapter = JpaFluxOneToneAdapter()
        ctx = _make_ctx(ml)
        req = _make_req(ml)
        schema = adapter.make_default_cfg(ctx)
        raw = schema_to_raw_dict(schema, ctx.md, ml)
        raw["dev"] = {JPA_FLUX_ROLE_KEY: "yoko"}
        cfg = adapter.build_exp_cfg(raw, req)
        assert isinstance(cfg, OneToneFluxCfg)

        fluxes = np.linspace(-0.005, 0.005, n_flux, dtype=np.float64)
        freqs = np.linspace(6490.0, 6510.0, n_freq, dtype=np.float64)
        real = np.arange(n_flux * n_freq, dtype=np.float64).reshape(n_flux, n_freq)
        signals = (real + 1j * (real + 0.5)).astype(np.complex128)
        result = OneToneFluxResult(
            fluxes=fluxes,
            freqs=freqs,
            signals=signals,
            cfg_snapshot=cfg,
        )
        path = str(tmp_path / "jpa_flux_onetone.hdf5")
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

        # On-disk axes are inner-first: inner frequency axis first, outer flux
        # axis last, so a non-square map keeps its shape and axis semantics.
        raw_data = load_labber_data(path)
        assert [axis.name for axis in raw_data.axes] == [
            "Readout frequency",
            "JPA Flux value",
        ]
        assert raw_data.z.shape == (n_flux, n_freq)

        loaded = adapter.load(LoadDataRequest(data_path=path, md=MetaDict(), ml=ml))
        assert isinstance(loaded, OneToneFluxResult)
        np.testing.assert_array_equal(loaded.fluxes, fluxes)
        np.testing.assert_array_equal(loaded.freqs, freqs)
        assert loaded.signals.shape == (n_flux, n_freq)
        assert loaded.signals.dtype == np.complex128
        np.testing.assert_allclose(loaded.signals, signals, rtol=0, atol=0)
        assert loaded.cfg_snapshot is not None
        assert isinstance(loaded.cfg_snapshot, OneToneFluxCfg)
        # The snapshot round-trips the exact cfg built by the adapter.
        assert loaded.cfg_snapshot.sweep.freq.expts == cfg.sweep.freq.expts
        assert loaded.cfg_snapshot.sweep.jpa_flux.expts == cfg.sweep.jpa_flux.expts
    finally:
        _drop_fake("yoko")


# --- A4: honest no-analysis capability, no writeback -----------------------


def test_capabilities_declare_no_analysis() -> None:
    caps = JpaFluxOneToneAdapter.capabilities
    assert caps.analysis is AnalysisMode.NONE
    assert caps.requires_soc is True


def test_no_writeback_items_are_produced() -> None:
    adapter = JpaFluxOneToneAdapter()
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


# --- A5: honest cfg — no invented RF selector; pump state in guide ----------


def test_cfg_defines_only_the_flux_device_selector() -> None:
    """The core experiment commands only the flux device — the cfg must not
    invent an RF selector the core never uses."""

    schema = JpaFluxOneToneAdapter().make_default_cfg(_make_ctx())
    dev_section = schema.value.fields["dev"]
    assert isinstance(dev_section, CfgSectionValue)
    assert set(dev_section.fields) == {JPA_FLUX_ROLE_KEY}
    assert JPA_RF_ROLE_KEY not in dev_section.fields

    dev_spec_section = schema.spec.fields["dev"]
    assert isinstance(dev_spec_section, CfgSectionSpec)
    assert set(dev_spec_section.fields) == {JPA_FLUX_ROLE_KEY}


def test_guide_tells_operator_to_confirm_pump_state_herself() -> None:
    guide = JpaFluxOneToneAdapter.guide()
    text = f"{guide.behavior} {guide.recommended}".lower()
    assert "pump" in text
    assert "confirm" in text


def test_sweep_label_and_guide_use_neutral_flux_device_value_wording() -> None:
    guide = JpaFluxOneToneAdapter.guide()
    text = f"{guide.behavior} {guide.typical_writeback}".lower()
    assert "flux device value" in text
    import re

    for unit_claim in (r"a\.u\.", r"ampere", r"\bmA\b", r"current \(a\)"):
        assert re.search(unit_claim, text) is None

    from zcu_tools.gui.cfg import SweepSpec

    schema = JpaFluxOneToneAdapter().make_default_cfg(_make_ctx())
    sweep_spec_section = schema.spec.fields["sweep"]
    assert isinstance(sweep_spec_section, CfgSectionSpec)
    sweep_spec = sweep_spec_section.fields["jpa_flux"]
    assert isinstance(sweep_spec, SweepSpec)
    assert sweep_spec.label == "JPA flux device value"


# --- A6: checks ------------------------------------------------------------


def test_cfg_definition_type() -> None:
    assert isinstance(JpaFluxOneToneAdapter.cfg_definition(), MeasureCfgDefinition)
