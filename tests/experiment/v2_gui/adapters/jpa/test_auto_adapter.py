"""jpa/auto_optimize adapter — registry reachability, device preflight,
run adaptation, persistence, analysis writeback and operator guide."""

from __future__ import annotations

from typing import Any, Literal
from unittest.mock import MagicMock

import numpy as np
import pytest
from matplotlib.figure import Figure
from zcu_tools.device import (
    AnritsuMG3692Info,
    BaseDeviceInfo,
    FakeDevice,
    FakeDeviceInfo,
    GlobalDeviceManager,
    RohdeSchwarzSGS100AInfo,
    YOKOGS200Info,
)
from zcu_tools.experiment.v2.jpa import AutoOptimizeExp, JPAOptCfg
from zcu_tools.experiment.v2.jpa.jpa_auto_optimize import (
    JPA_AUTO_FLUX_ROLE,
    JPA_AUTO_FREQ_ROLE,
    JPA_AUTO_GROUPED_ROLES,
    JPA_AUTO_PHASE_ROLE,
    JPA_AUTO_POWER_ROLE,
    JPA_AUTO_SNR_ROLE,
    JPAOptimizeResult,
)
from zcu_tools.experiment.v2_gui.adapters._support import MeasureCfgDefinition
from zcu_tools.experiment.v2_gui.adapters.jpa import (
    JpaAutoAnalyzeResult,
    JpaAutoOptimizeAdapter,
)
from zcu_tools.experiment.v2_gui.adapters.jpa._shared import (
    JPA_FLUX_LABEL,
    JPA_FLUX_ROLE_KEY,
    JPA_RF_LABEL,
    JPA_RF_ROLE_KEY,
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
from zcu_tools.gui.cfg import CfgSectionValue, DirectValue, EvalValue, SweepValue
from zcu_tools.meta_tool import MetaDict, ModuleLibrary
from zcu_tools.utils.datasaver import DatasetRole, load_grouped_labber_data

_SGS = RohdeSchwarzSGS100AInfo(address="sgs")
_SGS2 = RohdeSchwarzSGS100AInfo(address="sgs2")
_YOKO = YOKOGS200Info(address="yoko")
_ANRITSU = AnritsuMG3692Info(address="anritsu")


class _FreqOnlyInfo(BaseDeviceInfo):
    """Probe knob matrix: frequency only — no power knob."""

    type: Literal["_FreqOnlyInfo"] = "_FreqOnlyInfo"

    def set_freq(self, freq_Hz: float) -> None:
        del freq_Hz


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
    dev: dict[str, str] = {
        JPA_RF_ROLE_KEY: dev_selection.get(JPA_RF_ROLE_KEY, ""),
        JPA_FLUX_ROLE_KEY: dev_selection.get(JPA_FLUX_ROLE_KEY, ""),
    }
    # A fresh cfg always carries a legal iteration budget.
    return {"dev": dev, "num_points": 1001}


def _register_fake(name: str) -> None:
    GlobalDeviceManager.register_device(name, FakeDevice())


def _drop_fake(name: str) -> None:
    GlobalDeviceManager.drop_device(name, ignore_error=True)


def _sample_result(cfg: JPAOptCfg | None = None) -> JPAOptimizeResult:
    params = np.array(
        [
            [1.1e-3, 7234.1, -20.0],
            [1.2e-3, 7235.2, -18.0],
            [1.3e-3, 7236.3, -16.0],
            [1.4e-3, 7237.4, -14.0],
        ],
        dtype=np.float64,
    )
    phases = np.array([0, 1, 2, 3], dtype=np.int32)
    signals = np.array([2.0, 4.5, 6.0, 5.25], dtype=np.float64)
    return JPAOptimizeResult(
        params=params,
        phases=phases,
        signals=signals,
        cfg_snapshot=cfg,
    )


# --- A1: registry reachability + fresh cfg --------------------------------


def test_jpa_auto_optimize_registered_listable_and_creatable() -> None:
    assert "jpa/auto_optimize" in ADAPTERS
    assert ADAPTERS["jpa/auto_optimize"] is JpaAutoOptimizeAdapter

    registry = Registry()
    register_all(registry)
    assert "jpa/auto_optimize" in registry.list_names()
    adapter = registry.create("jpa/auto_optimize")
    assert isinstance(adapter, JpaAutoOptimizeAdapter)
    assert isinstance(adapter, ExpAdapterProtocol)


def test_fresh_cfg_carries_both_devices_three_bounds_and_legal_num_points() -> None:
    ml = _make_ml()
    ctx = _make_ctx(ml, r_f=6500.0)
    schema = JpaAutoOptimizeAdapter().make_default_cfg(ctx)  # validates internally

    raw = schema_to_raw_dict(schema, ctx.md, ml)
    dev_raw = raw["dev"]
    assert isinstance(dev_raw, dict)
    assert dev_raw == {JPA_RF_ROLE_KEY: "", JPA_FLUX_ROLE_KEY: ""}
    assert raw["num_points"] == 1001  # legal iteration budget

    sweep_raw = raw["sweep"]
    assert isinstance(sweep_raw, dict)
    for name in ("jpa_flux", "jpa_freq", "jpa_power"):
        from zcu_tools.program.v2 import SweepCfg

        assert isinstance(sweep_raw[name], SweepCfg), name

    sweep_section = schema.value.fields["sweep"]
    assert isinstance(sweep_section, CfgSectionValue)
    for name in ("jpa_flux", "jpa_freq", "jpa_power"):
        gui_sweep = sweep_section.fields[name]
        assert isinstance(gui_sweep, SweepValue), name
    num_points = schema.value.fields["num_points"]
    assert num_points == DirectValue(1001)


def test_fresh_sweep_bounds_prefer_existing_best_values() -> None:
    ctx = _make_ctx(
        _make_ml(),
        best_jpa_flux=0.02,
        best_jpa_freq=12900.0,
        best_jpa_power=-12.0,
    )
    schema = JpaAutoOptimizeAdapter().make_default_cfg(ctx)
    sweep_section = schema.value.fields["sweep"]
    assert isinstance(sweep_section, CfgSectionValue)

    flux = sweep_section.fields["jpa_flux"]
    assert isinstance(flux, SweepValue)
    flux_start = flux.start
    assert isinstance(flux_start, EvalValue)
    assert flux_start.expr == "best_jpa_flux - 0.005"
    flux_stop = flux.stop
    assert isinstance(flux_stop, EvalValue)
    assert flux_stop.expr == "best_jpa_flux + 0.005"

    freq = sweep_section.fields["jpa_freq"]
    assert isinstance(freq, SweepValue)
    freq_start = freq.start
    assert isinstance(freq_start, EvalValue)
    assert freq_start.expr == "(1 - 0.02) * (best_jpa_freq)"
    freq_stop = freq.stop
    assert isinstance(freq_stop, EvalValue)
    assert freq_stop.expr == "(1 + 0.02) * (best_jpa_freq)"

    power = sweep_section.fields["jpa_power"]
    assert isinstance(power, SweepValue)
    power_start = power.start
    assert isinstance(power_start, EvalValue)
    assert power_start.expr == "best_jpa_power - 5.0"
    power_stop = power.stop
    assert isinstance(power_stop, EvalValue)
    assert power_stop.expr == "best_jpa_power + 5.0"


def test_fresh_sweep_bounds_fall_back_to_literal_seeds_without_md() -> None:
    ctx = _make_ctx(_make_ml())
    schema = JpaAutoOptimizeAdapter().make_default_cfg(ctx)
    sweep_section = schema.value.fields["sweep"]
    assert isinstance(sweep_section, CfgSectionValue)

    flux = sweep_section.fields["jpa_flux"]
    assert isinstance(flux, SweepValue)
    assert flux.start == -5.0e-3
    assert flux.stop == 5.0e-3
    assert flux.expts == 101

    freq = sweep_section.fields["jpa_freq"]
    assert isinstance(freq, SweepValue)
    assert freq.start == 13000.0 * (1 - 0.02)
    assert freq.stop == 13000.0 * (1 + 0.02)
    assert freq.expts == 101

    power = sweep_section.fields["jpa_power"]
    assert isinstance(power, SweepValue)
    assert power.start == -20.0
    assert power.stop == -5.0
    assert power.expts == 101


# --- A2: run adaptation ----------------------------------------------------


def test_run_passes_num_points_as_explicit_argument_only(monkeypatch) -> None:
    _register_fake("sgs")
    _register_fake("yoko")
    try:
        ml = _make_ml()
        adapter = JpaAutoOptimizeAdapter()
        ctx = _make_ctx(ml)
        req = _make_req(ml, soc=MagicMock(), soccfg=MagicMock())
        schema = adapter.make_default_cfg(ctx)
        dev_section = schema.value.fields["dev"]
        assert isinstance(dev_section, CfgSectionValue)
        dev_section.fields[JPA_RF_ROLE_KEY] = DirectValue("sgs")
        dev_section.fields[JPA_FLUX_ROLE_KEY] = DirectValue("yoko")

        captured: dict[str, Any] = {}

        def _fake_run(
            soc: Any, soccfg: Any, cfg: JPAOptCfg, num_points: int
        ) -> JPAOptimizeResult:
            captured["soc"] = soc
            captured["soccfg"] = soccfg
            captured["cfg"] = cfg
            captured["num_points"] = num_points
            return _sample_result()

        monkeypatch.setattr(AutoOptimizeExp, "run", staticmethod(_fake_run))
        result = adapter.run(req, schema)

        assert isinstance(result, JPAOptimizeResult)
        assert captured["num_points"] == 1001  # explicit core run argument
        cfg = captured["cfg"]
        assert isinstance(cfg, JPAOptCfg)
        # num_points never enters the Experiment cfg.
        assert "num_points" not in type(cfg).model_fields
        assert cfg.dev is not None
        assert cfg.dev["sgs"].label == JPA_RF_LABEL
        assert cfg.dev["yoko"].label == JPA_FLUX_LABEL
    finally:
        _drop_fake("sgs")
        _drop_fake("yoko")


def test_build_exp_cfg_pops_num_points_before_model_validation() -> None:
    _register_fake("sgs")
    _register_fake("yoko")
    try:
        ml = _make_ml()
        adapter = JpaAutoOptimizeAdapter()
        ctx = _make_ctx(ml)
        req = _make_req(ml)
        schema = adapter.make_default_cfg(ctx)
        raw = schema_to_raw_dict(schema, ctx.md, ml)
        raw["dev"] = {JPA_RF_ROLE_KEY: "sgs", JPA_FLUX_ROLE_KEY: "yoko"}
        assert raw["num_points"] == 1001
        cfg = adapter.build_exp_cfg(raw, req)
        assert isinstance(cfg, JPAOptCfg)
        assert "num_points" not in type(cfg).model_fields
    finally:
        _drop_fake("sgs")
        _drop_fake("yoko")


@pytest.mark.parametrize("num_points", [0, 1, 2, 3])
def test_run_request_fast_fails_num_points_below_four_before_device_checks(
    num_points: int,
) -> None:
    adapter = JpaAutoOptimizeAdapter()
    raw = _raw_with_selection(jpa_rf_dev="sgs", jpa_flux_dev="yoko")
    raw["num_points"] = num_points
    with pytest.raises(ValueError, match="num_points >= 4"):
        adapter.validate_run_request(_make_req(), raw)


def test_validate_run_request_rejects_non_integer_num_points() -> None:
    adapter = JpaAutoOptimizeAdapter()
    raw = _raw_with_selection()
    raw["num_points"] = "many"
    with pytest.raises(ValueError, match="num_points must be an integer"):
        adapter.validate_run_request(_make_req(), raw)


def test_run_num_points_below_four_fails_before_hardware_work(
    monkeypatch,
) -> None:
    """The core fast-fail fires before sampling or device setup; the adapter
    never gets past lowering for a too-small budget."""

    import zcu_tools.experiment.v2.jpa.jpa_auto_optimize as jpa_mod

    def boom(*args, **kwargs) -> None:
        raise AssertionError("auto-optimize reached sampling or device setup")

    monkeypatch.setattr(jpa_mod, "JPAOptimizer", boom)
    monkeypatch.setattr(jpa_mod, "setup_devices", boom)
    monkeypatch.setattr(jpa_mod, "Schedule", boom)

    _register_fake("sgs")
    _register_fake("yoko")
    try:
        ml = _make_ml()
        adapter = JpaAutoOptimizeAdapter()
        ctx = _make_ctx(ml)
        req = _make_req(ml, soc=MagicMock(), soccfg=MagicMock())
        schema = adapter.make_default_cfg(ctx)
        dev_section = schema.value.fields["dev"]
        assert isinstance(dev_section, CfgSectionValue)
        dev_section.fields[JPA_RF_ROLE_KEY] = DirectValue("sgs")
        dev_section.fields[JPA_FLUX_ROLE_KEY] = DirectValue("yoko")
        schema.value.fields["num_points"] = DirectValue(3)

        with pytest.raises(ValueError, match="num_points >= 4"):
            adapter.run(req, schema)
    finally:
        _drop_fake("sgs")
        _drop_fake("yoko")


# --- A3: device preflight --------------------------------------------------


def test_lower_jpa_auto_devs_requires_both_selections() -> None:
    with pytest.raises(ValueError, match="missing JPA RF device selection"):
        _lower_via_validate({}, {})
    with pytest.raises(ValueError, match="missing JPA RF device selection"):
        _lower_via_validate({"dev": {JPA_RF_ROLE_KEY: ""}}, {})
    with pytest.raises(ValueError, match="missing JPA flux device selection"):
        _lower_via_validate(
            {"dev": {JPA_RF_ROLE_KEY: "sgs", JPA_FLUX_ROLE_KEY: ""}},
            {"sgs": _SGS},
        )


def _lower_via_validate(raw: dict[str, object], snapshot: dict[str, Any]) -> None:
    # validate_run_request is the observable preflight entry point; the
    # snapshot is injected by monkeypatching cached_device_snapshot. A fresh
    # cfg always carries a legal num_points, so inject one for device-focused
    # preflight checks.
    validated_raw = dict(raw)
    validated_raw.setdefault("num_points", 1001)
    adapter = JpaAutoOptimizeAdapter()
    import zcu_tools.experiment.v2_gui.adapters.jpa.auto as auto_mod

    original = auto_mod.cached_device_snapshot

    def _snapshot() -> dict[str, Any]:
        return snapshot

    auto_mod.cached_device_snapshot = _snapshot
    try:
        adapter.validate_run_request(_make_req(), validated_raw)
    finally:
        auto_mod.cached_device_snapshot = original


def test_lower_jpa_auto_devs_requires_known_devices() -> None:
    with pytest.raises(ValueError, match="not found in the device snapshot"):
        _lower_via_validate(
            _raw_with_selection(jpa_rf_dev="ghost", jpa_flux_dev="yoko"),
            {"yoko": _YOKO},
        )
    with pytest.raises(ValueError, match="not found in the device snapshot"):
        _lower_via_validate(
            _raw_with_selection(jpa_rf_dev="sgs", jpa_flux_dev="ghost"),
            {"sgs": _SGS},
        )


def test_lower_jpa_auto_devs_requires_freq_and_power_knobs_on_rf() -> None:
    # YOKO has no frequency knob.
    with pytest.raises(ValueError, match="does not support the frequency knob"):
        _lower_via_validate(
            _raw_with_selection(jpa_rf_dev="yoko", jpa_flux_dev="yoko2"),
            {"yoko": _YOKO, "yoko2": _YOKO},
        )
    # Anritsu exposes set_frequency, not the set_freq knob.
    with pytest.raises(ValueError, match="does not support the frequency knob"):
        _lower_via_validate(
            _raw_with_selection(jpa_rf_dev="anritsu", jpa_flux_dev="yoko"),
            {"anritsu": _ANRITSU, "yoko": _YOKO},
        )
    # Frequency-only probe: passes the freq knob, fails the power knob.
    with pytest.raises(ValueError, match="does not support the power knob"):
        _lower_via_validate(
            _raw_with_selection(jpa_rf_dev="freqonly", jpa_flux_dev="yoko"),
            {"freqonly": _FreqOnlyInfo(address="freqonly"), "yoko": _YOKO},
        )


def test_lower_jpa_auto_devs_requires_flux_knob_on_flux_device() -> None:
    with pytest.raises(ValueError, match="does not support the flux knob"):
        _lower_via_validate(
            _raw_with_selection(jpa_rf_dev="sgs", jpa_flux_dev="sgs2"),
            {"sgs": _SGS, "sgs2": _SGS2},
        )


def test_lower_jpa_auto_devs_produces_exactly_two_labeled_patches() -> None:
    snapshot = {"sgs": _SGS, "sgs2": _SGS2, "yoko": _YOKO}
    from zcu_tools.experiment.v2_gui.adapters.jpa.auto import _lower_jpa_auto_devs

    patch = _lower_jpa_auto_devs(
        _raw_with_selection(jpa_rf_dev="sgs", jpa_flux_dev="yoko"),
        snapshot,
    )
    # Exactly two selected devices, each with one role label.
    assert patch == {
        "sgs": {"label": JPA_RF_LABEL},
        "yoko": {"label": JPA_FLUX_LABEL},
    }
    assert set(patch) == {"sgs", "yoko"}
    assert all(set(entry) == {"label"} for entry in patch.values())


def test_lower_jpa_auto_devs_rejects_same_device_for_both_roles() -> None:
    # FakeDeviceInfo supports all three knobs, so only the duplicate-role
    # guard can reject this selection.
    from zcu_tools.experiment.v2_gui.adapters.jpa.auto import _lower_jpa_auto_devs

    fake = FakeDeviceInfo(address="fake")
    with pytest.raises(ValueError, match="distinct RF and flux devices"):
        _lower_jpa_auto_devs(
            _raw_with_selection(jpa_rf_dev="fake", jpa_flux_dev="fake"),
            {"fake": fake},
        )


def test_validate_run_request_preflights_both_selections(monkeypatch) -> None:
    snapshot = {"sgs": _SGS, "yoko": _YOKO}
    monkeypatch.setattr(
        "zcu_tools.experiment.v2_gui.adapters.jpa.auto.cached_device_snapshot",
        lambda: snapshot,
    )
    adapter = JpaAutoOptimizeAdapter()
    req = _make_req()

    adapter.validate_run_request(
        req, _raw_with_selection(jpa_rf_dev="sgs", jpa_flux_dev="yoko")
    )

    with pytest.raises(ValueError, match="missing JPA RF device selection"):
        adapter.validate_run_request(req, _raw_with_selection())
    with pytest.raises(ValueError, match="not found"):
        adapter.validate_run_request(
            req, _raw_with_selection(jpa_rf_dev="ghost", jpa_flux_dev="yoko")
        )
    with pytest.raises(ValueError, match="flux knob"):
        adapter.validate_run_request(
            req, _raw_with_selection(jpa_rf_dev="sgs", jpa_flux_dev="sgs")
        )


def test_preflight_refusals_and_pass_happen_without_hardware_queries(
    monkeypatch,
) -> None:
    """The production preflight path never queries or commands a live device.

    The registry membership is patched to return probe devices whose
    ``get_info`` would raise — so any hardware query in preflight would fail
    the test. All applicable refusals (and the pass) must still complete
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
        lambda: _devices(
            sgs=RohdeSchwarzSGS100AInfo,
            yoko=YOKOGS200Info,
            freqonly=_FreqOnlyInfo,
        ),
    )
    adapter = JpaAutoOptimizeAdapter()
    req = _make_req()

    adapter.validate_run_request(
        req, _raw_with_selection(jpa_rf_dev="sgs", jpa_flux_dev="yoko")
    )

    with pytest.raises(ValueError, match="missing JPA RF device selection"):
        adapter.validate_run_request(req, _raw_with_selection())
    with pytest.raises(ValueError, match="not found"):
        adapter.validate_run_request(
            req, _raw_with_selection(jpa_rf_dev="ghost", jpa_flux_dev="yoko")
        )
    with pytest.raises(ValueError, match="power knob"):
        adapter.validate_run_request(
            req, _raw_with_selection(jpa_rf_dev="freqonly", jpa_flux_dev="yoko")
        )
    with pytest.raises(ValueError, match="flux knob"):
        adapter.validate_run_request(
            req, _raw_with_selection(jpa_rf_dev="sgs", jpa_flux_dev="sgs")
        )

    assert queried == []  # preflight completed without a single hardware query


# --- A4: canonical grouped persistence -------------------------------------


def test_canonical_save_writes_single_five_role_grouped_file(tmp_path) -> None:
    _register_fake("sgs")
    _register_fake("yoko")
    try:
        ml = _make_ml()
        adapter = JpaAutoOptimizeAdapter()
        ctx = _make_ctx(ml)
        req = _make_req(ml)
        schema = adapter.make_default_cfg(ctx)
        raw = schema_to_raw_dict(schema, ctx.md, ml)
        raw["dev"] = {JPA_RF_ROLE_KEY: "sgs", JPA_FLUX_ROLE_KEY: "yoko"}
        cfg = adapter.build_exp_cfg(raw, req)
        assert isinstance(cfg, JPAOptCfg)

        result = _sample_result(cfg=cfg)
        base = tmp_path / "jpa_auto"
        adapter.save(
            SaveDataRequest(
                data_path=str(base),
                run_result=result,
                md=MetaDict(),
                ml=ml,
                chip_name="chip",
                qub_name="Q1",
                res_name="R1",
                active_label="1",
            )
        )

        path = tmp_path / "jpa_auto.hdf5"
        assert path.exists()
        # No legacy sidecars.
        assert list(tmp_path.glob("jpa_auto*_params*.hdf5")) == []
        assert list(tmp_path.glob("jpa_auto*_phases*.hdf5")) == []
        assert list(tmp_path.glob("jpa_auto*_signals*.hdf5")) == []

        grouped = load_grouped_labber_data(
            str(path), required_roles=JPA_AUTO_GROUPED_ROLES
        )
        assert list(grouped.roles) == [
            DatasetRole(role) for role in JPA_AUTO_GROUPED_ROLES
        ]
        assert np.allclose(
            grouped.roles[DatasetRole(JPA_AUTO_FLUX_ROLE)].z, result.params[:, 0]
        )
        assert np.allclose(
            grouped.roles[DatasetRole(JPA_AUTO_FREQ_ROLE)].z, result.params[:, 1] * 1e6
        )
        assert np.allclose(
            grouped.roles[DatasetRole(JPA_AUTO_POWER_ROLE)].z, result.params[:, 2]
        )
        assert np.array_equal(
            grouped.roles[DatasetRole(JPA_AUTO_PHASE_ROLE)].z,
            result.phases.astype(np.int64),
        )
        assert np.allclose(
            grouped.roles[DatasetRole(JPA_AUTO_SNR_ROLE)].z, result.signals
        )

        loaded = adapter.load(
            LoadDataRequest(data_path=str(path), md=MetaDict(), ml=ml)
        )
        assert isinstance(loaded, JPAOptimizeResult)
        np.testing.assert_allclose(loaded.params, result.params)
        np.testing.assert_array_equal(loaded.phases, result.phases)
        np.testing.assert_allclose(loaded.signals, result.signals)
        assert loaded.cfg_snapshot is not None
        assert isinstance(loaded.cfg_snapshot, JPAOptCfg)
    finally:
        _drop_fake("sgs")
        _drop_fake("yoko")


# --- A5/A6: analysis + writeback -------------------------------------------


def test_analyze_projects_best_flux_freq_power_and_figure() -> None:
    result = _sample_result()
    adapter = JpaAutoOptimizeAdapter()
    analyze_result = adapter.analyze(
        AnalyzeRequest(
            run_result=result,
            analyze_params=NoAnalyzeParams(),
            md=MetaDict(),
            ml=ModuleLibrary(),
            predictor=None,
        )
    )
    assert isinstance(analyze_result, JpaAutoAnalyzeResult)
    # argmax of |signals| is index 2.
    assert analyze_result.best_flux == pytest.approx(1.3e-3)
    assert analyze_result.best_freq == pytest.approx(7236.3)
    assert analyze_result.best_power == pytest.approx(-16.0)
    assert isinstance(analyze_result.figure, Figure)

    # Neutral flux-axis wording on the review figure; freq/power axes keep
    # their own quantities.
    labels = {ax.get_xlabel() for ax in analyze_result.figure.axes}
    assert "JPA flux device value" in labels
    assert "JPA Frequency (MHz)" in labels
    assert "JPA Power (dBm)" in labels


def test_analyze_and_writeback_never_touch_live_devices_or_cur_jpa_A() -> None:
    adapter = JpaAutoOptimizeAdapter()
    analyze_result = adapter.analyze(
        AnalyzeRequest(
            run_result=_sample_result(),
            analyze_params=NoAnalyzeParams(),
            md=MetaDict(),
            ml=ModuleLibrary(),
            predictor=None,
        )
    )
    items = list(
        adapter.get_writeback_items(
            WritebackRequest(
                run_result=MagicMock(),
                analyze_result=analyze_result,
                ctx=_make_ctx(),
            )
        )
    )
    # Exactly three independently selectable MetaDict writebacks.
    assert len(items) == 3
    targets = [item.target_name for item in items]
    assert targets == ["best_jpa_flux", "best_jpa_freq", "best_jpa_power"]
    assert len(set(targets)) == 3
    for item in items:
        assert isinstance(item, MetaDictWriteback)
        assert "cur_jpa_A" not in item.target_name
    flux_item, freq_item, power_item = items
    assert isinstance(flux_item, MetaDictWriteback)
    assert isinstance(freq_item, MetaDictWriteback)
    assert isinstance(power_item, MetaDictWriteback)
    assert flux_item.proposed_value == analyze_result.best_flux
    assert freq_item.proposed_value == analyze_result.best_freq
    assert power_item.proposed_value == analyze_result.best_power


# --- A7: operator guide ----------------------------------------------------


def test_guide_warns_to_review_devices_and_bounds() -> None:
    guide = JpaAutoOptimizeAdapter.guide()
    text = f"{guide.behavior} {guide.recommended}".lower()
    assert "review" in text
    assert "device" in text
    assert "bounds" in text


def test_guide_explains_num_points_and_separate_writebacks() -> None:
    guide = JpaAutoOptimizeAdapter.guide()
    assert "num_points" in guide.recommended
    assert "1001" in guide.recommended
    assert "best_jpa_flux" in guide.typical_writeback
    assert "best_jpa_freq" in guide.typical_writeback
    assert "best_jpa_power" in guide.typical_writeback
    assert "never touches 'cur_jpa_A'" in guide.typical_writeback


def test_cfg_definition_type() -> None:
    assert isinstance(JpaAutoOptimizeAdapter.cfg_definition(), MeasureCfgDefinition)
