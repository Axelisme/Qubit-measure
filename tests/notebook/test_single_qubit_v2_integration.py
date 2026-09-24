"""Executable-interface probes for the single_qubit notebook v2 integration.

Reads ``notebook_md/single_qubit.md`` with jupytext, extracts the relevant
cells and validates the A1-A5 behaviors with fakes — no hardware, no live
VISA, no result data.
"""

from __future__ import annotations

import ast
import inspect
import sys
import types
from pathlib import Path

import jupytext
import numpy as np
import pandas as pd
import pytest

from zcu_tools.device import GlobalDeviceManager
from zcu_tools.device.base import BaseDevice, BaseDeviceInfo
from zcu_tools.experiment.v2.lookback import LookbackExp
from zcu_tools.meta_tool import (
    SampleTable,
    SampleTableV2Error,
    validate_sample_table_v2,
)

NOTEBOOK_PATH = Path(__file__).resolve().parents[2] / "notebook_md" / "single_qubit.md"

_FORBIDDEN_HW_CALLS = (
    "output_off",
    "output_on",
    "set_current",
    "set_voltage",
    "set_power",
    "set_frequency",
    "set_mode",
    "IQ_off",
    "ramp",
    "reset",
)


def _code_cells() -> list[str]:
    nb = jupytext.read(NOTEBOOK_PATH)
    return [c.source for c in nb.cells if c.cell_type == "code"]


def _cell(*needles: str) -> str:
    for source in _code_cells():
        if all(needle in source for needle in needles):
            return source
    raise AssertionError(f"no code cell containing {needles}")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _FakeMD:
    """Minimal MetaDict stand-in: attribute reads + .get() with default."""

    def __init__(self, data: dict[str, object]) -> None:
        self._data = dict(data)

    def __getattr__(self, name: str) -> object:
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(name) from None

    def get(self, name: str, default: object = None) -> object:
        return self._data.get(name, default)


class _RecordingTable:
    """Records the validation/append order and the appended row dict."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.added: list[dict[str, object]] = []
        self.samples = pd.DataFrame()

    def add_sample(self, **kwargs) -> None:
        self.calls.append("add_sample")
        self.added.append(kwargs)


_MEASUREMENT_DEFAULTS = {
    "q_f": 4000.0,
    "t1": 10.0,
    "t1err": 0.1,
    "t2r": 20.0,
    "t2r_err": 0.2,
    "t2e": 30.0,
    "t2e_err": 0.3,
}


def _exec_sample_row_cell(
    md_data: dict[str, object],
    cur_value: float,
    table: object,
    *,
    validate: object = validate_sample_table_v2,
) -> dict[str, object]:
    cell = _cell("sample_table.add_sample", '"dev_value"')
    ns: dict[str, object] = {
        "np": np,
        "md": _FakeMD({**_MEASUREMENT_DEFAULTS, **md_data}),
        "cur_value": cur_value,
        "sample_table": table,
        "validate_sample_table_v2": validate,
    }
    exec(compile(cell, "<save-sample-cell>", "exec"), ns)
    return ns


def _reset_registry() -> None:
    GlobalDeviceManager._devices.clear()
    GlobalDeviceManager._close_claims.clear()


class _FakeSession:
    read_termination = "\n"
    write_termination = "\n"


class _FakeRM:
    def __init__(self, log: list[str]) -> None:
        self.log = log
        self.closed = False

    def open_resource(self, address: str) -> _FakeSession:
        return _FakeSession()

    def close(self) -> None:
        self.log.append("rm_close")
        self.closed = True


FLUX_YOKO_ADDR = "USB0::0x0B21::0x0039::91WB18859::INSTR"
JPA_YOKO_ADDR = "USB0::0x0B21::0x0039::91T810992::INSTR"
JPA_SGS_ADDR = "TCPIP0::192.168.10.89::inst0::INSTR"


class _FakeYokoDeviceInfo(BaseDeviceInfo):
    pass


class _FakeYoko(BaseDevice):
    info_model = _FakeYokoDeviceInfo
    # class-level log shared by devices constructed from notebook cells
    _shared_log: list[str] | None = None

    def __init__(
        self,
        address: str,
        rm: object = None,
        *,
        name: str = "",
        log: list[str] | None = None,
        fail_close: bool = False,
    ) -> None:
        self.address = address
        self.rm = rm
        self.name = name
        if log is not None:
            self.log = log
        elif _FakeYoko._shared_log is not None:
            self.log = _FakeYoko._shared_log
        else:
            self.log = []
        self.fail_close = fail_close
        self.session = None
        self.log.append(f"construct:{self._tag}")

    @property
    def _tag(self) -> str:
        return self.name or self.address

    def _setup(self, cfg, progress=True, stop_event=None) -> None:
        pass

    def get_info(self) -> dict[str, object]:
        return {}

    def close(self) -> None:
        self.log.append(f"close:{self._tag}")
        if self.fail_close:
            raise RuntimeError(f"boom {self._tag}")

    def set_mode(self, *args, **kwargs) -> None:
        self.log.append(f"set_mode:{self._tag}")


@pytest.fixture
def fake_modules(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Install fake pyvisa/yoko/sgs modules for one test, then restore them.

    monkeypatch restores every prior ``sys.modules`` entry (and
    ``_FakeYoko._shared_log``) after the test, so lifecycle tests stay
    independent: a later import never resolves ``pyvisa``, ``YOKOGS200`` or
    ``RohdeSchwarzSGS100A`` to a leftover fake.
    """
    log: list[str] = []
    monkeypatch.setattr(_FakeYoko, "_shared_log", log)
    fake_pyvisa = types.ModuleType("pyvisa")
    fake_pyvisa.ResourceManager = lambda: _FakeRM(log)  # type: ignore[method-assign]
    fake_yoko = types.ModuleType("zcu_tools.device.yoko")
    fake_yoko.YOKOGS200 = _FakeYoko  # type: ignore[attr-defined]
    fake_sgs = types.ModuleType("zcu_tools.device.sgs100a")
    fake_sgs.RohdeSchwarzSGS100A = _FakeYoko  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyvisa", fake_pyvisa)
    monkeypatch.setitem(sys.modules, "zcu_tools.device.yoko", fake_yoko)
    monkeypatch.setitem(sys.modules, "zcu_tools.device.sgs100a", fake_sgs)
    return log


def _exec(ns: dict[str, object], source: str) -> None:
    exec(compile(source, "<cell>", "exec"), ns)


def _run_full_init(log: list[str]) -> dict[str, object]:
    """Run the connect -> device creation flow once; returns the cell namespace."""
    ns: dict[str, object] = {}
    _exec(ns, _cell("resource_manager = pyvisa.ResourceManager"))
    _exec(ns, _cell('close_device("flux_yoko"'))
    _exec(ns, _cell('close_device("jpa_yoko"'))
    _exec(ns, _cell('close_device("jpa_sgs"'))
    return ns


# ---------------------------------------------------------------------------
# A1 — Lookback
# ---------------------------------------------------------------------------


def test_lookback_analyze_cell_drops_stale_ro_cfg_keeps_ratio_smooth() -> None:
    from IPython.core.inputtransformer2 import TransformerManager

    cell = _cell("lookback_exp.analyze")
    tree = ast.parse(TransformerManager().transform_cell(cell))
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "analyze"
    ]
    assert len(calls) == 1
    call = calls[0]
    assert call.args == []
    kwargs = {kw.arg: kw for kw in call.keywords}
    assert "ro_cfg" not in kwargs
    assert ast.literal_eval(kwargs["ratio"].value) == 0.1
    assert ast.literal_eval(kwargs["smooth"].value) == 1.0


def test_lookback_analyze_binds_to_current_signature() -> None:
    sig = inspect.signature(LookbackExp.analyze)
    assert "ro_cfg" not in sig.parameters
    sig.bind(None, ratio=0.1, smooth=1.0)
    with pytest.raises(TypeError):
        sig.bind(None, ratio=0.1, smooth=1.0, ro_cfg=object())


# ---------------------------------------------------------------------------
# A2 — v2 sample row
# ---------------------------------------------------------------------------


def test_sample_row_uses_measurement_bound_cur_value_as_A_dev_value() -> None:
    table = _RecordingTable()
    _exec_sample_row_cell(
        {"flx_int": 0.0, "flx_period": 2.0},
        cur_value=2.0e-3,
        table=table,
    )
    row = table.added[0]
    # measurement-bound value persisted as-is in A: no live query, no x1000
    assert row["dev_value"] == 2.0e-3
    assert row["dev_unit"] == "A"
    assert "calibrated mA" not in row
    # measurement columns and comment/date preserved
    assert row["Freq (MHz)"] == 4000.0
    assert row["T1 (us)"] == 10.0
    assert row["Tcomment"] == "Manual Added"
    assert "date" in row


def test_sample_row_frame_pair_only_when_both_finite() -> None:
    cases: list[tuple[dict[str, object], bool]] = [
        ({"flx_int": 0.0, "flx_period": 2.0}, True),
        ({"flx_int": 0.0, "flx_period": None}, False),
        ({"flx_int": None, "flx_period": 2.0}, False),
        ({"flx_int": float("nan"), "flx_period": 2.0}, False),
        ({"flx_int": 0.0, "flx_period": float("inf")}, False),
        ({"flx_int": 0.0, "flx_period": 0.0}, False),
        ({}, False),
    ]
    for md_data, expect_frame in cases:
        table = _RecordingTable()
        _exec_sample_row_cell(md_data, cur_value=1.0e-3, table=table)
        row = table.added[0]
        assert ("flux_int" in row) == expect_frame, md_data
        assert ("flux_period" in row) == expect_frame, md_data
        if expect_frame:
            assert row["flux_int"] == md_data["flx_int"]
            assert row["flux_period"] == md_data["flx_period"]


def test_sample_row_validates_existing_table_before_append() -> None:
    table = _RecordingTable()
    calls: list[str] = []

    def recording_validate(samples, *, allow_empty: bool = False) -> None:
        calls.append("validate")
        validate_sample_table_v2(samples, allow_empty=allow_empty)

    _exec_sample_row_cell(
        {"flx_int": 0.0, "flx_period": 2.0},
        cur_value=1.0e-3,
        table=table,
        validate=recording_validate,
    )
    assert calls == ["validate"]
    assert table.calls == ["add_sample"]


# ---------------------------------------------------------------------------
# A3 — existing table validated before append (real SampleTable on disk)
# ---------------------------------------------------------------------------


def test_legacy_existing_table_fails_before_mutation(tmp_path: Path) -> None:
    csv = tmp_path / "samples.csv"
    csv.write_text("calibrated mA,Freq (MHz),T1 (us)\n1.0,4000.0,10.0\n")
    table = SampleTable(str(csv))
    with pytest.raises(SampleTableV2Error, match="calibrated mA"):
        _exec_sample_row_cell(
            {"flx_int": 0.0, "flx_period": 2.0}, cur_value=2.0e-3, table=table
        )
    assert csv.read_text() == "calibrated mA,Freq (MHz),T1 (us)\n1.0,4000.0,10.0\n"


def test_invalid_existing_table_fails_before_mutation(tmp_path: Path) -> None:
    csv = tmp_path / "samples.csv"
    # missing required dev_value/dev_unit columns
    csv.write_text("Freq (MHz),T1 (us)\n4000.0,10.0\n")
    table = SampleTable(str(csv))
    with pytest.raises(SampleTableV2Error, match="missing required"):
        _exec_sample_row_cell(
            {"flx_int": 0.0, "flx_period": 2.0}, cur_value=2.0e-3, table=table
        )
    assert csv.read_text() == "Freq (MHz),T1 (us)\n4000.0,10.0\n"


def test_valid_v2_existing_table_appends(tmp_path: Path) -> None:
    csv = tmp_path / "samples.csv"
    csv.write_text(
        "dev_value,dev_unit,flux_int,flux_period,Freq (MHz),T1 (us)\n"
        "0.001,A,0.0,2.0,4000.0,10.0\n"
    )
    table = SampleTable(str(csv))
    _exec_sample_row_cell(
        {"flx_int": 0.0, "flx_period": 2.0}, cur_value=2.0e-3, table=table
    )
    rows = pd.read_csv(csv)
    assert len(rows) == 2
    assert rows.iloc[1]["dev_value"] == pytest.approx(2.0e-3)
    assert rows.iloc[1]["dev_unit"] == "A"
    assert rows.iloc[1]["flux_int"] == pytest.approx(0.0)
    assert rows.iloc[1]["flux_period"] == pytest.approx(2.0)


def test_new_table_append_creates_v2_file(tmp_path: Path) -> None:
    csv = tmp_path / "samples.csv"
    table = SampleTable(str(csv))
    _exec_sample_row_cell(
        {},
        cur_value=2.0e-3,
        table=table,  # no frame metadata -> no frame columns
    )
    rows = pd.read_csv(csv)
    assert len(rows) == 1
    assert list(rows.columns)[:2] == ["dev_value", "dev_unit"]
    assert rows.iloc[0]["dev_value"] == pytest.approx(2.0e-3)
    assert rows.iloc[0]["dev_unit"] == "A"
    assert "flux_int" not in rows.columns
    assert "flux_period" not in rows.columns
    # round-trips through the v2 validator
    validate_sample_table_v2(pd.read_csv(csv))


# ---------------------------------------------------------------------------
# A4 — VISA lifecycle ordering (real GlobalDeviceManager, fake devices/RM)
# ---------------------------------------------------------------------------


def test_rm_init_closes_all_devices_before_recreating_manager(
    fake_modules: list[str],
) -> None:
    _reset_registry()
    ns = _run_full_init(fake_modules)
    rm1 = ns["resource_manager"]
    assert isinstance(rm1, _FakeRM)

    # re-run the RM init cell: close all devices, close old RM, then recreate
    _exec(ns, _cell("resource_manager = pyvisa.ResourceManager"))
    rm2 = ns["resource_manager"]
    assert rm1.closed
    assert rm2 is not rm1
    assert fake_modules[-4:] == [
        f"close:{FLUX_YOKO_ADDR}",
        f"close:{JPA_YOKO_ADDR}",
        f"close:{JPA_SGS_ADDR}",
        "rm_close",
    ]
    _reset_registry()


def test_same_name_recreation_closes_old_device_first(
    fake_modules: list[str],
) -> None:
    _reset_registry()
    ns = _run_full_init(fake_modules)
    first = ns["flux_yoko"]
    assert GlobalDeviceManager.get_device("flux_yoko") is first

    # re-run the flux_yoko creation cell: close old session, then replace
    _exec(ns, _cell('close_device("flux_yoko"'))
    second = ns["flux_yoko"]
    assert second is not first
    assert fake_modules.count(f"close:{FLUX_YOKO_ADDR}") == 1
    assert fake_modules.count(f"construct:{FLUX_YOKO_ADDR}") == 2
    assert fake_modules.index(f"close:{FLUX_YOKO_ADDR}") < fake_modules.index(
        f"construct:{FLUX_YOKO_ADDR}", 1
    )
    # the registry holds exactly one flux_yoko entry, the new identity
    assert GlobalDeviceManager.get_device("flux_yoko") is second
    _reset_registry()


def test_recreation_aborts_before_construct_on_close_failure(
    fake_modules: list[str],
) -> None:
    _reset_registry()
    ns: dict[str, object] = {}
    _exec(ns, _cell("resource_manager = pyvisa.ResourceManager"))
    failing = _FakeYoko("USB::old", name="flux_yoko", log=None, fail_close=True)
    GlobalDeviceManager.register_device("flux_yoko", failing)

    with pytest.raises(Exception) as excinfo:
        _exec(ns, _cell('close_device("flux_yoko"'))
    assert "Device close failed" in str(excinfo.value)
    # no replacement was constructed or registered; old entry retained for retry
    assert f"construct:{FLUX_YOKO_ADDR}" not in fake_modules
    assert GlobalDeviceManager.get_device("flux_yoko") is failing
    _reset_registry()


def test_close_all_failure_retains_manager_handle(
    fake_modules: list[str],
) -> None:
    _reset_registry()
    ns: dict[str, object] = {}
    _exec(ns, _cell("resource_manager = pyvisa.ResourceManager"))
    rm1 = ns["resource_manager"]
    ok_dev = _FakeYoko("USB::ok", name="jpa_yoko", log=fake_modules)
    bad_dev = _FakeYoko("USB::bad", name="flux_yoko", log=fake_modules, fail_close=True)
    GlobalDeviceManager.register_device("flux_yoko", bad_dev)
    GlobalDeviceManager.register_device("jpa_yoko", ok_dev)

    with pytest.raises(ExceptionGroup):
        _exec(ns, _cell("resource_manager = pyvisa.ResourceManager"))
    # the healthy device was still closed; the RM was NOT closed and the
    # handle was retained so the user can retry or diagnose
    assert "close:jpa_yoko" in fake_modules
    assert "rm_close" not in fake_modules
    assert ns["resource_manager"] is rm1
    assert isinstance(rm1, _FakeRM)
    assert not rm1.closed
    _reset_registry()


def test_final_disconnect_closes_devices_then_rm_then_none(
    fake_modules: list[str],
) -> None:
    _reset_registry()
    ns = _run_full_init(fake_modules)
    rm1 = ns["resource_manager"]

    _exec(ns, _cell("close_all_devices()", "resource_manager = None"))
    assert fake_modules[-4:] == [
        f"close:{FLUX_YOKO_ADDR}",
        f"close:{JPA_YOKO_ADDR}",
        f"close:{JPA_SGS_ADDR}",
        "rm_close",
    ]
    assert isinstance(rm1, _FakeRM)
    assert rm1.closed
    assert ns["resource_manager"] is None
    assert GlobalDeviceManager.get_all_devices() == {}
    # idempotent: running the disconnect cell again is a no-op
    _exec(ns, _cell("close_all_devices()", "resource_manager = None"))
    assert fake_modules.count("rm_close") == 1
    _reset_registry()


# ---------------------------------------------------------------------------
# A5 — lifecycle cells add no hardware-state mutation
# ---------------------------------------------------------------------------


def test_disconnect_and_rm_init_cells_have_no_hardware_state_mutation() -> None:
    disconnect = _cell("close_all_devices()", "resource_manager = None")
    rm_init = _cell("resource_manager = pyvisa.ResourceManager")
    for source in (disconnect, rm_init):
        tree = ast.parse(source)
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in _FORBIDDEN_HW_CALLS
        ]
        assert calls == [], (source, calls)


def test_recreation_cells_close_before_construct_in_source() -> None:
    for name in ("flux_yoko", "jpa_yoko", "jpa_sgs"):
        cell = _cell(f'close_device("{name}"')
        tree = ast.parse(cell)
        close = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "close_device"
            and any(isinstance(a, ast.Constant) and a.value == name for a in n.args)
        )
        assert any(
            kw.arg == "ignore_missing"
            and isinstance(kw.value, ast.Constant)
            and kw.value.value is True
            for kw in close.keywords
        ), f"{name}: close_device must use ignore_missing=True"
        ctor = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        )
        assert close.lineno < ctor.lineno, name


# ---------------------------------------------------------------------------
# A6 — structural: all code cells parse under IPython transform
# ---------------------------------------------------------------------------


def test_all_code_cells_parse_via_ipython_transform() -> None:
    from IPython.core.inputtransformer2 import TransformerManager

    tm = TransformerManager()
    cells = _code_cells()
    assert len(cells) > 200
    for source in cells:
        transformed = tm.transform_cell(source)
        ast.parse(transformed)
