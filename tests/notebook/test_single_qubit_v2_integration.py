"""Executable-interface probes for the single_qubit notebook v2 integration.

Reads ``notebook_md/single_qubit.md`` with jupytext, extracts the relevant
cells and validates the A1-A5 behaviors with fakes — no hardware, no live
VISA, no result data.
"""

from __future__ import annotations

import ast
from pathlib import Path

import jupytext
import numpy as np
import pandas as pd
import pytest
from zcu_tools.resources.sample_table import (
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


# ---------------------------------------------------------------------------
# A1 — Lookback
# ---------------------------------------------------------------------------


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
