"""Native save and restore with optional Data Variables."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from zcu_tools.datafile import load
from zcu_tools.experiment import restore_run, save_run

from tests.experiment._optional_variables_support import (
    diagnostic_record,
    diagnostic_spec,
)


@pytest.mark.parametrize("present", [False, True])
def test_optional_variable_round_trip_and_resave(
    tmp_path: Path, *, present: bool
) -> None:
    spec = diagnostic_spec()
    source = diagnostic_record(present=present)
    path = tmp_path / "data.h5"
    save_run(source, path, spec=spec)
    stored = load(path)
    keys = ("data", "diagnostics") if present else ("data",)
    assert set(stored.vars) == set(keys)
    restored = restore_run(stored, spec=spec)
    assert restored.cfg_issue is None and restored.record.cfg == source.cfg
    np.testing.assert_array_equal(restored.record.result.signal, [1 + 2j, 3 - 4j])
    if present:
        np.testing.assert_array_equal(
            restored.record.result.diagnostics, [[2.0, 10.0], [3.0, 16.0]]
        )
    else:
        assert restored.record.result.diagnostics is None
    save_run(restored.record, path, spec=spec, replace=True)
    assert set(load(path).vars) == set(keys)


def test_grouped_declaration_requires_at_least_one_required_variable() -> None:
    spec = diagnostic_spec()
    with pytest.raises(ValueError, match="at least one required"):
        replace(
            spec,
            variables=tuple(replace(var, required=False) for var in spec.variables),
        )


@pytest.mark.parametrize(
    "defect, message",
    [
        ("missing", r"missing variables.*data"),
        ("unit", r"/vars/diagnostics.*unit"),
        ("axes", r"/vars/diagnostics.*axes"),
        ("shape", r"/vars/diagnostics.*shape"),
        ("dtype", r"/vars/diagnostics.*imaginary component"),
    ],
)
def test_restore_rejects_missing_required_or_malformed_present_optional(
    tmp_path: Path, defect: str, message: str
) -> None:
    spec = diagnostic_spec()
    path = tmp_path / "data.h5"
    save_run(diagnostic_record(present=True), path, spec=spec)
    stored = load(path)
    variables = dict(stored.vars)
    data = variables["diagnostics"]
    if defect == "missing":
        del variables["data"]
    elif defect == "unit":
        variables["diagnostics"] = replace(data, unit="V")
    elif defect == "axes":
        variables["diagnostics"] = replace(data, axes=data.axes[:1])
    elif defect == "shape":
        variables["diagnostics"] = replace(data, values=data.values[:, :1])
    else:
        variables["diagnostics"] = replace(data, values=data.values + 1j)
    with pytest.raises(ValueError, match=message):
        restore_run(replace(stored, vars=variables), spec=spec)
