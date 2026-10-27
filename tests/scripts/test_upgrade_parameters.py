"""Entry upgrade CLI through its same-process main seam, with synthetic schemas."""

import json
from pathlib import Path

import pytest
from zcu_tools.resources.parameter_schema import ParameterSchemaPolicy
from zcu_tools.resources.parameter_upgrade import ParameterDocument, UpgradePlan
from zcu_tools.resources.predictor_file import load_predictor_file

from scripts.upgrade_parameters import main


@pytest.fixture
def plan() -> UpgradePlan:
    def validate(document: ParameterDocument, version: int) -> None:
        if document.data != {"schema_version": version}:
            raise ValueError("expected synthetic current component")

    return UpgradePlan(
        ParameterSchemaPolicy(1, ("upgrade",)), (), validate, lambda _: False
    )


@pytest.fixture
def entry(tmp_path: Path) -> Path:
    root = tmp_path / "entry with spaces"
    root.mkdir()
    (root / "setup.yaml").write_text("schema_version: 1\n", encoding="utf-8")
    predictors = root / "predictors"
    predictors.mkdir()
    (predictors / "qubit.json").write_text(
        json.dumps(
            {
                "format": "zcu.predictor",
                "version": 1,
                "model": "fluxonium",
                "params": {
                    "EJ": 5,
                    "EC": 1,
                    "EL": 0.5,
                    "flux_half": 0.3,
                    "flux_period": -0.4,
                    "flux_bias": 0.1,
                },
                "flux_unit": "mA",
                "source": {"run_id": None, "created_at": None},
            }
        ),
        encoding="utf-8",
    )
    return root


def test_entry_preview_reports_predictor_without_writing(
    entry: Path, plan: UpgradePlan, capsys: pytest.CaptureFixture[str]
) -> None:
    path = entry / "predictors" / "qubit.json"
    original = path.read_bytes()
    assert main([str(entry)], plan=plan) == 0
    report = capsys.readouterr().out
    assert f"{path}: dry-run" in report
    assert "unnamed_device" in report
    assert "offset" in report
    assert path.read_bytes() == original
    assert sorted(p.name for p in path.parent.iterdir()) == ["qubit.json"]


@pytest.mark.parametrize("apply", [False, True])
def test_rejected_predictor_does_not_stop_valid_sibling(
    entry: Path, plan: UpgradePlan, capsys: pytest.CaptureFixture[str], apply: bool
) -> None:
    invalid = entry / "predictors" / "bad.json"
    original = b'{"format":"zcu.predictor","version":99}'
    invalid.write_bytes(original)
    args = [str(entry), "--apply"] if apply else [str(entry)]
    assert main(args, plan=plan) == 1
    report = capsys.readouterr().out
    assert f"{invalid}: rejected" in report
    assert (
        f"{entry / 'predictors' / 'qubit.json'}: {'applied' if apply else 'dry-run'}"
        in report
    )
    assert invalid.read_bytes() == original
    assert not list(invalid.parent.glob("bad.json.*.bak"))


def test_entry_apply_backs_up_original_and_skips_current_predictor(
    entry: Path, plan: UpgradePlan, capsys: pytest.CaptureFixture[str]
) -> None:
    path = entry / "predictors" / "qubit.json"
    original = path.read_bytes()
    assert main([str(entry), "--apply"], plan=plan) == 0
    assert f"{path}: applied" in capsys.readouterr().out
    assert load_predictor_file(path).flux.offset == pytest.approx(1.0)
    backups = list(path.parent.glob("*.bak"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == original
    current = path.read_bytes()
    assert main([str(entry), "--apply"], plan=plan) == 0
    assert f"{path}: skipped" in capsys.readouterr().out
    assert path.read_bytes() == current
    assert list(path.parent.glob("*.bak")) == backups
