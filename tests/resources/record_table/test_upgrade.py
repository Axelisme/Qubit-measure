"""Operator-declared upgrades through the complete-history storage seam."""

import json
from pathlib import Path

import pytest
from zcu_tools.resources.record_table.upgrade import upgrade_record_files


def test_mixed_history_dry_run_apply_backup_and_idempotence(tmp_path: Path) -> None:
    path = tmp_path / "Q1.jsonl"
    original = (
        b'{"run_id":"same","dev_value":0.25,"dev_unit":" mV ","flux_int":0.1,"flux_period":0.8,"extra":{"a":[null,true]}}\r\n'
        b'{ "run_id": "other", "devices": {"bias": {"value": 3, "value_unit": "V", "tag": "keep"}}, "component": "Q2" }\n'
        b'{"run_id":"same","dev_value":7,"dev_unit":"native","schema_version":1}\n'
        b'{"run_id":"opaque","payload":[1,2]}\n'
    )
    path.write_bytes(original)
    (dry,) = upgrade_record_files((path,), device_name="bias")
    assert dry.status == "dry-run"
    assert dry.device_name == "bias"
    assert [(row.line_number, row.classification) for row in dry.rows] == [
        (1, "old"),
        (2, "new"),
        (3, "old"),
        (4, "new"),
    ]
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
    (applied,) = upgrade_record_files((path,), device_name="bias", apply=True)
    assert applied.status == "applied"
    assert applied.backup_path is not None
    assert applied.backup_path.read_bytes() == original
    assert [json.loads(line) for line in path.read_bytes().splitlines()] == [
        {
            "run_id": "same",
            "devices": {"bias": {"value": 0.25, "value_unit": " mV "}},
            "flux_int": 0.1,
            "flux_period": 0.8,
            "extra": {"a": [None, True]},
        },
        {
            "run_id": "other",
            "devices": {"bias": {"value": 3, "value_unit": "V", "tag": "keep"}},
            "component": "Q2",
        },
        {
            "run_id": "same",
            "devices": {"bias": {"value": 7, "value_unit": "native"}},
            "schema_version": 1,
        },
        {"run_id": "opaque", "payload": [1, 2]},
    ]
    assert (
        path.read_bytes().splitlines(keepends=True)[1]
        == original.splitlines(keepends=True)[1]
    )
    after = path.read_bytes()
    files = set(tmp_path.iterdir())
    (rerun,) = upgrade_record_files((path,), device_name="bias", apply=True)
    assert rerun.status == "skipped"
    assert rerun.backup_path is None
    assert path.read_bytes() == after
    assert set(tmp_path.iterdir()) == files


@pytest.mark.parametrize(
    ("bad_line", "classification", "diagnostic"),
    [
        (b'{"run_id":"x","dev_value":1}\n', "partial", "dev_unit"),
        (b'{"run_id":"x","dev_unit":"V"}\n', "partial", "dev_value"),
        (
            b'{"run_id":"x","dev_value":1,"dev_unit":"V","devices":{"bias":{"value":1}}}\n',
            "partial",
            "conflict",
        ),
        (
            b'{"run_id":"x","dev_value":1,"dev_unit":"V","devices":{"bias":{"value_unit":"V"}}}\n',
            "partial",
            "conflict",
        ),
        (b'{"run_id":"x","devices":{"bias":{"value_unit":"V"}}}\n', "partial", "value"),
        (b'{"run_id":"x","devices":{"bias":{"value":1}}}\n', "partial", "value_unit"),
        (b'{"run_id":"x","dev_value":true,"dev_unit":"V"}\n', "corrupt", "dev_value"),
        (b'{"run_id":"x","dev_value":NaN,"dev_unit":"V"}\n', "corrupt", "dev_value"),
        (
            b'{"run_id":"x","devices":{"bias":{"value":Infinity,"value_unit":"V"}}}\n',
            "corrupt",
            "value",
        ),
        (
            b'{"run_id":"x","devices":{"bias":{"value":1,"value_unit":" "}}}\n',
            "corrupt",
            "value_unit",
        ),
        (b'{"run_id":"x","dev_value":"1","dev_unit":"V"}\n', "corrupt", "dev_value"),
        (b'{"run_id":"x","dev_value":1,"dev_unit":null}\n', "corrupt", "dev_unit"),
        (b'{"run_id":"x","dev_value":1,"dev_unit":""}\n', "corrupt", "dev_unit"),
        (b'{"run_id":"x","devices":[]}\n', "corrupt", "devices"),
        (b'{"run_id":"x","devices":{"bias":null}}\n', "corrupt", "bias"),
        (b'{"dev_value":1,"dev_unit":"V"}\n', "corrupt", "run_id"),
        (b'{"run_id":""}\n', "corrupt", "run_id"),
        (b'{"run_id":7}\n', "corrupt", "run_id"),
        (b"[]\n", "corrupt", "object"),
        (b"{broken}\n", "corrupt", "JSON"),
        (b'{"run_id":"tail"}', "corrupt", "LF"),
        (b"\xff\n", "corrupt", "JSON"),
    ],
)
def test_partial_or_corrupt_line_rejects_whole_file_but_continues_siblings(
    tmp_path: Path, bad_line: bytes, classification: str, diagnostic: str
) -> None:
    bad, good = tmp_path / "bad.jsonl", tmp_path / "good.jsonl"
    old = b'{"run_id":"ok","dev_value":2,"dev_unit":"V"}\n'
    original = old + bad_line
    bad.write_bytes(original)
    good.write_bytes(old)
    rejected, applied = upgrade_record_files(
        (bad, good), device_name="bias", apply=True
    )
    assert rejected.status == "rejected"
    assert rejected.backup_path is None
    assert [(row.line_number, row.classification) for row in rejected.rows] == [
        (1, "old"),
        (2, classification),
    ]
    assert diagnostic in (rejected.rows[1].error or "")
    assert bad.read_bytes() == original
    assert list(tmp_path.glob("bad.jsonl.*.bak")) == []
    assert applied.status == "applied"
    assert json.loads(good.read_bytes()) == {
        "run_id": "ok",
        "devices": {"bias": {"value": 2, "value_unit": "V"}},
    }


def test_metadata_merge_preserves_other_devices_and_all_diagnostics(
    tmp_path: Path,
) -> None:
    path = tmp_path / "records.jsonl"
    original = (
        '{"run_id":"a","dev_value":-3,"dev_unit":" arb. ","devices":'
        '{" 偏壓 ":{"calibration":{"who":"operator"}},"other":{"value":NaN},"unused":null}}\n'
        '{"run_id":"b","devices":{"other":{"value_unit":"V"}}}\n'
    ).encode()
    path.write_bytes(original)
    (result,) = upgrade_record_files((path,), device_name=" 偏壓 ", apply=True)
    assert result.status == "applied"
    lines = path.read_bytes().splitlines(keepends=True)
    assert json.loads(lines[0], parse_constant=str) == {
        "run_id": "a",
        "devices": {
            " 偏壓 ": {
                "calibration": {"who": "operator"},
                "value": -3,
                "value_unit": " arb. ",
            },
            "other": {"value": "NaN"},
            "unused": None,
        },
    }
    assert lines[1] == original.splitlines(keepends=True)[1]
    path.write_bytes(
        b'{"run_id":"a","dev_value":1}\n'
        b'{"run_id":"b","dev_value":1,"dev_unit":"V","devices":{"bias":{"value":1,"value_unit":"V"}}}\n'
        b'{"run_id":"c","devices":{"bias":{"value":false,"value_unit":"V"}}}\n'
    )
    (rejected,) = upgrade_record_files((path,), device_name="bias", apply=True)
    assert [(row.line_number, row.classification) for row in rejected.rows] == [
        (1, "partial"),
        (2, "partial"),
        (3, "corrupt"),
    ]


@pytest.mark.parametrize("name", ["", " ", "\t\n", "bias.dot", "."])
def test_invalid_operator_name_fails_before_io(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match="name"):
        upgrade_record_files(
            (tmp_path / "missing.jsonl",), device_name=name, apply=True
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("fail_replace", [False, True])
def test_backup_precedes_atomic_replace_and_survives_publication_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_replace: bool
) -> None:
    path = tmp_path / "record.jsonl"
    original = b'{"run_id":"a","dev_value":1,"dev_unit":"V"}\n'
    path.write_bytes(original)
    replace = Path.replace

    def observed_replace(temporary: Path, destination: Path) -> Path:
        (backup,) = tmp_path.glob("record.jsonl.*.bak")
        assert backup.read_bytes() == original
        assert destination.read_bytes() == original
        if fail_replace:
            raise OSError("publication unavailable")
        return replace(temporary, destination)

    monkeypatch.setattr(Path, "replace", observed_replace)
    (result,) = upgrade_record_files((path,), device_name="bias", apply=True)
    assert result.status == ("rejected" if fail_replace else "applied")
    assert result.backup_path is not None
    assert result.backup_path.read_bytes() == original
    if fail_replace:
        assert result.error == "publication unavailable"
        assert path.read_bytes() == original
    assert set(tmp_path.iterdir()) == {path, result.backup_path, Path(f"{path}.lock")}


def test_unresolvable_source_rejects_without_stopping_later_files(
    tmp_path: Path,
) -> None:
    loop, good = tmp_path / "loop", tmp_path / "good.jsonl"
    loop.symlink_to(loop.name)
    good.write_bytes(b'{"run_id":"a","dev_value":1,"dev_unit":"V"}\n')
    rejected, applied = upgrade_record_files(
        (loop, good), device_name="bias", apply=True
    )
    assert rejected.status == "rejected"
    assert rejected.error is not None
    assert applied.status == "applied"
