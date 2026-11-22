"""Manifest and selector errors stop the public CLI before opening a Run."""

import json
from pathlib import Path

import pytest

from tools.experiment_wall_time import main


def _write_scenes(directory: Path, *scenes: dict[str, object]) -> Path:
    path = directory / "case-inputs.json"
    path.write_text(
        json.dumps(
            {
                "entry": "demo",
                "point": "initial",
                "unpinned": {"form": "current Form", "point": "fixture point"},
                "scenes": scenes,
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    "value", ["NaN", "[Infinity]", '{"stop": 1e999}', '{"stop": 1, "stop": 2}']
)
def test_main_rejects_ambiguous_or_nonfinite_override_values_before_registration(
    tmp_path: Path, value: str
) -> None:
    path = _write_scenes(
        tmp_path,
        {
            "scene_id": "scan",
            "adapter": "fake/frequency",
            "basis": "keep scout",
            "overrides": [
                {"path": ["freq"], "value": "sentinel", "basis": "short scan"}
            ],
        },
    )
    path.write_text(path.read_text().replace('"sentinel"', value))
    output = tmp_path / "output"
    with pytest.raises(SystemExit) as raised:
        main(
            ["--output", str(output)],
            data_dir=tmp_path,
            discover=lambda: pytest.fail(
                "invalid manifest must fail before registration"
            ),
        )
    assert raised.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize(
    ("issue", "diagnostic"),
    [
        ("duplicate_scene", "Duplicate scene ID"),
        ("duplicate_override", "Duplicate override"),
        ("invalid_override", "overrides.0.path"),
        ("unknown_selector", "unknown scene ID or adapter family"),
        ("duplicate_selector", "duplicate selector"),
    ],
)
def test_invalid_manifest_or_selector_fails_before_registration(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    issue: str,
    diagnostic: str,
) -> None:
    scene: dict[str, object] = {
        "scene_id": "sweep",
        "adapter": "fake/frequency",
        "basis": "test",
    }
    override = {"path": ["wait"], "value": 2, "basis": "bound wait"}
    argv = ["--output", str(tmp_path / "output")]
    if issue == "unknown_selector":
        # It is an adapter key, but neither an authored ID nor a family.
        argv += ["--select", "fake/frequency"]
    elif issue == "duplicate_selector":
        argv += ["--select", "sweep", "--select", "sweep"]
    elif issue == "duplicate_override":
        scene["overrides"] = [override, override | {"value": 3}]
    elif issue == "invalid_override":
        scene["overrides"] = [override | {"path": ["tones", -1]}]
    _write_scenes(
        tmp_path, *([scene, scene] if issue == "duplicate_scene" else [scene])
    )

    with pytest.raises(SystemExit) as raised:
        main(
            argv,
            data_dir=tmp_path,
            discover=lambda: pytest.fail(
                "must reject malformed inputs before registration"
            ),
        )
    assert raised.value.code == 2
    assert diagnostic in capsys.readouterr().err
    assert not (tmp_path / "output").exists()
