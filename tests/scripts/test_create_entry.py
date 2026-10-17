"""Entry materialization uses public resources capabilities, without lab imports."""

from collections.abc import Generator
from pathlib import Path

import pytest
from pydantic import ConfigDict
from ruamel.yaml import YAML
from zcu_tools.resources.entry import (
    ComponentSchema,
    DestinationExistsError,
    ResultEntry,
    component_registry,
    snapshot,
)


class Meter(ComponentSchema):
    model_config = ConfigDict(extra="forbid")
    reading: float


@pytest.fixture
def template_path(tmp_path: Path) -> Generator[Path]:
    kind = "synthetic/meter"
    component_registry.register(kind, Meter)
    try:
        path = tmp_path / "template.yaml"
        YAML().dump(
            {
                "format": "zcu.entry-template",
                "format_version": "1.0",
                "components": {"M1": {"kind": kind, "reading": 7.5}},
                "aliases": {
                    "format": "zcu.aliases",
                    "format_version": "1.0",
                    "aliases": {
                        "meter": "M1",
                        "spare": "",
                        "reading": "$meter.reading",
                    },
                },
            },
            path,
        )
        yield path
    finally:
        component_registry.unregister(kind)


def test_materialized_entry_opens_with_setup_aliases_and_first_point(
    tmp_path: Path, template_path: Path
) -> None:
    from scripts.create_entry import materialize_entry

    materialize_entry(
        template_path, "demo", project_root=tmp_path, point_label="initial"
    )
    entry = ResultEntry.open(
        "demo", result_root=tmp_path / "result", database_root=tmp_path / "Database"
    )

    assert snapshot(entry.setup).components["M1"].model_dump()["reading"] == 7.5
    assert entry.list_points() == ["initial"]
    point = entry.use_point("initial")
    assert snapshot(point).components["M1"].model_dump()["reading"] == 7.5
    aliases = entry.load_aliases()
    assert aliases is not None
    assert list(aliases.aliases.items()) == [
        ("meter", "M1"),
        ("spare", ""),
        ("reading", "$meter.reading"),
    ]
    entry.setup.M1.reading = 8.0
    assert point.M1.reading == 7.5


@pytest.mark.parametrize("existing_root", ["result", "Database"])
def test_existing_destination_is_never_overwritten(
    tmp_path: Path, template_path: Path, existing_root: str
) -> None:
    from scripts.create_entry import materialize_entry

    existing = tmp_path / existing_root / "demo"
    existing.mkdir(parents=True)
    marker = existing / "keep.txt"
    marker.write_text("untouched")
    with pytest.raises(DestinationExistsError):
        materialize_entry(
            template_path, "demo", project_root=tmp_path, point_label="initial"
        )
    assert marker.read_text() == "untouched"
    other = "Database" if existing_root == "result" else "result"
    assert not (tmp_path / other / "demo").exists()
