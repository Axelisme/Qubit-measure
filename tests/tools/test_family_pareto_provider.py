"""Injected numeric workload providers; no concrete lab imports."""

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from zcu_tools.simulate.evolution import ModelSource, TimelineBatch

from tests.simulate.evolution.contract.accuracy_cases import accuracy_case
from tools.family_pareto import load_provider, main
from tools.family_pareto_inputs import Input


@dataclass(frozen=True)
class SyntheticWorkload:
    """Minimal external provider record implementing the consumer's contract."""

    name: str
    source: ModelSource
    batch: TimelineBatch
    metadata: dict[str, object]
    manifest: dict[str, object]
    domain_spans: dict[str, int]
    family_domains: dict[str, tuple[str, ...]]
    companions: dict[str, tuple[str, ...]]


def synthetic_workload() -> SyntheticWorkload:
    """Return a small, deterministic idle input with complete provenance."""
    case = accuracy_case("idle_coherence")
    return SyntheticWorkload(
        "synthetic_provider",
        case.source,
        case.batch,
        {"reason": "synthetic injection"},
        {"points": len(case.batch.points)},
        {"idle": len(case.batch.points)},
        {"t4": ("idle",)},
        {},
    )


def incomplete_workload() -> object:
    """An external provider that omits the numeric workload contract."""
    return {"name": "not_a_workload"}


def test_provider_rejects_an_incomplete_record() -> None:
    with pytest.raises(TypeError, match="Workload protocol"):
        load_provider(f"{__name__}:incomplete_workload")


def test_cli_freezes_injected_provider_and_its_declarations(tmp_path: Path) -> None:
    output = tmp_path / "manifest.json"
    args = [
        "--provider",
        f"{__name__}:synthetic_workload",
        "--freeze-inputs",
        str(output),
    ]
    assert main(args, inputs=[]) == 0
    first = output.read_bytes()
    assert main(args, inputs=[]) == 0
    assert output.read_bytes() == first
    records = json.loads(first)["inputs"]
    assert len(records) == 1
    assert records[0]["name"] == "synthetic_provider"
    assert records[0]["domain_spans"] == {"idle": 2}
    assert records[0]["manifest"] == {"points": 2}
    assert records[0]["metadata"] == {"reason": "synthetic injection"}


def test_cli_can_keep_synthetic_inventory_alongside_real_providers(
    tmp_path: Path,
) -> None:
    output = tmp_path / "combined.json"
    assert (
        main(
            [
                "--provider",
                f"{__name__}:synthetic_workload",
                "--include-synthetic",
                "--freeze-inputs",
                str(output),
            ],
            inputs=[
                Input("original", lambda: accuracy_case("idle_coherence"), "existing")
            ],
        )
        == 0
    )
    records = json.loads(output.read_text())["inputs"]
    assert [record["name"] for record in records] == ["original", "synthetic_provider"]
