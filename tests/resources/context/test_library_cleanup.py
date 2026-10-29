"""Read-only disk admission through the public point-local library handle."""

import json
from pathlib import Path

import pytest
from zcu_tools.cfg_model import ConstWaveformCfg
from zcu_tools.resources.context.library import ModuleLibrary

HEADER = "format: zcu.module-library\nformat_version: '2.0'\n"
GOOD = {"style": "const", "length": 0.1}


def test_legacy_sources_do_not_reject_specs_or_change_disk(tmp_path: Path) -> None:
    path = tmp_path / "library.yaml"
    path.write_text(
        HEADER
        + "components: {Q1: {waveforms: {pi: {style: const, length: 0.2}}}}\n"
        + "provenance: {Q1.waveforms.pi.length: {source: missing-event, kind: fit, "
        + "run_id: null, at: '2026-10-28T00:00:00Z', stderr: null}}\n"
    )
    original = path.read_bytes()
    library = ModuleLibrary(path)
    assert library.snapshot().waveform_keys() == ("Q1.waveforms.pi",)
    assert library.snapshot().get_waveform("Q1.waveforms.pi").length == 0.2
    assert path.read_bytes() == original
    assert list(path.parent.glob("*.bak*")) == []


@pytest.mark.parametrize("partition", ["pulses", "readout", "reset", "waveforms"])
@pytest.mark.parametrize(
    "bad",
    [None, 12, {}, {"type": "unknown"}, {"style": "const", "length": 0.1, "mask": 3}],
)
def test_bad_specs_are_skipped_without_changing_disk(
    tmp_path: Path, partition: str, bad: object
) -> None:
    path = tmp_path / "library.yaml"
    partitions: dict[str, dict[str, object]] = {"waveforms": {"pi_half": GOOD}}
    partitions.setdefault(partition, {})["bad"] = bad
    path.write_text(
        HEADER
        + "# original µ comment\ncomponents: "
        + json.dumps({"Q1": partitions})
        + "\n"
    )
    original = path.read_bytes()
    library = ModuleLibrary(path)
    assert library.snapshot().waveform_keys() == ("Q1.waveforms.pi_half",)
    assert library.snapshot().module_keys() == ()
    assert path.read_bytes() == original
    assert list(path.parent.glob("*.bak*")) == []
    assert library.snapshot().invalid_entries()[f"Q1.{partition}.bad"]
    assert not library.refresh()
    assert path.read_bytes() == original


@pytest.mark.parametrize("legacy", ["[]", "{orphan: {}}", "null"])
def test_retired_source_contents_are_not_interpreted(
    tmp_path: Path, legacy: str
) -> None:
    path = tmp_path / "library.yaml"
    path.write_text(HEADER + f"provenance: {legacy}\n")
    original = path.read_bytes()
    library = ModuleLibrary(path)
    assert library.snapshot().module_keys() == ()
    with library.edit():
        pass
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "content",
    [
        "components: []",
        "components: {Q1: null}",
        "components: {Q1: {pulses: 3}}",
        "components: {'Q.1': {}}",
        "components: {Q1: {pulses: {'a.b': {}}}}",
    ],
)
def test_container_errors_do_not_change_disk(tmp_path: Path, content: str) -> None:
    path = tmp_path / "library.yaml"
    path.write_text(HEADER + content + "\n")
    original = path.read_bytes()
    with pytest.raises(ValueError, match="components|[Ii]nvalid|library"):
        ModuleLibrary(path)
    assert path.read_bytes() == original


def test_shared_yaml_partition_reports_each_invalid_root(tmp_path: Path) -> None:
    path = tmp_path / "library.yaml"
    path.write_text(
        HEADER + "components:\n"
        "  Q1:\n    waveforms: &shared {bad: null, good: {style: const, length: 0.1}}\n"
        "  Q2:\n    waveforms: *shared\n"
    )
    original = path.read_bytes()
    snapshot = ModuleLibrary(path).snapshot()
    assert snapshot.waveform_keys() == ("Q1.waveforms.good", "Q2.waveforms.good")
    assert set(snapshot.invalid_entries()) == {"Q1.waveforms.bad", "Q2.waveforms.bad"}
    assert path.read_bytes() == original


def test_nonempty_edit_drops_legacy_sources_and_preserves_future_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "library.yaml"
    path.write_text(
        "format: zcu.module-library\nformat_version: '2.9'\nfuture: {x: 1}\n"
        "components: {Q1: {future_partition: {x: 2}, waveforms: {pi: {style: const, length: 0.1}}}}\n"
        "provenance: {obsolete: {source: absent}}\n"
    )
    library = ModuleLibrary(path)
    with library.edit() as draft:
        draft.set_waveform("Q1.waveforms.pi", ConstWaveformCfg(length=0.2))
    from ruamel.yaml import YAML

    stored = YAML(typ="safe").load(path.read_text())
    assert stored["future"] == {"x": 1}
    assert stored["components"]["Q1"]["future_partition"] == {"x": 2}
    assert stored["components"]["Q1"]["waveforms"]["pi"]["length"] == 0.2
    assert "provenance" not in stored
