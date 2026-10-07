"""Point-local library binding through ResultEntry public capabilities."""

import numpy as np
import pytest
from zcu_tools.cfg_model import ConstWaveformCfg, PulseSpec, WaveformSamples
from zcu_tools.resources.entry import (
    ResultEntry,
)

from tests.program.v2.support.waveform_assets import InMemoryWaveformAsset


@pytest.fixture
def library_assets() -> InMemoryWaveformAsset:
    return InMemoryWaveformAsset(
        "present",
        WaveformSamples(
            time=np.array([0.0, 0.05]), idata=np.array([0.0, 1.0]), qdata=None
        ),
    )


def test_point_local_library_uses_entry_scope_without_reading_point_or_assets(
    entry: ResultEntry,
    library_assets: InMemoryWaveformAsset,
) -> None:
    entry.new_point("a")
    entry.new_point("b")
    source = entry.result_path / "points/a"
    (source / "point.yaml").write_text("invalid: [", encoding="utf-8")
    library = entry.module_library("a", assets=library_assets)
    assert library.source_path == (source / "module_cfg.yaml").resolve()
    assert library.snapshot().module_keys() == ()
    with library.edit() as draft:
        draft.set_module(
            "Q1.pulses.pi",
            PulseSpec(
                ch=0, nqz=1, freq=100.0, gain=0.25, waveform="Q1.waveforms.missing"
            ),
        )
        draft.set_waveform("Q1.waveforms.base", ConstWaveformCfg(length=0.05))
    reopened = entry.module_library("a", assets=library_assets)
    copied = reopened.snapshot().module_spec("Q1.pulses.pi")
    assert isinstance(copied, PulseSpec)
    assert copied.waveform == "Q1.waveforms.missing"
    assert (
        entry.module_library("b", assets=library_assets).snapshot().module_keys() == ()
    )
    assert (source / "point.yaml").read_text(encoding="utf-8") == "invalid: ["


@pytest.mark.parametrize("label", ["", ".", "..", "a/b", "/absolute"])
def test_module_library_rejects_unsafe_labels(
    entry: ResultEntry,
    library_assets: InMemoryWaveformAsset,
    label: str,
) -> None:
    with pytest.raises(ValueError, match="name"):
        entry.module_library(label, assets=library_assets)
    assert entry.list_points() == []


def test_module_library_does_not_create_missing_document(
    entry: ResultEntry,
    library_assets: InMemoryWaveformAsset,
) -> None:
    with pytest.raises(FileNotFoundError):
        entry.module_library("absent", assets=library_assets)
    assert not (entry.result_path / "points/absent").exists()
