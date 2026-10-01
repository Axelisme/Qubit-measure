"""Application-facing arbitrary waveform assets and their independent preview."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from zcu_tools.resources.waveform_assets import (
    ArbWaveformData,
    ArbWaveformInfo,
    FormulaRecipe,
)

type ArbWaveformWriteStatus = Literal["created", "overwritten"]


@dataclass(frozen=True)
class ArbWaveformPreviewResult:
    recipe: FormulaRecipe | None
    figure_path: str


class ArbWaveformKeys(Protocol):
    """Read capability used by cfg option sources."""

    def list_data_keys(self) -> list[str]: ...


class ArbWaveformPort(ArbWaveformKeys, Protocol):
    """Qubit-scoped asset operations; saving never requests a preview.

    Successful mutations advance the asset revision once. Preview failures do
    not undo saved assets or their revisions. Rename/delete do not rewrite refs.
    """

    def list_infos(self) -> list[ArbWaveformInfo]: ...

    def load_data(self, data_key: str) -> ArbWaveformData: ...

    def set_formula(
        self,
        data_key: str,
        recipe: FormulaRecipe | dict[str, object],
        *,
        overwrite: bool,
    ) -> ArbWaveformWriteStatus: ...

    def get_preview(self, data_key: str) -> ArbWaveformPreviewResult: ...

    def delete(self, data_key: str) -> None: ...

    def rename(self, old_data_key: str, new_data_key: str) -> None: ...
