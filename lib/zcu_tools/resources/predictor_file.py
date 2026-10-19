"""Typed predictor-file interface (implementation follows contract tests)."""

from dataclasses import dataclass
from pathlib import Path

from zcu_tools.simulate.parameters import ModelName, ModelParams


@dataclass(frozen=True)
class PredictorSource:
    """Authored run identifier and timezone-aware ISO timestamp; None means unknown."""

    run_id: str | None
    created_at: str | None


@dataclass(frozen=True)
class PredictorFile:
    """Model-tagged numeric inputs, native flux unit and authored source metadata."""

    model: ModelName
    params: ModelParams
    flux_unit: str | None
    source: PredictorSource


def load_predictor_file(path: Path) -> PredictorFile | None:
    """Load and validate a zcu.predictor v1 JSON file."""
    return None


def save_predictor_file(path: Path, predictor: PredictorFile) -> None:
    """Write a zcu.predictor v1 JSON file."""
