"""Entry-local predictor selection and dispersive resonator seed policy."""

from __future__ import annotations

from pathlib import Path

from zcu_tools.gui.app.analysis_project.entry import AnalysisTarget, require_target
from zcu_tools.gui.app.dispersive.state import (
    DEFAULT_BARE_RF,
    DispersiveState,
    FluxoniumInputs,
)
from zcu_tools.gui.session.aliases import AliasRow, AliasTable
from zcu_tools.resources.dispersive_file import load_dispersive_file
from zcu_tools.resources.entry import RoleResolutionError, document_value, snapshot
from zcu_tools.resources.predictor_file import load_predictor_file
from zcu_tools.simulate.fluxonium.predict import FluxoniumParams


def _bare_rf_seed(target: AnalysisTarget) -> float:
    path = target.resource.dispersive_path / f"{target.component}.json"
    if path.exists():
        params = load_dispersive_file(path).params
        if params.bare_rf is not None:
            return params.bare_rf
        if params.plot_transitions.r_f is not None:
            return params.plot_transitions.r_f
    aliases = AliasTable(
        [AliasRow(k, v) for k, v in target.resource.load_aliases().aliases.items()]
    )
    document = snapshot(target.resource.setup)
    try:
        resonator = aliases.resolve("resonator", document)
    except RoleResolutionError as exc:
        # Only absent/unbound optional aliases fall through; malformed links fail.
        if exc.reason not in ("Unknown alias", "Alias is not bound"):
            raise
        return DEFAULT_BARE_RF
    component = document.components[resonator]
    if "freq" not in type(component).model_fields:
        return DEFAULT_BARE_RF
    freq = document_value(document, f"{resonator}.freq")
    if freq is None:
        return DEFAULT_BARE_RF
    if isinstance(freq, bool) or not isinstance(freq, (int, float)):
        raise ValueError(f"Resonator {resonator!r} freq must be MHz")
    return float(freq) * 1e-3


class ProjectService:
    """Read selected predictor files without changing setup or Database."""

    def __init__(self, state: DispersiveState) -> None:
        self._state = state

    def list_predictors(self) -> tuple[str, ...]:
        """List JSON filenames in the opened entry; no directory is created."""
        root = require_target(self._state.project).resource.predictors_path
        if not root.exists():
            return ()
        return tuple(
            sorted(path.name for path in root.glob("*.json") if path.is_file())
        )

    def load_fit_inputs(self, predictor: str) -> FluxoniumInputs:
        """Read an entry-local JSON filename, validate inputs, then publish.

        Reject absolute paths, directory traversal, non-JSON names and other
        models with ValueError. Missing/invalid/incomplete files propagate their
        errors without replacing state. Seed priority is dispersive bare_rf,
        plot_transitions.r_f, resonator alias freq (MHz to GHz), then 5 GHz.
        An existing tuning value is retained.
        """
        target = require_target(self._state.project)
        root = target.resource.predictors_path
        path = root / predictor
        if (
            Path(predictor).name != predictor
            or "/" in predictor
            or "\\" in predictor
            or path.suffix != ".json"
            or not path.resolve().is_relative_to(root.resolve())
        ):
            raise ValueError("predictor must be an entry-local JSON filename")
        loaded = load_predictor_file(path)
        if not isinstance(loaded.params, FluxoniumParams):
            raise ValueError("dispersive requires a fluxonium predictor")
        model = loaded.params.build_predictor()
        inputs = FluxoniumInputs(
            params=model.params,
            flux=loaded.flux,
            bare_rf_seed=_bare_rf_seed(target),
        )
        self._state.set_fit_inputs(inputs, predictor=predictor)
        return inputs
