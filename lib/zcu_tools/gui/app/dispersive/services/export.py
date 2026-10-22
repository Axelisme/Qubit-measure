"""Export dispersive fits to the entry artifact and declared setup fields."""

from __future__ import annotations

from datetime import datetime, timezone

from zcu_tools.gui.app.analysis_project.entry import require_target
from zcu_tools.gui.app.dispersive.state import DispersiveState
from zcu_tools.resources.analysis_source import AnalysisSource
from zcu_tools.resources.dispersive_file import (
    DispersiveFile,
    DispersiveParams,
    load_dispersive_file,
    save_dispersive_file,
)
from zcu_tools.resources.entry import snapshot


class ExportService:
    """Write accepted GHz values without adding undeclared component fields."""

    def __init__(self, state: DispersiveState) -> None:
        self._state = state

    def export_params(self) -> str:
        """Write dispersive/<component>.json; update only declared g/bare_rf.

        Reject no entry or no accepted result before writing. Preserve authored
        transition annotations from an existing artifact. Source run is unknown;
        created_at is this export's UTC time. Setup edits and the artifact are
        separate commits; filesystem/schema failures propagate without rollback.
        Database is never saved.
        """
        target = require_target(self._state.project)
        fit = self._state.disp_fit
        if fit.g is None or fit.bare_rf is None:
            raise RuntimeError("no dispersive fit result to export (fit g first)")
        path = target.resource.dispersive_path / f"{target.component}.json"
        params = (
            load_dispersive_file(path).params if path.exists() else DispersiveParams()
        )
        document = DispersiveFile(
            format="zcu.dispersive",
            version=1,
            params=params.model_copy(update={"g": fit.g, "bare_rf": fit.bare_rf}),
            source=AnalysisSource(
                run_id=None, created_at=datetime.now(timezone.utc).isoformat()
            ),
        )
        target.resource.dispersive_path.mkdir(parents=True, exist_ok=True)
        save_dispersive_file(path, document)
        component = snapshot(target.resource.setup).components[target.component]
        updates = {
            field: value
            for field, value in (("g", fit.g), ("bare_rf", fit.bare_rf))
            if field in type(component).model_fields
        }
        if updates:
            with target.resource.setup.edit() as draft:
                for field, value in updates.items():
                    draft.set(f"{target.component}.{field}", value)
        return str(path)
