from __future__ import annotations

from pathlib import Path
from tempfile import gettempdir
from typing import TYPE_CHECKING

from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.meta_tool import (
    ArbWaveformData,
    ArbWaveformDatabase,
    ArbWaveformInfo,
    FormulaRecipe,
    prepare_preview_series,
)

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.state import State
    from zcu_tools.gui.session.types import ExpContext


ARB_WAVEFORMS_VERSION_KEY = "arb_waveforms"


class ArbWaveformService:
    """App-local adapter for the shared arbitrary waveform repository."""

    def __init__(self, state: State) -> None:
        self._state = state

    def root_path(self) -> Path:
        root = resolve_arb_waveform_root(self._state.exp_context)
        root.mkdir(parents=True, exist_ok=True)
        return root

    def list_data_keys(self) -> list[str]:
        if not self._try_init_database():
            return []
        return ArbWaveformDatabase.list()

    def list_infos(self) -> list[ArbWaveformInfo]:
        if not self._try_init_database():
            return []
        return [
            ArbWaveformDatabase.inspect(data_key)
            for data_key in ArbWaveformDatabase.list()
        ]

    def load_data(self, data_key: str) -> ArbWaveformData:
        self._init_database()
        return ArbWaveformDatabase.load(data_key)

    def get_preview(self, data_key: str) -> dict[str, object]:
        self._init_database()
        data = ArbWaveformDatabase.load(data_key)
        return {
            "recipe": data.recipe.to_dict() if data.recipe is not None else None,
            "preview_figure": render_preview_png(data, data_key=data_key),
        }

    def set_formula(
        self,
        data_key: str,
        recipe: FormulaRecipe | dict[str, object],
        *,
        overwrite: bool,
    ) -> dict[str, object]:
        self._init_database()
        existed = ArbWaveformDatabase.exists(data_key)
        ArbWaveformDatabase.create_from_formula(data_key, recipe, overwrite=overwrite)
        self._state.version.bump(ARB_WAVEFORMS_VERSION_KEY)
        data = ArbWaveformDatabase.load(data_key)
        return {
            "status": "overwritten" if existed else "created",
            "preview_figure": render_preview_png(data, data_key=data_key),
        }

    def delete(self, data_key: str) -> None:
        self._init_database()
        ArbWaveformDatabase.delete(data_key)
        self._state.version.bump(ARB_WAVEFORMS_VERSION_KEY)

    def rename(self, old_data_key: str, new_data_key: str) -> None:
        self._init_database()
        ArbWaveformDatabase.rename(old_data_key, new_data_key)
        self._state.version.bump(ARB_WAVEFORMS_VERSION_KEY)

    def _init_database(self) -> None:
        ArbWaveformDatabase.init(self.root_path())

    def _try_init_database(self) -> bool:
        root = try_resolve_arb_waveform_root(self._state.exp_context)
        if root is None:
            return False
        root.mkdir(parents=True, exist_ok=True)
        ArbWaveformDatabase.init(root)
        return True


def resolve_arb_waveform_root(ctx: ExpContext) -> Path:
    """Resolve the qubit-scoped arbitrary waveform root from the active project."""

    root = try_resolve_arb_waveform_root(ctx)
    if root is None:
        raise FailedPreconditionError(
            "No project database_path is configured.", reason_code="no_project"
        )
    return root


def try_resolve_arb_waveform_root(ctx: ExpContext) -> Path | None:
    """Resolve the arbitrary waveform root, or ``None`` before project setup."""

    if not ctx.database_path:
        return None
    database_path = Path(ctx.database_path)
    chip = ctx.chip_name
    qub = ctx.qub_name
    parts = database_path.parts
    if chip and qub:
        for index in range(len(parts) - 1):
            if parts[index] == chip and parts[index + 1] == qub:
                return Path(*parts[: index + 2]) / "arb_waveforms"
    return database_path / "arb_waveforms"


def render_preview_png(
    data: ArbWaveformData, *, data_key: str, out_path: str | Path | None = None
) -> str:
    """Render static normalized I/Q/Abs preview PNG for agent and GUI adapters."""

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if out_path is None:
        out_path = Path(gettempdir()) / f"zcu_arb_waveform_{data_key}.png"
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig = Figure(figsize=(6.4, 4.0), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    # Agent/MCP PNG always uses normalized output; GUI dialog respects user toggle.
    series = prepare_preview_series(data, normalize=True)
    ax.plot(series.time, series.idata, label="I")
    ax.plot(series.time, series.qdata, label="Q")
    ax.plot(series.time, series.abs_data, label="Abs")
    ax.set_title(data_key)
    ax.set_xlabel("Time (us)")
    ax.set_ylabel("Normalized amplitude")
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlim(float(series.time[0]), float(series.time[-1]))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    return str(path)
