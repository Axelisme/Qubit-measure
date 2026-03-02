from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from zcu_tools.library import ModuleLibrary
from zcu_tools.table import MetaDict


class FluxManager:
    """
    Manages per-flux metadata and module configurations.

    Each flux context is a self-contained directory with its own
    ``module_cfg.yaml`` and ``meta_info.json``.  Use :meth:`new_flux` to
    start a fresh context and :meth:`use_flux` to resume an existing one.
    """

    def __init__(self, result_dir: str) -> None:
        self.result_dir = Path(result_dir)
        self._ml: Optional[ModuleLibrary] = None
        self._md: Optional[MetaDict] = None
        self._flx_dir: Optional[Path] = None
        self._label: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_contexts(self) -> List[str]:
        """Return sorted labels of all existing flux contexts."""
        if not self.result_dir.exists():
            return []
        return sorted(
            d.name
            for d in self.result_dir.iterdir()
            if d.is_dir() and (d / "meta_info.json").exists()
        )

    def new_flux(
        self,
        cur_A: float,
        label: Optional[str] = None,
    ) -> Tuple[ModuleLibrary, MetaDict]:
        """
        Create a brand-new flux context.

        Parameters
        ----------
        cur_A : float
            Flux current in Amperes.
        label : str, optional
            Explicit folder name.  When *None*, an auto-generated label of the
            form ``MMDD_X.XXXmA`` is used (with ``_2``, ``_3``, … suffixes to
            avoid collisions).

        Returns
        -------
        (ModuleLibrary, MetaDict)
        """
        if label is None:
            label = self._auto_label(cur_A)

        flx_dir = self.result_dir / label
        if (flx_dir / "meta_info.json").exists():
            raise FileExistsError(
                f"Context '{label}' already exists. "
                "Use use_flux() to load it, or provide a different label."
            )

        self._activate(label)
        assert self._md is not None
        self._md.cur_A = cur_A
        return self.ml, self.md

    def use_flux(self, label: str) -> Tuple[ModuleLibrary, MetaDict]:
        """
        Load an existing flux context to continue working.

        Parameters
        ----------
        label : str
            Folder name of the context (e.g. ``"0302_0.000mA"``).

        Returns
        -------
        (ModuleLibrary, MetaDict)
        """
        flx_dir = self.result_dir / label
        if not (flx_dir / "meta_info.json").exists():
            raise FileNotFoundError(
                f"Context '{label}' not found. "
                f"Available: {self.list_contexts()}"
            )

        self._activate(label)
        return self.ml, self.md

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ml(self) -> ModuleLibrary:
        if self._ml is None:
            raise RuntimeError("No active context. Call new_flux() or use_flux() first.")
        return self._ml

    @property
    def md(self) -> MetaDict:
        if self._md is None:
            raise RuntimeError("No active context. Call new_flux() or use_flux() first.")
        return self._md

    @property
    def flx_dir(self) -> str:
        if self._flx_dir is None:
            raise RuntimeError("No active context. Call new_flux() or use_flux() first.")
        return str(self._flx_dir)

    @property
    def label(self) -> str:
        if self._label is None:
            raise RuntimeError("No active context. Call new_flux() or use_flux() first.")
        return self._label

    @property
    def img_dir(self) -> str:
        """Convenience path for saving experiment images."""
        return str(Path(self.flx_dir) / "exp_image")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _activate(self, label: str) -> None:
        # persist any unsaved changes before switching
        if self._md is not None:
            self._md.sync()
        if self._ml is not None:
            self._ml.sync()

        self._label = label
        self._flx_dir = self.result_dir / label
        self._ml = ModuleLibrary(cfg_path=str(self._flx_dir / "module_cfg.yaml"))
        self._md = MetaDict(str(self._flx_dir / "meta_info.json"))

    def _auto_label(self, cur_A: float) -> str:
        date_prefix = datetime.now().strftime("%m%d")
        base = f"{date_prefix}_{cur_A * 1e3:.3f}mA"
        existing = set(self.list_contexts())
        if base not in existing:
            return base
        idx = 2
        while f"{base}_{idx}" in existing:
            idx += 1
        return f"{base}_{idx}"

    def __repr__(self) -> str:
        ctx = self._label or "(none)"
        return f"FluxManager(result_dir='{self.result_dir}', active='{ctx}')"
