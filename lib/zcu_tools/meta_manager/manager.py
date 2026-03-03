from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

from .library import ModuleLibrary
from .metadict import MetaDict


class ExperimentManager:
    def __init__(self, exp_dir: str) -> None:
        self.exp_dir = Path(exp_dir)
        self._flx_dir: Optional[Path] = None

    def list_contexts(self) -> List[str]:
        """Return sorted labels of all existing experiment contexts."""
        if not self.exp_dir.exists():
            return []
        return sorted(
            d.name
            for d in self.exp_dir.iterdir()
            if d.is_dir() and (d / "meta_info.json").exists()
        )

    def new_flux(
        self,
        cur: float,
        clone_from: Optional[Union[Tuple[ModuleLibrary, MetaDict], str]] = None,
        label: Optional[str] = None,
        unit: Literal["mA", "V"] = "mA",
    ) -> Tuple[ModuleLibrary, MetaDict]:
        if label is None:
            label = self._auto_label(cur, unit)

        flx_dir = self.exp_dir / label
        if (flx_dir / "meta_info.json").exists():
            raise FileExistsError(
                f"Context '{label}' already exists. "
                "Use use_flux() to load it, or provide a different label."
            )

        flx_dir.mkdir(parents=True, exist_ok=True)
        if clone_from is not None:
            if isinstance(clone_from, str):
                ml = ModuleLibrary(
                    str(Path(clone_from) / "module_cfg.yaml"), read_only=True
                )
                md = MetaDict(str(Path(clone_from) / "meta_info.json"), read_only=True)
            else:
                (ml, md) = clone_from
            ml = ml.clone(dst_path=str(flx_dir / "module_cfg.yaml"))
            md = md.clone(dst_path=str(flx_dir / "meta_info.json"))
        else:
            ml = ModuleLibrary(str(flx_dir / "module_cfg.yaml"))
            md = MetaDict(str(flx_dir / "meta_info.json"))

        return ml, md

    def use_flux(
        self, label: str, readonly: bool = False
    ) -> Tuple[ModuleLibrary, MetaDict]:
        flx_dir = self.exp_dir / label
        if not (flx_dir / "meta_info.json").exists():
            raise FileNotFoundError(
                f"Folder '{label}' not found. Available: {self.list_contexts()}"
            )

        ml = ModuleLibrary(str(flx_dir / "module_cfg.yaml"), read_only=readonly)
        md = MetaDict(str(flx_dir / "meta_info.json"), read_only=readonly)

        return ml, md

    @property
    def flx_dir(self) -> Path:
        if self._flx_dir is None:
            raise RuntimeError(
                "No active context. Call new_flux() or use_flux() first."
            )
        return self._flx_dir

    def _auto_label(self, cur: float, unit: Literal["mA", "V"]) -> str:
        date_prefix = datetime.now().strftime("%m%d")
        flx_value = f"{cur * 1e3:.3f}mA" if unit == "mA" else f"{cur:.3f}V"
        base = f"{date_prefix}_{flx_value}"
        existing = set(self.list_contexts())
        if base not in existing:
            return base
        idx = 2
        while f"{base}_{idx}" in existing:
            idx += 1
        return f"{base}_{idx}"
