from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

from .library import ModuleLibrary
from .metadict import MetaDict


class ExperimentManager:
    def __init__(self, exp_dir: str) -> None:
        self.exp_dir = Path(exp_dir)
        self._label: Optional[str] = None

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
        value: Optional[float] = None,
        clone_from: Optional[Union[Tuple[ModuleLibrary, MetaDict], str]] = None,
        label: Optional[str] = None,
        unit: Literal["A", "V", "K"] = "A",
    ) -> Tuple[ModuleLibrary, MetaDict]:
        if label is None:
            label = self._auto_label(value, unit)

        flx_dir = self.exp_dir / label
        if (flx_dir / "meta_info.json").exists():
            raise FileExistsError(
                f"Context '{label}' already exists. "
                "Use use_flux() to load it, or provide a different label."
            )

        self._label = label

        flx_dir.mkdir(parents=True, exist_ok=True)
        if clone_from is not None:
            if isinstance(clone_from, str):
                src_folder = Path(clone_from)
                ml = ModuleLibrary(src_folder / "module_cfg.yaml", read_only=True)
                md = MetaDict(src_folder / "meta_info.json", read_only=True)
            else:
                (ml, md) = clone_from
            ml = ml.clone(dst_path=flx_dir / "module_cfg.yaml")
            md = md.clone(dst_path=flx_dir / "meta_info.json")
        else:
            ml = ModuleLibrary(flx_dir / "module_cfg.yaml")
            md = MetaDict(flx_dir / "meta_info.json")

        return ml, md

    def use_flux(
        self, label: str, readonly: bool = False
    ) -> Tuple[ModuleLibrary, MetaDict]:
        flx_dir = self.exp_dir / label
        if not (flx_dir / "meta_info.json").exists():
            raise FileNotFoundError(
                f"Folder '{label}' not found. Available: {self.list_contexts()}"
            )

        self._label = label
        ml = ModuleLibrary(flx_dir / "module_cfg.yaml", read_only=readonly)
        md = MetaDict(flx_dir / "meta_info.json", read_only=readonly)

        return ml, md

    @property
    def label(self) -> str:
        if self._label is None:
            raise RuntimeError(
                "No active context. Call new_flux() or use_flux() first."
            )
        return self._label

    @property
    def flx_dir(self) -> Path:
        return self.exp_dir / self.label

    def _auto_label(
        self, value: Optional[float] = None, unit: Literal["A", "V", "K"] = "A"
    ) -> str:

        if value is not None:
            if unit == "A":
                if value <= 0.1:  # 100 mA
                    united_value = f"{value * 1e3:.3f}mA"  # convert to mA
                else:
                    united_value = f"{value:.3f}A"
            elif unit == "V":
                if value <= 0.1:  # 100 mV
                    united_value = f"{value * 1e3:.3f}mV"  # convert to mV
                else:
                    united_value = f"{value:.3f}V"
            elif unit == "K":
                if value <= 1.0:  # 1000 mK
                    united_value = f"{value * 1e3:.3f}mK"  # convert to mK
                else:
                    united_value = f"{value:.3f}K"
            else:
                raise ValueError(f"Invalid unit: {unit}")
        else:
            united_value = "NoValue"

        base_name = f"{datetime.now().strftime('%m%d%H')}_{united_value}"
        existing = set(self.list_contexts())
        if base_name not in existing:
            return base_name
        idx = 2
        while f"{base_name}_{idx}" in existing:
            idx += 1
        return f"{base_name}_{idx}"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(exp_dir={self.exp_dir}, active={None if self.label is None else self.label})"

    def __repr__(self) -> str:
        return self.__str__()
