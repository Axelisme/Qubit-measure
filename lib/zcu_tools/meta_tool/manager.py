from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from .library import ModuleLibrary
from .metadict import MetaDict


class ExperimentManager:
    def __init__(self, exp_dir: str | Path) -> None:
        self.exp_dir = Path(exp_dir).resolve()
        self._label: str | None = None

    def list_contexts(self) -> list[str]:
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
        value: float | None = None,
        clone_from: tuple[ModuleLibrary, MetaDict] | str | None = None,
        label: str | None = None,
        unit: Literal["A", "V", "K", "none"] = "none",
    ) -> tuple[ModuleLibrary, MetaDict]:
        if label is None:
            label = self._auto_label(value, unit)

        flux_dir = self.exp_dir / label
        if (flux_dir / "meta_info.json").exists():
            raise FileExistsError(
                f"Context '{label}' already exists. "
                "Use use_flux() to load it, or provide a different label."
            )

        self._label = label

        flux_dir.mkdir(parents=True, exist_ok=True)
        if clone_from is not None:
            if isinstance(clone_from, str):
                src_folder = self.exp_dir / clone_from
                if not src_folder.is_dir():
                    raise FileNotFoundError(
                        f"Source context '{clone_from}' not found. Available: {self.list_contexts()}"
                    )
                ml = ModuleLibrary(src_folder / "module_cfg.yaml", readonly=True)
                md = MetaDict(src_folder / "meta_info.json", readonly=True)
            else:
                (ml, md) = clone_from
            ml = ml.clone(dst_path=flux_dir / "module_cfg.yaml")
            md = md.clone(dst_path=flux_dir / "meta_info.json")
        else:
            ml = ModuleLibrary(flux_dir / "module_cfg.yaml")
            md = MetaDict(flux_dir / "meta_info.json")

        return ml, md

    def use_flux(
        self, label: str, readonly: bool = False
    ) -> tuple[ModuleLibrary, MetaDict]:
        flux_dir = self.exp_dir / label
        if not (flux_dir / "meta_info.json").exists():
            raise FileNotFoundError(
                f"Folder '{label}' not found. Available: {self.list_contexts()}"
            )

        self._label = label

        ml = ModuleLibrary(flux_dir / "module_cfg.yaml", readonly=readonly)
        md = MetaDict(flux_dir / "meta_info.json", readonly=readonly)

        return ml, md

    @property
    def label(self) -> str:
        if self._label is None:
            raise RuntimeError(
                "No active context. Call new_flux() or use_flux() first."
            )
        return self._label

    @property
    def current_label(self) -> str | None:
        return self._label

    @property
    def flux_dir(self) -> Path:
        return self.exp_dir / self.label

    def _auto_label(
        self,
        value: float | None = None,
        unit: Literal["A", "V", "K", "none"] = "none",
    ) -> str:

        UNIT_RANGES = {
            "A": [
                ("mA", 100e-3, 1e3),  # 100 mA
                ("A", float("inf"), 1),
            ],
            "V": [
                ("mV", 100e-3, 1e3),  # 100 mV
                ("V", float("inf"), 1),
            ],
            "K": [
                ("mK", 1000e-3, 1e3),  # 1000 mK
                ("K", float("inf"), 1),
            ],
            "none": [
                ("u", 1e-3, 1e6),
                ("m", 1e-1, 1e3),
                ("", 1e3, 1),
                ("k", 1e6, 1e-3),
                ("M", 1e9, 1e-6),
                ("G", 1e12, 1e-9),
                ("T", float("inf"), 1e-12),
            ],
        }

        united_value = None
        if value is not None:
            for unit_suffix, max_value, scale in UNIT_RANGES[unit]:
                if abs(value) <= max_value:
                    united_value = f"{value * scale:.3f}{unit_suffix}"
                    break

        base_name = datetime.now().strftime("%m%d%H")
        if united_value is not None:
            base_name += f"_{united_value}"

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
