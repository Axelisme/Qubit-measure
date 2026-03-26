from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from typing_extensions import ClassVar, Optional, Union


class ArbWaveformDatabase:
    """Manages arbitrary waveform data stored in per-waveform subdirectories.

    Directory structure::

        <database_path>/
            <waveform_name>/
                data.npz      # idata, qdata, time arrays
                example.png   # preview plot

    Each data.npz contains:
      - idata: NDArray with max abs value normalized to 1
      - qdata: NDArray with max abs value normalized to 1 (all zeros if not used)
      - time:  NDArray in microseconds (monotonically increasing)

    The database path must be initialized via `init()` before any access.
    """

    _database_path: ClassVar[Optional[Path]] = None

    @classmethod
    def init(cls, path: Union[str, Path]) -> None:
        """Set and create the database directory for arbitrary waveforms."""
        cls._database_path = Path(path)
        cls._database_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _check_initialized(cls) -> Path:
        if cls._database_path is None:
            raise RuntimeError(
                "ArbWaveformDatabase is not initialized. "
                "Call ArbWaveformDatabase.init(path) first."
            )
        return cls._database_path

    @classmethod
    def get(cls, name: str) -> tuple[NDArray, Optional[NDArray], NDArray]:
        """Load a waveform by name.

        Returns
        -------
        idata : NDArray
            In-phase data, max abs value normalized to 1.
        qdata : NDArray or None
            Quadrature data (None if all zeros in file).
        time : NDArray
            Time axis in microseconds.
        """
        db_path = cls._check_initialized()
        filepath = db_path / name / "data.npz"

        if not filepath.exists():
            available = cls.list()
            raise FileNotFoundError(
                f"Arbitrary waveform '{name}' not found at {filepath}. "
                f"Available waveforms: {available}"
            )

        data = np.load(filepath)
        idata: NDArray = data["idata"]
        qdata: NDArray = data["qdata"]
        time: NDArray = data["time"]

        qdata_out: Optional[NDArray] = None if np.all(qdata == 0) else qdata

        return idata, qdata_out, time

    @classmethod
    def save(
        cls,
        name: str,
        idata: NDArray,
        time: NDArray,
        qdata: Optional[NDArray] = None,
    ) -> None:
        """Save a waveform to the database.

        Creates a subdirectory named `name` containing data.npz and example.png.

        Parameters
        ----------
        name : str
            Waveform name (used as subdirectory name).
        idata : NDArray
            In-phase data, max abs value must be <= 1.
        time : NDArray
            Time axis in microseconds, must be monotonically increasing.
        qdata : NDArray or None
            Quadrature data, max abs value must be <= 1.
            If None, stored as all zeros.
        """
        db_path = cls._check_initialized()

        if np.max(np.abs(idata)) > 1.0:
            raise ValueError(
                f"idata max abs value ({np.max(np.abs(idata)):.4f}) exceeds 1.0"
            )
        if qdata is not None and np.max(np.abs(qdata)) > 1.0:
            raise ValueError(
                f"qdata max abs value ({np.max(np.abs(qdata)):.4f}) exceeds 1.0"
            )
        if not np.all(np.diff(time) > 0):
            raise ValueError("time array must be monotonically increasing")

        wav_dir = db_path / name
        wav_dir.mkdir(parents=True, exist_ok=True)

        qdata_save = qdata if qdata is not None else np.zeros_like(idata)
        np.savez(wav_dir / "data.npz", idata=idata, qdata=qdata_save, time=time)
        cls._save_preview(wav_dir / "example.png", name, idata, qdata, time)

    @classmethod
    def list(cls) -> list[str]:
        """List all available waveform names in the database."""
        db_path = cls._check_initialized()
        return sorted(
            d.name
            for d in db_path.iterdir()
            if d.is_dir() and (d / "data.npz").exists()
        )

    @staticmethod
    def _save_preview(
        path: Path,
        name: str,
        idata: NDArray,
        qdata: Optional[NDArray],
        time: NDArray,
    ) -> None:
        fig, ax = plt.subplots(figsize=(8, 2.5))

        ax.plot(time, idata, label="I")
        if qdata is not None:
            ax.plot(time, qdata, label="Q")

        ax.legend()
        ax.set_title(name)
        ax.set_xlabel("Time (us)")
        ax.set_ylabel("Amplitude (norm.)")
        ax.set_xlim(time[0], time[-1])
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
