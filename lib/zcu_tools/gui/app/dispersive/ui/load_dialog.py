"""One-tone source selection over the shared native/Labber chooser."""

from __future__ import annotations

from typing import NamedTuple

from qtpy.QtWidgets import QWidget

from zcu_tools.gui.app.analysis_project.native import NativeSpectrumSelection
from zcu_tools.gui.app.analysis_project.native_dialog import (
    NativeDeviceChoices,
    NativeSpectrumDialog,
)


class LoadOnetoneRequest(NamedTuple):
    """Exact source path, Labber-only transpose and explicit native selection."""

    filepath: str
    transpose_axes: bool
    selection: NativeSpectrumSelection | None = None


class LoadOnetoneDialog(NativeSpectrumDialog):
    """Offer predictor-provided devices without reading a predictor or component."""

    def __init__(
        self,
        parent: QWidget | None = None,
        start_dir: str = "",
        *,
        choices: NativeDeviceChoices | None = None,
    ) -> None:
        """Preview one-tone sources with caller-supplied predictor device choices.

        Without choices native inputs can only be unnamed_device. start_dir
        controls Browse; it never chooses a file or infers an instrument.
        """
        super().__init__(choices or NativeDeviceChoices({}), parent, start_dir)
        self.setWindowTitle("Load one-tone spectrum")

    def result_request(self) -> LoadOnetoneRequest | None:
        """Return explicit source choices, or None without path/required fields."""
        if not self._filepath:
            return None
        selection = self.native_selection()
        if self.is_native_source and selection is None:
            return None
        return LoadOnetoneRequest(
            self._filepath,
            self._transpose if not self.is_native_source else False,
            selection,
        )
