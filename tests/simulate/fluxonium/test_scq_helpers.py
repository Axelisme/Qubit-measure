from __future__ import annotations

import pytest
import scqubits.settings as scq_settings
from zcu_tools.simulate.fluxonium import DressedLabelingError
from zcu_tools.simulate.fluxonium.dressed import require_dressed_index
from zcu_tools.simulate.fluxonium.scq_settings import (
    scq_progress,
    scq_t1_default_warning,
)


def test_scq_progress_restores_after_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scq_settings, "PROGRESSBAR_DISABLED", False)

    with pytest.raises(RuntimeError, match="boom"):
        with scq_progress(progress=False):
            assert scq_settings.PROGRESSBAR_DISABLED is True
            raise RuntimeError("boom")

    assert scq_settings.PROGRESSBAR_DISABLED is False


def test_scq_t1_warning_restores_after_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(scq_settings, "T1_DEFAULT_WARNING", True)

    with pytest.raises(RuntimeError, match="boom"):
        with scq_t1_default_warning(enabled=False):
            assert scq_settings.T1_DEFAULT_WARNING is False
            raise RuntimeError("boom")

    assert scq_settings.T1_DEFAULT_WARNING is True


def test_require_dressed_index_returns_int() -> None:
    assert require_dressed_index(3, (0, 1), context="flux=0.5") == 3


def test_require_dressed_index_raises_contextual_error() -> None:
    with pytest.raises(DressedLabelingError, match=r"\(0, 1\).*flux=0.5"):
        require_dressed_index(None, (0, 1), context="flux=0.5")
