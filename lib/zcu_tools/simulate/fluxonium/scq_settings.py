from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def scq_progress(progress: bool) -> Iterator[None]:
    """Temporarily set the scqubits progress-bar switch for one operation."""

    import scqubits.settings as scq_settings

    old = scq_settings.PROGRESSBAR_DISABLED
    scq_settings.PROGRESSBAR_DISABLED = not progress
    try:
        yield
    finally:
        scq_settings.PROGRESSBAR_DISABLED = old


@contextmanager
def scq_t1_default_warning(enabled: bool) -> Iterator[None]:
    """Temporarily set the scqubits T1 default-warning switch for one operation."""

    import scqubits.settings as scq_settings

    old = scq_settings.T1_DEFAULT_WARNING
    scq_settings.T1_DEFAULT_WARNING = enabled
    try:
        yield
    finally:
        scq_settings.T1_DEFAULT_WARNING = old
