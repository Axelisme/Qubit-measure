"""Shared writeback-item builders (avoid per-adapter copy-paste)."""

from __future__ import annotations

from typing import Sequence

from zcu_tools.gui.adapter import MetaDictWriteback, WritebackItem


def make_onetone_freq_writeback_items(
    freq: float, fwhm: float
) -> Sequence[WritebackItem]:
    """Writeback items for a one-tone resonator-frequency fit (real + fake share).

    Proposes only the fitted resonator frequency / linewidth into the MetaDict
    (r_f / rf_w). The readout module + probe waveform are left to the user — the
    fit result alone does not justify rewriting the whole readout config.
    """
    return [
        MetaDictWriteback(
            target_name="r_f",
            description="Resonator frequency (MHz)",
            proposed_value=freq,
        ),
        MetaDictWriteback(
            target_name="rf_w",
            description="Resonator linewidth FWHM (MHz)",
            proposed_value=fwhm,
        ),
    ]
