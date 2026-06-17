"""Tests for the standardized twotone analyze-result summary shapes (Proposal 3).

The analyze summary is ``AnalyzeResultBase.to_summary_dict()`` — a reflection over
the AnalyzeResult dataclass's JSON-safe fields (the Figure is skipped). These tests
pin the wire/mcp-facing summary keys:

  - twotone/freq carries freq + freq_err and fwhm + fwhm_err (the fit uncertainties
    the figure title already shows), alongside the json-safe empty ``params`` dict.
  - twotone/rabi/len_rabi carries pi_len/pi2_len/rabi_f each with its *_err.
  - twotone/rabi/amp_rabi's scalar summary keys are pi_gain/pi2_gain (matching the
    MetaDict writeback target + glossary), each with its *_err — NOT pi_amp/pi2_amp
    (those names belong to the pi-pulse MODULE writeback, not the scalar summary).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from zcu_tools.experiment.v2_gui.adapters.twotone.freq import FreqAnalyzeResult
from zcu_tools.experiment.v2_gui.adapters.twotone.rabi.amp_rabi import (
    AmpRabiAnalyzeResult,
)
from zcu_tools.experiment.v2_gui.adapters.twotone.rabi.len_rabi import (
    LenRabiAnalyzeResult,
)


def test_freq_summary_includes_freq_err_and_fwhm_err() -> None:
    result = FreqAnalyzeResult(
        freq=3000.0,
        freq_err=0.9,
        fwhm=2.5,
        fwhm_err=0.3,
        params={},
        figure=MagicMock(),
    )
    summary = result.to_summary_dict()
    # The Figure is not json-safe and is dropped; every other field is kept.
    assert summary == {
        "freq": 3000.0,
        "freq_err": 0.9,
        "fwhm": 2.5,
        "fwhm_err": 0.3,
        "params": {},
    }


def test_len_rabi_summary_includes_per_quantity_err() -> None:
    result = LenRabiAnalyzeResult(
        pi_len=0.05,
        pi_len_err=0.001,
        pi2_len=0.025,
        pi2_len_err=0.001,
        rabi_f=10.0,
        rabi_f_err=0.1,
        figure=MagicMock(),
    )
    summary = result.to_summary_dict()
    assert summary == {
        "pi_len": 0.05,
        "pi_len_err": 0.001,
        "pi2_len": 0.025,
        "pi2_len_err": 0.001,
        "rabi_f": 10.0,
        "rabi_f_err": 0.1,
    }


def test_amp_rabi_summary_uses_gain_keys_with_err() -> None:
    result = AmpRabiAnalyzeResult(
        pi_gain=0.4,
        pi_gain_err=0.01,
        pi2_gain=0.2,
        pi2_gain_err=0.01,
        figure=MagicMock(),
    )
    summary = result.to_summary_dict()
    # Scalar gains surface as pi_gain/pi2_gain (writeback target), never pi_amp.
    assert summary == {
        "pi_gain": 0.4,
        "pi_gain_err": 0.01,
        "pi2_gain": 0.2,
        "pi2_gain_err": 0.01,
    }
    assert "pi_amp" not in summary
    assert "pi2_amp" not in summary
